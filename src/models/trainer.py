"""
Shared ASR trainer base for both monolingual and multilingual Wav2Vec2 CTC models.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

import jiwer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that dynamically pads audio inputs and label sequences
    for CTC training. Padding in labels uses -100 so that it is ignored
    by the CTC loss.
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_values": f["input_values"]} for f in features
        ]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Processor / model helpers
# ---------------------------------------------------------------------------

def build_processor(vocab_path: str, sampling_rate: int = 16000) -> Wav2Vec2Processor:
    """
    Build a Wav2Vec2Processor from a local vocabulary file.

    Args:
        vocab_path: Path to the vocab.json file.
        sampling_rate: Target audio sample rate.

    Returns:
        Wav2Vec2Processor ready for training.
    """
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=sampling_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    return processor


def build_model(
    vocab_size: int,
    model_cfg: Dict[str, Any],
) -> Wav2Vec2ForCTC:
    """
    Instantiate a Wav2Vec2ForCTC model from a pre-trained checkpoint.

    The classification head (lm_head) is re-initialised to match the
    vocabulary size of the target language(s).

    Args:
        vocab_size: Size of the target vocabulary.
        model_cfg: Model configuration dict from the YAML config.

    Returns:
        Wav2Vec2ForCTC model ready for fine-tuning.
    """
    model = Wav2Vec2ForCTC.from_pretrained(
        model_cfg["pretrained_model_name"],
        attention_dropout=model_cfg.get("attention_dropout", 0.1),
        hidden_dropout=model_cfg.get("hidden_dropout", 0.1),
        feat_proj_dropout=model_cfg.get("feat_proj_dropout", 0.0),
        mask_time_prob=model_cfg.get("mask_time_prob", 0.05),
        layerdrop=model_cfg.get("layerdrop", 0.1),
        ctc_loss_reduction=model_cfg.get("ctc_loss_reduction", "mean"),
        pad_token_id=model_cfg.get("pad_token_id", 0),
        vocab_size=vocab_size,
    )
    # Freeze the feature encoder to avoid training convolutional layers
    model.freeze_feature_encoder()
    return model


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_dataset_features(
    dataset: DatasetDict,
    processor: Wav2Vec2Processor,
    sampling_rate: int = 16000,
) -> DatasetDict:
    """
    Apply feature extraction and label encoding to all dataset splits.

    Args:
        dataset: DatasetDict with 'train', 'validation', and 'test' splits.
        processor: Wav2Vec2Processor used for feature extraction and tokenisation.
        sampling_rate: Target audio sample rate.

    Returns:
        DatasetDict with 'input_values' and 'labels' columns added.
    """

    def prepare_batch(batch: Dict) -> Dict:
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        with processor.as_target_processor():
            batch["labels"] = processor(batch["transcription"]).input_ids
        return batch

    return dataset.map(
        prepare_batch,
        remove_columns=dataset["train"].column_names,
        num_proc=1,
    )


# ---------------------------------------------------------------------------
# Metrics callback
# ---------------------------------------------------------------------------

def make_compute_metrics(processor: Wav2Vec2Processor):
    """
    Return a compute_metrics function compatible with Hugging Face Trainer.

    Computes Word Error Rate (WER) on the validation set.

    Args:
        processor: Wav2Vec2Processor used to decode predicted token IDs.

    Returns:
        A callable that accepts EvalPrediction and returns a metrics dict.
    """

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = jiwer.wer(label_str, pred_str)
        return {"wer": wer}

    return compute_metrics


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(
    config: Dict[str, Any],
    output_dir: str,
    dataset_path: str,
    vocab_path: str,
) -> None:
    """
    Fine-tune a Wav2Vec2ForCTC model using the given config.

    Args:
        config: Full configuration dictionary (from YAML).
        output_dir: Directory to save checkpoints and final model.
        dataset_path: Path to the preprocessed DatasetDict saved on disk.
        vocab_path: Path to the vocab.json file.
    """
    logger.info("Loading dataset from %s", dataset_path)
    raw_dataset = load_from_disk(dataset_path)

    processor = build_processor(vocab_path, sampling_rate=config["audio"]["sampling_rate"])
    processor.save_pretrained(output_dir)

    logger.info("Preparing dataset features...")
    dataset = prepare_dataset_features(
        raw_dataset, processor, sampling_rate=config["audio"]["sampling_rate"]
    )

    model = build_model(len(processor.tokenizer), config["model"])

    training_cfg = config["training"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        warmup_steps=training_cfg["warmup_steps"],
        weight_decay=training_cfg["weight_decay"],
        evaluation_strategy=training_cfg["evaluation_strategy"],
        save_strategy=training_cfg["save_strategy"],
        eval_steps=training_cfg["eval_steps"],
        save_steps=training_cfg["save_steps"],
        logging_steps=training_cfg["logging_steps"],
        load_best_model_at_end=training_cfg["load_best_model_at_end"],
        metric_for_best_model=training_cfg["metric_for_best_model"],
        greater_is_better=training_cfg["greater_is_better"],
        save_total_limit=training_cfg["save_total_limit"],
        fp16=training_cfg.get("fp16", False) and torch.cuda.is_available(),
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 4),
        seed=training_cfg.get("seed", 42),
        report_to=["tensorboard"],
        logging_dir=os.path.join(output_dir, "logs"),
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    compute_metrics = make_compute_metrics(processor)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor.feature_extractor,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving final model to %s", output_dir)
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    logger.info("Training complete.")
