"""
Evaluate a trained Wav2Vec2 CTC ASR model on a test set.

Computes Word Error Rate (WER) and Character Error Rate (CER) and
saves per-sample predictions alongside aggregate metrics.

Usage:
    python -m src.evaluation.evaluate \\
        --model_dir outputs/nyanja \\
        --dataset_path outputs/nyanja/dataset \\
        --split test \\
        [--language nyanja] \\
        [--output_dir outputs/nyanja/evaluation]
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
from datasets import load_from_disk
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.utils.metrics import compute_all_metrics, normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ASR model.")
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Directory containing the fine-tuned model and processor.",
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to the preprocessed DatasetDict on disk.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language label for output file naming (e.g. 'nyanja').",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write evaluation results. Defaults to <model_dir>/evaluation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Inference batch size.",
    )
    return parser.parse_args()


def transcribe_batch(
    batch_audio: list,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    device: torch.device,
) -> list:
    """
    Run inference on a batch of audio arrays.

    Args:
        batch_audio: List of dicts with 'array' and 'sampling_rate' keys.
        processor: Wav2Vec2Processor for feature extraction and decoding.
        model: Fine-tuned Wav2Vec2ForCTC model.
        device: Torch device.

    Returns:
        List of decoded transcription strings.
    """
    inputs = processor(
        [a["array"] for a in batch_audio],
        sampling_rate=batch_audio[0]["sampling_rate"],
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)


def evaluate_model(
    model_dir: str,
    dataset_path: str,
    split: str = "test",
    language: str = None,
    output_dir: str = None,
    batch_size: int = 8,
) -> dict:
    """
    Run full evaluation of a trained ASR model and save results.

    Args:
        model_dir: Directory with the fine-tuned model and processor.
        dataset_path: Path to the DatasetDict on disk.
        split: Dataset split to evaluate ('train', 'validation', or 'test').
        language: Language label for naming output files.
        output_dir: Directory to save evaluation results.
        batch_size: Inference batch size.

    Returns:
        Dictionary with 'wer', 'cer', and 'predictions' keys.
    """
    output_dir = output_dir or os.path.join(model_dir, "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    logger.info("Loading processor and model from %s", model_dir)
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device)
    model.eval()

    logger.info("Loading dataset from %s (split: %s)", dataset_path, split)
    dataset = load_from_disk(dataset_path)
    split_data = dataset[split]

    all_predictions = []
    all_references = []

    for i in range(0, len(split_data), batch_size):
        batch = split_data[i : i + batch_size]
        audio_batch = batch["audio"] if isinstance(batch["audio"], list) else [batch["audio"]]
        ref_batch = batch["transcription"] if isinstance(batch["transcription"], list) else [batch["transcription"]]
        preds = transcribe_batch(audio_batch, processor, model, device)
        all_predictions.extend(preds)
        all_references.extend(ref_batch)
        if (i // batch_size) % 10 == 0:
            logger.info("Processed %d / %d samples", min(i + batch_size, len(split_data)), len(split_data))

    metrics = compute_all_metrics(all_predictions, all_references)
    label = language or "model"
    logger.info(
        "[%s] WER: %.4f | CER: %.4f", label, metrics["wer"], metrics["cer"]
    )

    per_sample = [
        {
            "reference": ref,
            "prediction": pred,
            "wer": compute_all_metrics([pred], [ref])["wer"],
        }
        for ref, pred in zip(all_references, all_predictions)
    ]

    prefix = f"{language}_" if language else ""
    metrics_path = os.path.join(output_dir, f"{prefix}metrics.json")
    predictions_path = os.path.join(output_dir, f"{prefix}predictions.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(per_sample, f, ensure_ascii=False, indent=2)

    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved predictions to %s", predictions_path)

    return {**metrics, "predictions": per_sample}


def main() -> None:
    args = parse_args()
    evaluate_model(
        model_dir=args.model_dir,
        dataset_path=args.dataset_path,
        split=args.split,
        language=args.language,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
