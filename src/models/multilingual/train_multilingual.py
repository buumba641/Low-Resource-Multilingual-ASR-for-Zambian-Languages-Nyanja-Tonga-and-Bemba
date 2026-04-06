"""
Train a multilingual Wav2Vec2 CTC model on Nyanja, Tonga, and Bemba combined.

The multilingual model is trained on all three languages simultaneously using
a shared character vocabulary. This allows direct comparison with the three
monolingual models.

Usage:
    python -m src.models.multilingual.train_multilingual \\
        [--config configs/multilingual_config.yaml] \\
        [--output_dir outputs/multilingual] \\
        [--dataset_path outputs/multilingual/dataset] \\
        [--vocab_path outputs/multilingual/vocab.json]

Before running this script, prepare the multilingual dataset with:
    python -m src.data_preparation.prepare_dataset \\
        --config configs/multilingual_config.yaml \\
        --multilingual
"""

import argparse
import logging
import os

import yaml

from src.models.trainer import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/multilingual_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multilingual Wav2Vec2 CTC model on Nyanja, Tonga, and Bemba."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to the multilingual YAML config file.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override the output directory from the config.",
    )
    parser.add_argument(
        "--dataset_path",
        default=None,
        help="Override the path to the preprocessed DatasetDict.",
    )
    parser.add_argument(
        "--vocab_path",
        default=None,
        help="Override the path to vocab.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = args.output_dir or config.get("output_dir", "outputs/multilingual")
    dataset_path = args.dataset_path or os.path.join(output_dir, "dataset")
    vocab_path = args.vocab_path or os.path.join(output_dir, "vocab.json")

    logger.info("=== Multilingual ASR Training (Nyanja + Tonga + Bemba) ===")
    logger.info("Config:       %s", args.config)
    logger.info("Output dir:   %s", output_dir)
    logger.info("Dataset path: %s", dataset_path)
    logger.info("Vocab path:   %s", vocab_path)

    train(
        config=config,
        output_dir=output_dir,
        dataset_path=dataset_path,
        vocab_path=vocab_path,
    )


if __name__ == "__main__":
    main()
