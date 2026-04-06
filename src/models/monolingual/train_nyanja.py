"""
Train a monolingual Wav2Vec2 CTC model on Nyanja.

Usage:
    python -m src.models.monolingual.train_nyanja \\
        [--config configs/nyanja_config.yaml] \\
        [--output_dir outputs/nyanja] \\
        [--dataset_path outputs/nyanja/dataset] \\
        [--vocab_path outputs/nyanja/vocab.json]

Before running this script, prepare the dataset with:
    python -m src.data_preparation.prepare_dataset \\
        --config configs/nyanja_config.yaml
"""

import argparse
import logging
import os

import yaml

from src.models.trainer import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/nyanja_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train monolingual Wav2Vec2 CTC model on Nyanja."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to the Nyanja YAML config file.",
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

    output_dir = args.output_dir or config.get("output_dir", "outputs/nyanja")
    dataset_path = args.dataset_path or os.path.join(output_dir, "dataset")
    vocab_path = args.vocab_path or os.path.join(output_dir, "vocab.json")

    logger.info("=== Nyanja Monolingual ASR Training ===")
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
