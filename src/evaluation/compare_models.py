"""
Compare monolingual and multilingual ASR model performance across
Nyanja, Tonga, and Bemba languages.

Loads evaluation metrics from JSON files produced by evaluate.py, builds
a comparison table, and generates bar charts saved as PNG files.

Usage:
    python -m src.evaluation.compare_models \\
        --results_dir outputs \\
        [--output_dir outputs/comparison]

Expected directory layout in results_dir:
    outputs/
        nyanja/evaluation/nyanja_metrics.json
        tonga/evaluation/tonga_metrics.json
        bemba/evaluation/bemba_metrics.json
        multilingual/evaluation/nyanja_metrics.json   (per-language results)
        multilingual/evaluation/tonga_metrics.json
        multilingual/evaluation/bemba_metrics.json
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LANGUAGES = ["nyanja", "tonga", "bemba"]
MODEL_TYPES = ["monolingual", "multilingual"]
METRICS = ["wer", "cer"]


def load_metrics_file(path: str) -> Optional[Dict]:
    """
    Load a JSON metrics file.

    Args:
        path: Path to the metrics JSON file.

    Returns:
        Dict of metrics, or None if the file does not exist.
    """
    if not os.path.exists(path):
        logger.warning("Metrics file not found: %s", path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_results(results_dir: str) -> pd.DataFrame:
    """
    Collect evaluation results for all models and languages.

    Searches for metrics files following the expected directory convention
    described in this module's docstring.

    Args:
        results_dir: Root output directory.

    Returns:
        DataFrame with columns: model_type, language, wer, cer.
    """
    rows = []

    for lang in LANGUAGES:
        monolingual_path = os.path.join(
            results_dir, lang, "evaluation", f"{lang}_metrics.json"
        )
        metrics = load_metrics_file(monolingual_path)
        if metrics:
            rows.append(
                {
                    "model_type": "monolingual",
                    "language": lang,
                    "wer": metrics.get("wer"),
                    "cer": metrics.get("cer"),
                }
            )

        multilingual_path = os.path.join(
            results_dir, "multilingual", "evaluation", f"{lang}_metrics.json"
        )
        metrics = load_metrics_file(multilingual_path)
        if metrics:
            rows.append(
                {
                    "model_type": "multilingual",
                    "language": lang,
                    "wer": metrics.get("wer"),
                    "cer": metrics.get("cer"),
                }
            )

    if not rows:
        raise ValueError(
            f"No evaluation metrics found in '{results_dir}'. "
            "Run evaluate.py for each model first."
        )

    return pd.DataFrame(rows)


def print_comparison_table(df: pd.DataFrame) -> None:
    """
    Print a formatted comparison table to stdout.

    Args:
        df: DataFrame with model comparison results.
    """
    pivot = df.pivot_table(
        index="language",
        columns="model_type",
        values=["wer", "cer"],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{model}" for metric, model in pivot.columns]
    pivot = pivot.reset_index()
    pivot = pivot.sort_values("language")

    col_order = ["language"]
    for metric in METRICS:
        for model in MODEL_TYPES:
            col = f"{metric}_{model}"
            if col in pivot.columns:
                col_order.append(col)

    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON: Monolingual vs Multilingual")
    print("  Metrics: WER (Word Error Rate) | CER (Character Error Rate)")
    print("=" * 60)
    print(pivot.to_string(index=False, float_format="{:.4f}".format))
    print("=" * 60 + "\n")


def plot_wer_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate and save a grouped bar chart comparing WER across languages
    and model types.

    Args:
        df: DataFrame with model comparison results.
        output_dir: Directory to save the PNG chart.
    """
    wer_df = df.pivot_table(
        index="language", columns="model_type", values="wer"
    ).reindex(LANGUAGES)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(LANGUAGES))
    width = 0.35
    colors = ["#2196F3", "#FF9800"]

    for i, model_type in enumerate(MODEL_TYPES):
        if model_type in wer_df.columns:
            bars = ax.bar(
                x + (i - 0.5) * width,
                wer_df[model_type],
                width,
                label=f"{model_type.capitalize()} Model",
                color=colors[i],
                alpha=0.85,
            )
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("Word Error Rate (WER)", fontsize=12)
    ax.set_title(
        "WER Comparison: Monolingual vs Multilingual ASR\n"
        "Zambian Languages (Nyanja, Tonga, Bemba)",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([lang.capitalize() for lang in LANGUAGES], fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(wer_df.values.max() * 1.25, 0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(output_dir, "wer_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved WER comparison chart to %s", path)


def plot_cer_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate and save a grouped bar chart comparing CER across languages
    and model types.

    Args:
        df: DataFrame with model comparison results.
        output_dir: Directory to save the PNG chart.
    """
    cer_df = df.pivot_table(
        index="language", columns="model_type", values="cer"
    ).reindex(LANGUAGES)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(LANGUAGES))
    width = 0.35
    colors = ["#4CAF50", "#F44336"]

    for i, model_type in enumerate(MODEL_TYPES):
        if model_type in cer_df.columns:
            bars = ax.bar(
                x + (i - 0.5) * width,
                cer_df[model_type],
                width,
                label=f"{model_type.capitalize()} Model",
                color=colors[i],
                alpha=0.85,
            )
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("Character Error Rate (CER)", fontsize=12)
    ax.set_title(
        "CER Comparison: Monolingual vs Multilingual ASR\n"
        "Zambian Languages (Nyanja, Tonga, Bemba)",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([lang.capitalize() for lang in LANGUAGES], fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(cer_df.values.max() * 1.25, 0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(output_dir, "cer_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved CER comparison chart to %s", path)


def plot_combined_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate a heatmap showing WER and CER for all model-language combinations.

    Args:
        df: DataFrame with model comparison results.
        output_dir: Directory to save the PNG heatmap.
    """
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "Model": f"{row['model_type'].capitalize()}\n({row['language'].capitalize()})",
                "WER": row["wer"],
                "CER": row["cer"],
            }
        )
    heat_df = pd.DataFrame(rows).set_index("Model")

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(heat_df) * 0.5 + 1)))
    for ax, metric in zip(axes, ["WER", "CER"]):
        data = heat_df[[metric]]
        im = ax.imshow(data.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks([0])
        ax.set_xticklabels([metric])
        ax.set_yticks(range(len(heat_df)))
        ax.set_yticklabels(heat_df.index, fontsize=9)
        for i, val in enumerate(data.values.flatten()):
            ax.text(0, i, f"{val:.3f}", ha="center", va="center", fontsize=10)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{metric} by Model & Language", fontsize=11)

    plt.suptitle(
        "ASR Performance Heatmap\nNyanja, Tonga & Bemba — Monolingual vs Multilingual",
        fontsize=12,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "performance_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved performance heatmap to %s", path)


def compare_models(results_dir: str, output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Run the full model comparison pipeline.

    Collects metrics, prints a comparison table, and generates charts.

    Args:
        results_dir: Root output directory containing per-model results.
        output_dir: Directory to save charts and summary CSV.

    Returns:
        DataFrame with all model comparison results.
    """
    output_dir = output_dir or os.path.join(results_dir, "comparison")
    os.makedirs(output_dir, exist_ok=True)

    df = collect_results(results_dir)
    print_comparison_table(df)

    plot_wer_comparison(df, output_dir)
    plot_cer_comparison(df, output_dir)
    plot_combined_heatmap(df, output_dir)

    csv_path = os.path.join(output_dir, "comparison_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Saved comparison results to %s", csv_path)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare monolingual and multilingual ASR models."
    )
    parser.add_argument(
        "--results_dir",
        default="outputs",
        help="Root directory containing per-model evaluation results.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save comparison charts and CSV.",
    )
    args = parser.parse_args()
    compare_models(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
