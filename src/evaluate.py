"""Evaluation metrics and figure generation for sentiment models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.clean import clean_series
from src.load import LABEL_NAMES, PROJECT_ROOT

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def evaluate_model(pipeline, texts, labels, model_name: str = "model") -> Dict[str, Any]:
    X = clean_series(texts)
    y_true = np.asarray(labels, dtype=int)
    y_pred = pipeline.predict(X)
    labels_present = sorted(set(y_true) | set(y_pred))
    target_names = [LABEL_NAMES[i] for i in labels_present]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_present,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "n_samples": int(len(y_true)),
        "labels": [LABEL_NAMES[i] for i in labels_present],
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    return metrics


def plot_confusion_matrix_svg(
    cm: List[List[int]],
    labels: List[str],
    title: str,
    out_path: Path,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    arr = np.asarray(cm)
    im = ax.imshow(arr, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True",
        xlabel="Predicted",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = arr.max() / 2.0 if arr.size else 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(
                j,
                i,
                format(arr[i, j], "d"),
                ha="center",
                va="center",
                color="white" if arr[i, j] > thresh else "black",
            )
    fig.tight_layout()
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".svg":
        out_path = out_path.with_suffix(".svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_metrics_bars_svg(all_metrics: List[Dict[str, Any]], out_path: Path) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    names = [m["model"] for m in all_metrics]
    acc = [m["accuracy"] for m in all_metrics]
    f1m = [m["f1_macro"] for m in all_metrics]
    f1w = [m["f1_weighted"] for m in all_metrics]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width, acc, width, label="Accuracy")
    ax.bar(x, f1m, width, label="F1 macro")
    ax.bar(x + width, f1w, width, label="F1 weighted")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Sentiment Model Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path).with_suffix(".svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_topic_words_svg(topics: List[Dict[str, Any]], out_path: Path, max_topics: int = 6) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    topics = topics[:max_topics]
    n = len(topics)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.8 * n), squeeze=False)
    for ax, topic in zip(axes[:, 0], topics):
        words = topic["top_words"][:8][::-1]
        weights = topic["weights"][:8][::-1]
        ax.barh(words, weights, color="#4C78A8")
        ax.set_title(topic["label"], fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Top Words per Topic (NMF)", fontsize=12, y=1.01)
    fig.tight_layout()
    out_path = Path(out_path).with_suffix(".svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_sentiment_distribution_svg(sentiments, out_path: Path) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    from collections import Counter

    counts = Counter(sentiments)
    labels = ["negative", "neutral", "positive"]
    values = [counts.get(l, 0) for l in labels]
    colors = ["#E45756", "#F58518", "#54A24B"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=colors)
    ax.set_title("Sentiment Distribution (sample)")
    ax.set_ylabel("Count")
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.01, str(v), ha="center")
    fig.tight_layout()
    out_path = Path(out_path).with_suffix(".svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_metrics(metrics: Dict[str, Any], path: Optional[Path] = None) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = path or (REPORTS_DIR / "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path
