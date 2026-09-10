#!/usr/bin/env python3
"""Train sentiment + topic models, evaluate, and write reports/figures."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix_svg,
    plot_metrics_bars_svg,
    plot_sentiment_distribution_svg,
    plot_topic_words_svg,
    save_metrics,
)
from src.load import load_reviews, train_test_split_reviews
from src.topics import fit_lda_topics, fit_nmf_topics, save_topic_artifacts, topic_diversity
from src.train_sentiment import model_summary, save_model, train_models


def main():
    print("Loading reviews...")
    df = load_reviews(prefer_full=True)
    print(f"Loaded {len(df)} reviews")
    print(df["sentiment"].value_counts().to_string())

    train_df, test_df = train_test_split_reviews(df, test_size=0.2, random_state=42)
    print(f"Train={len(train_df)} Test={len(test_df)}")

    print("\nTraining sentiment models...")
    models = train_models(train_df["text"], train_df["label"])

    all_metrics = []
    for name, pipe in models.items():
        print(f"Evaluating {name}...")
        m = evaluate_model(pipe, test_df["text"], test_df["label"], model_name=name)
        all_metrics.append(m)
        plot_confusion_matrix_svg(
            m["confusion_matrix"],
            m["labels"],
            title=f"Confusion Matrix — {name}",
            out_path=ROOT / "reports" / "figures" / f"cm_{name}.svg",
        )
        print(
            f"  accuracy={m['accuracy']:.4f}  f1_macro={m['f1_macro']:.4f}  "
            f"f1_weighted={m['f1_weighted']:.4f}"
        )

    best = max(all_metrics, key=lambda x: x["f1_macro"])
    best_name = best["model"]
    best_pipe = models[best_name]
    model_path = save_model(best_pipe, name="sentiment_pipeline")
    save_model(models["tfidf_logreg"], name="tfidf_logreg")
    save_model(models["tfidf_linearsvc"], name="tfidf_linearsvc")

    b64_path = ROOT / "models" / "sentiment_pipeline.joblib.b64"
    with open(model_path, "rb") as f:
        b64_path.write_text(base64.b64encode(f.read()).decode("ascii"), encoding="utf-8")
    print(f"Saved best model ({best_name}) -> {model_path}")
    print(f"Saved b64 sidecar -> {b64_path}")

    plot_metrics_bars_svg(all_metrics, ROOT / "reports" / "figures" / "model_comparison.svg")
    plot_sentiment_distribution_svg(
        df["sentiment"], ROOT / "reports" / "figures" / "sentiment_distribution.svg"
    )

    print("\nFitting NMF topics...")
    nmf = fit_nmf_topics(train_df["text"], n_topics=8)
    save_topic_artifacts(nmf, prefix="nmf")
    plot_topic_words_svg(nmf["topics"], ROOT / "reports" / "figures" / "nmf_topics.svg")
    print(f"  topic_diversity={topic_diversity(nmf['topics']):.3f}")
    for t in nmf["topics"]:
        print(f"  {t['label']}")

    print("\nFitting LDA topics...")
    lda = fit_lda_topics(train_df["text"], n_topics=8)
    save_topic_artifacts(lda, prefix="lda")
    print(f"  perplexity={lda['perplexity']:.2f} diversity={topic_diversity(lda['topics']):.3f}")

    report = {
        "dataset": {
            "n_total": int(len(df)),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "sentiment_counts": df["sentiment"].value_counts().to_dict(),
        },
        "models": all_metrics,
        "best_model": best_name,
        "best_model_summary": model_summary(best_pipe),
        "topics_nmf": {
            "n_topics": nmf["n_topics"],
            "topic_diversity": topic_diversity(nmf["topics"]),
            "reconstruction_err": nmf["reconstruction_err"],
            "topics": [
                {"id": t["id"], "label": t["label"], "top_words": t["top_words"]}
                for t in nmf["topics"]
            ],
        },
        "topics_lda": {
            "n_topics": lda["n_topics"],
            "topic_diversity": topic_diversity(lda["topics"]),
            "perplexity": lda["perplexity"],
            "topics": [
                {"id": t["id"], "label": t["label"], "top_words": t["top_words"]}
                for t in lda["topics"]
            ],
        },
    }
    metrics_path = save_metrics(report)
    print(f"\nWrote metrics -> {metrics_path}")
    print("Done.")


if __name__ == "__main__":
    main()
