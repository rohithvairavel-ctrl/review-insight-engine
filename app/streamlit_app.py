#!/usr/bin/env python3
"""Review Insight Engine — Streamlit dashboard for sentiment + topics."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load import LABEL_NAMES, load_reviews
from src.train_sentiment import load_model, predict_sentiment
from src.topics import assign_topic, load_topics

st.set_page_config(
    page_title="Review Insight Engine",
    page_icon="📝",
    layout="wide",
)


@st.cache_resource
def get_sentiment_model():
    model_path = ROOT / "models" / "sentiment_pipeline.joblib"
    if not model_path.exists():
        b64_path = ROOT / "models" / "sentiment_pipeline.joblib.b64"
        parts = sorted((ROOT / "models").glob("sentiment_pipeline.joblib.b64.part*"))
        if b64_path.exists():
            data = b64_path.read_text(encoding="utf-8")
        elif parts:
            data = "".join(p.read_text(encoding="utf-8") for p in parts)
        else:
            raise FileNotFoundError(model_path)
        model_path.write_bytes(base64.b64decode(data))
    return load_model("sentiment_pipeline")


@st.cache_resource
def get_topic_model():
    try:
        return load_topics("nmf")
    except FileNotFoundError:
        return None


@st.cache_data
def get_metrics():
    path = ROOT / "reports" / "metrics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


@st.cache_data
def get_data_sample(n: int = 500):
    try:
        df = load_reviews(prefer_full=False)
        return df.sample(n=min(n, len(df)), random_state=42)
    except FileNotFoundError:
        return None


def main():
    st.title("📝 Review Insight Engine")
    st.markdown(
        "End-to-end product review NLP: **sentiment classification**, "
        "**topic discovery**, and interactive insights."
    )

    metrics = get_metrics()
    model = None
    topics = None
    try:
        model = get_sentiment_model()
    except FileNotFoundError:
        st.warning("Sentiment model not found. Run `python scripts/train.py` first.")
    try:
        topics = get_topic_model()
    except Exception:
        topics = None

    tab_predict, tab_topics, tab_insights, tab_about = st.tabs(
        ["🔮 Predict", "🗂️ Topics", "📊 Insights", "ℹ️ About"]
    )

    with tab_predict:
        st.subheader("Paste a product review")
        default = (
            "I absolutely love this coffee maker. It heats up fast, tastes great, "
            "and cleanup is easy. Best purchase this year!"
        )
        text = st.text_area("Review text", value=default, height=140)
        col1, col2 = st.columns([1, 3])
        with col1:
            run = st.button("Analyze", type="primary")
        if run and model is not None:
            label, conf, details = predict_sentiment(model, text)
            emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}[label]
            st.success(f"{emoji} **Sentiment:** `{label}`  |  **Confidence:** `{conf:.1%}`")
            st.bar_chart(pd.Series(details, name="score"))
            if topics is not None and "model" in topics:
                tid, tlabel, words = assign_topic(text, topics)
                st.info(f"**Closest topic:** {tlabel}")
                st.caption("Top words: " + ", ".join(words[:10]))
        elif run:
            st.error("Model unavailable.")

    with tab_topics:
        st.subheader("Discovered themes (NMF)")
        if metrics and "topics_nmf" in metrics:
            for t in metrics["topics_nmf"]["topics"]:
                with st.expander(t["label"], expanded=False):
                    st.write(", ".join(t["top_words"]))
            st.caption(
                f"Topic diversity: {metrics['topics_nmf'].get('topic_diversity', 'n/a'):.3f}"
                if isinstance(metrics["topics_nmf"].get("topic_diversity"), float)
                else ""
            )
            fig_path = ROOT / "reports" / "figures" / "nmf_topics.svg"
            if fig_path.exists():
                st.image(str(fig_path))
        else:
            st.info("Train the pipeline to populate topics.")

        if metrics and "topics_lda" in metrics:
            st.subheader("LDA topics (comparison)")
            st.caption(f"Perplexity: {metrics['topics_lda'].get('perplexity', 'n/a'):.2f}")
            for t in metrics["topics_lda"]["topics"][:4]:
                st.markdown(f"- **{t['label']}** — {', '.join(t['top_words'][:8])}")

    with tab_insights:
        st.subheader("Model performance")
        if metrics:
            best = metrics.get("best_model")
            st.markdown(f"**Best model:** `{best}`")
            rows = []
            for m in metrics.get("models", []):
                rows.append(
                    {
                        "Model": m["model"],
                        "Accuracy": round(m["accuracy"], 4),
                        "F1 macro": round(m["f1_macro"], 4),
                        "F1 weighted": round(m["f1_weighted"], 4),
                        "N test": m["n_samples"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            for name in ("model_comparison.svg", "sentiment_distribution.svg"):
                p = ROOT / "reports" / "figures" / name
                if p.exists():
                    st.image(str(p))
            if best:
                cm = ROOT / "reports" / "figures" / f"cm_{best}.svg"
                if cm.exists():
                    st.image(str(cm))
        sample = get_data_sample()
        if sample is not None:
            st.subheader("Sample reviews")
            st.dataframe(
                sample[["score", "sentiment", "text"]].head(20),
                use_container_width=True,
            )

    with tab_about:
        st.markdown(
            """
### What this project demonstrates
1. **Text cleaning** — HTML/URL stripping, punctuation/number removal, tokenization
2. **Sentiment models** — TF-IDF + Logistic Regression **and** TF-IDF + LinearSVC
3. **Topic modeling** — NMF (primary) and LDA with top words per topic
4. **Metrics** — Accuracy, macro/weighted F1, confusion matrices
5. **Productized demo** — this Streamlit app

### Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/train.py
streamlit run app/streamlit_app.py
```

### Data attribution
SNAP Amazon reviews + UCI Sentiment Labelled Sentences (see `data/SOURCE.txt`).
"""
        )
        if metrics:
            st.json(
                {
                    "dataset": metrics.get("dataset"),
                    "best_model": metrics.get("best_model"),
                    "best_model_summary": metrics.get("best_model_summary"),
                }
            )


if __name__ == "__main__":
    main()
