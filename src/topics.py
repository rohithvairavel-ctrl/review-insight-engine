"""Topic discovery with NMF (and optional LDA) over TF-IDF features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.clean import clean_series
from src.load import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def fit_nmf_topics(
    texts,
    n_topics: int = 8,
    max_features: int = 5000,
    n_top_words: int = 12,
) -> Dict[str, Any]:
    cleaned = clean_series(texts, remove_stopwords=True)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=3,
        max_df=0.85,
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(cleaned)
    model = NMF(
        n_components=n_topics,
        random_state=42,
        init="nndsvda",
        max_iter=400,
    )
    W = model.fit_transform(X)
    H = model.components_
    feature_names = vectorizer.get_feature_names_out()
    topics = _extract_topics(H, feature_names, n_top_words)
    doc_topics = W.argmax(axis=1).tolist()
    return {
        "method": "NMF",
        "n_topics": n_topics,
        "topics": topics,
        "doc_topic_assignments": doc_topics,
        "reconstruction_err": float(model.reconstruction_err_),
        "vectorizer": vectorizer,
        "model": model,
        "W": W,
    }


def fit_lda_topics(
    texts,
    n_topics: int = 8,
    max_features: int = 5000,
    n_top_words: int = 12,
) -> Dict[str, Any]:
    cleaned = clean_series(texts, remove_stopwords=True)
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=3,
        max_df=0.85,
        ngram_range=(1, 1),
    )
    X = vectorizer.fit_transform(cleaned)
    model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch",
        max_iter=20,
        n_jobs=-1,
    )
    W = model.fit_transform(X)
    H = model.components_
    feature_names = vectorizer.get_feature_names_out()
    topics = _extract_topics(H, feature_names, n_top_words)
    doc_topics = W.argmax(axis=1).tolist()
    # Approximate coherence: avg pairwise NPMI is heavy; use topic diversity + perplexity
    perplexity = float(model.perplexity(X))
    return {
        "method": "LDA",
        "n_topics": n_topics,
        "topics": topics,
        "doc_topic_assignments": doc_topics,
        "perplexity": perplexity,
        "vectorizer": vectorizer,
        "model": model,
        "W": W,
    }


def _extract_topics(H, feature_names, n_top_words: int) -> List[Dict[str, Any]]:
    topics = []
    for topic_idx, topic in enumerate(H):
        top_idx = topic.argsort()[: -n_top_words - 1 : -1]
        words = [str(feature_names[i]) for i in top_idx]
        weights = [float(topic[i]) for i in top_idx]
        topics.append(
            {
                "id": topic_idx,
                "label": f"Topic {topic_idx}: {', '.join(words[:4])}",
                "top_words": words,
                "weights": weights,
            }
        )
    return topics


def topic_diversity(topics: List[Dict[str, Any]], top_k: int = 10) -> float:
    """Fraction of unique words among top-k words across topics (higher = more diverse)."""
    all_words = []
    for t in topics:
        all_words.extend(t["top_words"][:top_k])
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


def save_topic_artifacts(result: Dict[str, Any], prefix: str = "nmf") -> Tuple[Path, Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{prefix}_topics.joblib"
    joblib.dump(
        {
            "model": result["model"],
            "vectorizer": result["vectorizer"],
            "method": result["method"],
            "topics": result["topics"],
        },
        model_path,
    )
    summary = {
        "method": result["method"],
        "n_topics": result["n_topics"],
        "topics": result["topics"],
        "topic_diversity": topic_diversity(result["topics"]),
    }
    if "reconstruction_err" in result:
        summary["reconstruction_err"] = result["reconstruction_err"]
    if "perplexity" in result:
        summary["perplexity"] = result["perplexity"]
    json_path = REPORTS_DIR / f"{prefix}_topics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return model_path, json_path


def load_topics(prefix: str = "nmf") -> Dict[str, Any]:
    path = MODELS_DIR / f"{prefix}_topics.joblib"
    if not path.exists():
        # fall back to JSON-only
        json_path = REPORTS_DIR / f"{prefix}_topics.json"
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(path)
    return joblib.load(path)


def assign_topic(text: str, artifact: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    from src.clean import clean_text

    cleaned = clean_text(text, remove_stopwords=True)
    X = artifact["vectorizer"].transform([cleaned])
    W = artifact["model"].transform(X)
    tid = int(W.argmax(axis=1)[0])
    topic = artifact["topics"][tid]
    return tid, topic["label"], topic["top_words"]
