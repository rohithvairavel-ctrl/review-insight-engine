"""Sentiment model training: TF-IDF + LogisticRegression and LinearSVC baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.clean import clean_series
from src.load import LABEL_NAMES, PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"


def build_logreg_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="saga",
                    random_state=42,
                ),
            ),
        ]
    )


def build_linearsvc_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=25000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LinearSVC(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42,
                    dual="auto",
                ),
            ),
        ]
    )


def prepare_xy(texts, labels, remove_stopwords: bool = False):
    X = clean_series(texts, remove_stopwords=remove_stopwords)
    y = np.asarray(labels, dtype=int)
    return X, y


def train_models(
    train_texts, train_labels
) -> Dict[str, Pipeline]:
    """Train both baselines; return dict of fitted pipelines."""
    X, y = prepare_xy(train_texts, train_labels)
    models: Dict[str, Pipeline] = {}

    logreg = build_logreg_pipeline()
    logreg.fit(X, y)
    models["tfidf_logreg"] = logreg

    svc = build_linearsvc_pipeline()
    svc.fit(X, y)
    models["tfidf_linearsvc"] = svc
    return models


def save_model(pipeline: Pipeline, name: str = "sentiment_pipeline") -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(pipeline, path)
    return path


def load_model(name: str = "sentiment_pipeline") -> Pipeline:
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run scripts/train.py first.")
    return joblib.load(path)


def predict_sentiment(pipeline: Pipeline, text: str) -> Tuple[str, float, Dict[str, float]]:
    """Return (label_name, confidence, class_probabilities_or_scores)."""
    from src.clean import clean_text

    cleaned = clean_text(text)
    pred = int(pipeline.predict([cleaned])[0])
    label = LABEL_NAMES[pred]
    conf = 0.0
    details: Dict[str, float] = {}

    clf = pipeline.named_steps["clf"]
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba([cleaned])[0]
            for i, p in enumerate(proba):
                details[LABEL_NAMES[i]] = float(p)
            conf = float(max(proba))
            return label, conf, details
        except Exception:
            pass

    if hasattr(clf, "decision_function"):
        scores = pipeline.decision_function([cleaned])[0]
        if np.ndim(scores) == 0:
            details = {label: float(scores)}
            conf = float(1 / (1 + np.exp(-abs(scores))))
        else:
            ex = np.exp(scores - np.max(scores))
            soft = ex / ex.sum()
            for i, p in enumerate(soft):
                details[LABEL_NAMES[i]] = float(p)
            conf = float(max(soft))
    else:
        conf = 1.0
        details[label] = 1.0
    return label, conf, details


def model_summary(pipeline: Pipeline) -> Dict[str, Any]:
    return {
        "steps": [name for name, _ in pipeline.steps],
        "vectorizer": type(pipeline.named_steps["tfidf"]).__name__,
        "classifier": type(pipeline.named_steps["clf"]).__name__,
        "vocab_size": int(len(pipeline.named_steps["tfidf"].vocabulary_)),
    }
