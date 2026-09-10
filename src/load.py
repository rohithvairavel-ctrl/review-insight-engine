"""Data loading helpers for Review Insight Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_PATH = DATA_DIR / "reviews_sample.csv"
FULL_PATH = DATA_DIR / "reviews_full.csv"

SENTIMENT_MAP = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def score_to_sentiment(score: float) -> str:
    """Map star rating (1-5) to ternary sentiment."""
    s = float(score)
    if s <= 2:
        return "negative"
    if s == 3:
        return "neutral"
    return "positive"


def load_reviews(path: Optional[Path] = None, prefer_full: bool = True) -> pd.DataFrame:
    """Load reviews CSV with columns: text, score, sentiment (and optional product/id)."""
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    if prefer_full:
        candidates.extend([FULL_PATH, SAMPLE_PATH])
    else:
        candidates.extend([SAMPLE_PATH, FULL_PATH])

    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            return _normalize_frame(df)

    raise FileNotFoundError(
        f"No reviews CSV found. Run scripts/download_data.py first. Looked in: {candidates}"
    )


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {c.lower().strip(): c for c in df.columns}
    text_col = None
    for key in ("text", "review", "reviewtext", "review_text", "content", "body"):
        if key in colmap:
            text_col = colmap[key]
            break
    score_col = None
    for key in ("score", "rating", "stars", "overall", "star_rating"):
        if key in colmap:
            score_col = colmap[key]
            break
    if text_col is None:
        raise ValueError(f"Could not find text column in {list(df.columns)}")

    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str)
    if score_col is not None:
        out["score"] = pd.to_numeric(df[score_col], errors="coerce")
    elif "sentiment" in colmap:
        sent = df[colmap["sentiment"]].astype(str).str.lower()
        out["sentiment"] = sent
        out["score"] = sent.map({"negative": 1.0, "neutral": 3.0, "positive": 5.0})
    else:
        raise ValueError("Need a score/rating column or a sentiment column")

    out = out.dropna(subset=["text", "score"])
    out = out[out["text"].str.strip().astype(bool)]
    if "sentiment" not in out.columns:
        out["sentiment"] = out["score"].map(score_to_sentiment)
    out["label"] = out["sentiment"].map(SENTIMENT_MAP)
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    out = out.reset_index(drop=True)
    return out


def train_test_split_reviews(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
