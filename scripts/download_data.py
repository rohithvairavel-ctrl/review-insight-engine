#!/usr/bin/env python3
"""Download public product-review datasets for Review Insight Engine.

Sources (tried in order):
1. SNAP Stanford Amazon Digital Music 5-core (JSON.gz, real star ratings)
2. UCI Sentiment Labelled Sentences (Amazon / Yelp / IMDB)
3. Curated synthetic fallback (offline reproducibility only)

Attribution is written to data/SOURCE.txt.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import random
import sys
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SNAP_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
    "reviews_Digital_Music_5.json.gz"
)
UCI_URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 review-insight-engine/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def load_snap_amazon(max_per_star: dict | None = None) -> pd.DataFrame:
    print(f"Downloading SNAP Amazon reviews...\n  {SNAP_URL}")
    raw = _get(SNAP_URL, timeout=180)
    max_per_star = max_per_star or {1: 900, 2: 900, 3: 1200, 4: 1500, 5: 1500}
    rows = []
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        for i, line in enumerate(gz):
            try:
                obj = json.loads(line.decode("utf-8", errors="ignore"))
            except Exception:
                continue
            text = (obj.get("reviewText") or "").strip()
            score = obj.get("overall")
            if not text or score is None or len(text) < 30:
                continue
            rows.append(
                {
                    "review_id": obj.get("reviewerID", f"snap_{i}"),
                    "product_title": str(obj.get("asin", "")),
                    "text": text,
                    "score": float(score),
                    "source": "SNAP Amazon Digital Music 5-core",
                }
            )
    df = pd.DataFrame(rows)
    parts = []
    for star, q in max_per_star.items():
        sub = df[df["score"] == float(star)]
        n = min(int(q), len(sub))
        if n:
            parts.append(sub.sample(n=n, random_state=42))
    return pd.concat(parts, ignore_index=True)


def load_uci() -> pd.DataFrame:
    print(f"Downloading UCI Sentiment Labelled Sentences...\n  {UCI_URL}")
    raw = _get(UCI_URL)
    rows = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for name in zf.namelist():
            if not name.endswith(".txt") or "readme" in name.lower():
                continue
            text_blob = zf.read(name).decode("utf-8", errors="ignore")
            stem = Path(name).stem
            for j, line in enumerate(text_blob.splitlines()):
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                text, lab = line.rsplit("\t", 1)
                if lab.strip() not in ("0", "1"):
                    continue
                score = 1.0 if lab.strip() == "0" else 5.0
                rows.append(
                    {
                        "review_id": f"uci_{stem}_{j}",
                        "product_title": stem,
                        "text": text,
                        "score": score,
                        "source": f"UCI Sentiment Labelled Sentences ({stem})",
                    }
                )
    return pd.DataFrame(rows)


def curated_sample(n: int = 5000) -> pd.DataFrame:
    """Offline synthetic fallback only."""
    random.seed(42)
    pos = [
        "Absolutely love this product",
        "Exceeded my expectations",
        "Great value for the money",
        "Works perfectly every time",
        "Highly recommend to anyone",
    ]
    neg = [
        "Very disappointed with this",
        "Stopped working after a week",
        "Poor quality and feels cheap",
        "Would not recommend to anyone",
        "Waste of money honestly",
    ]
    neu = [
        "It is okay for the price",
        "Does the job but nothing special",
        "Average product mixed feelings",
        "Neither great nor terrible",
        "Works as expected mostly",
    ]
    mids = [
        "The packaging was neat and shipping was fast.",
        "Battery life is a key factor for me.",
        "I compared it with two other brands before buying.",
        "There was a slight learning curve at first.",
        "Price seems fair compared to similar options.",
    ]
    products = ["wireless earbuds", "coffee maker", "yoga mat", "laptop stand", "air fryer"]
    rows = []
    rid = 0
    for openers, score, count in [
        (pos, 5, int(n * 0.4)),
        (neu, 3, int(n * 0.2)),
        (neg, 1, int(n * 0.4)),
    ]:
        for _ in range(count):
            text = (
                f"{random.choice(openers)}. I bought the {random.choice(products)}. "
                f"{random.choice(mids)} {random.choice(mids)}"
            )
            rows.append(
                {
                    "review_id": f"synth_{rid}",
                    "product_title": "synthetic",
                    "text": text,
                    "score": score,
                    "source": "Curated synthetic fallback",
                }
            )
            rid += 1
    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    def map_score(s):
        s = float(s)
        if s <= 2:
            return "negative"
        if s == 3:
            return "neutral"
        return "positive"

    out = df.copy()
    out["sentiment"] = out["score"].map(map_score)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-curated", action="store_true")
    parser.add_argument("--sample-size", type=int, default=5000)
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    sources = []
    if not args.force_curated:
        try:
            frames.append(load_snap_amazon())
            sources.append("SNAP Stanford Amazon Digital Music 5-core")
        except Exception as e:
            print(f"SNAP download failed: {e}")
        try:
            frames.append(load_uci())
            sources.append("UCI Sentiment Labelled Sentences")
        except Exception as e:
            print(f"UCI download failed: {e}")

    if not frames:
        print("Using curated synthetic fallback...")
        frames.append(curated_sample(n=max(args.sample_size, 5000)))
        sources.append("Curated synthetic fallback")

    df = pd.concat(frames, ignore_index=True)
    df = add_sentiment(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    sample_parts = []
    targets = {"negative": 1600, "neutral": 1200, "positive": 2200}
    scale = args.sample_size / 5000
    for s, n in targets.items():
        sub = df[df["sentiment"] == s]
        take = min(len(sub), max(1, int(n * scale)))
        sample_parts.append(sub.sample(n=take, random_state=42))
    sample = pd.concat(sample_parts).sample(frac=1, random_state=42).reset_index(drop=True)

    cols = ["review_id", "product_title", "text", "score", "sentiment", "source"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df[cols].to_csv(DATA_DIR / "reviews_full.csv", index=False)
    sample[cols].to_csv(DATA_DIR / "reviews_sample.csv", index=False)

    (DATA_DIR / "SOURCE.txt").write_text(
        "Dataset sources (public):\n\n"
        "1) SNAP Stanford — Amazon Product Reviews (Digital Music 5-core)\n"
        "   http://snap.stanford.edu/data/amazon/productGraph.html\n"
        "2) UCI ML Repository — Sentiment Labelled Sentences\n"
        "   https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences\n\n"
        f"Active sources this run: {', '.join(sources)}\n"
        f"Rows full: {len(df)}\nRows sample: {len(sample)}\n\n"
        f"Sentiment counts (full):\n{df['sentiment'].value_counts().to_string()}\n\n"
        f"Sentiment counts (sample):\n{sample['sentiment'].value_counts().to_string()}\n\n"
        "Star mapping: 1-2=negative, 3=neutral, 4-5=positive.\n",
        encoding="utf-8",
    )
    print(f"Wrote full={len(df)} sample={len(sample)}")
    print(df["sentiment"].value_counts())


if __name__ == "__main__":
    main()
