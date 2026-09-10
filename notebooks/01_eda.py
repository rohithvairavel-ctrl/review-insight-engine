"""Companion script for notebooks/01_eda.ipynb — run without Jupyter."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.load import load_reviews
from src.clean import clean_text, tokenize

df = load_reviews(prefer_full=True)
print("Rows:", len(df))
print(df["sentiment"].value_counts())
print(df["score"].describe())
sample = df["text"].iloc[0]
print("RAW:", sample[:200])
print("CLEAN:", clean_text(sample)[:200])
print("TOKENS:", tokenize(sample)[:20])
