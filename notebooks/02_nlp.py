"""Companion script for notebooks/02_nlp.ipynb — run without Jupyter."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.load import load_reviews, train_test_split_reviews
from src.train_sentiment import train_models
from src.evaluate import evaluate_model
from src.topics import fit_nmf_topics, topic_diversity

df = load_reviews()
train_df, test_df = train_test_split_reviews(df)
models = train_models(train_df["text"], train_df["label"])
for name, pipe in models.items():
    m = evaluate_model(pipe, test_df["text"], test_df["label"], name)
    print(name, round(m["accuracy"], 4), round(m["f1_macro"], 4))
nmf = fit_nmf_topics(train_df["text"], n_topics=8)
print("diversity", topic_diversity(nmf["topics"]))
for t in nmf["topics"]:
    print(t["label"], "->", ", ".join(t["top_words"][:8]))
