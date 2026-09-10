# Review Insight Engine

**End-to-end product-review NLP** — sentiment classification, topic/theme discovery, and a recruiter-friendly Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-ff4b4b)](https://streamlit.io/)

Paste a review → get **sentiment + confidence**. Explore **NMF/LDA topics**. Inspect **real train/test metrics** (accuracy, F1, confusion matrices).

---

## Highlights

| Capability | Implementation |
|---|---|
| Text cleaning | HTML/URL strip, punctuation & digit removal, tokenization, stopwords |
| Sentiment (baseline) | **TF-IDF (1–2 grams) + Logistic Regression** |
| Sentiment (stronger) | **TF-IDF + LinearSVC** — best model selected by macro-F1 |
| Topics | **NMF** (primary) + **LDA** with top words & topic diversity |
| Evaluation | Accuracy, macro/weighted F1, confusion matrix SVGs |
| Demo | `streamlit run app/streamlit_app.py` |

---

## Quickstart

```bash
git clone https://github.com/rohithvairavel-ctrl/review-insight-engine.git
cd review-insight-engine

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
python scripts/download_data.py   # SNAP Amazon + UCI (or curated fallback)
python scripts/train.py           # trains, evaluates, writes reports/ + models/
streamlit run app/streamlit_app.py
```

Decode the committed model sidecar if needed:

```bash
python scripts/decode_model.py
```

---

## Project structure

```
review-insight-engine/
├── app/streamlit_app.py          # Interactive insight dashboard
├── data/
│   ├── reviews_sample.csv        # Committed sample (thousands of reviews)
│   └── SOURCE.txt                # Dataset attribution
├── models/                       # joblib pipelines (+ .b64 sidecar for text remotes)
├── notebooks/
│   ├── 01_eda.ipynb / 01_eda.py
│   └── 02_nlp.ipynb / 02_nlp.py
├── reports/
│   ├── metrics.json              # Real metrics from last train run
│   └── figures/*.svg             # Confusion matrices, topics, distributions
├── scripts/
│   ├── download_data.py
│   └── train.py
└── src/
    ├── load.py                   # CSV load + star→sentiment mapping
    ├── clean.py                  # Cleaning / tokenization
    ├── train_sentiment.py        # TF-IDF classifiers
    ├── topics.py                 # NMF + LDA
    └── evaluate.py               # Metrics + SVG plots
```

---

## Dataset & attribution

`scripts/download_data.py` pulls **public** review corpora:

1. [SNAP Stanford Amazon Product Reviews](http://snap.stanford.edu/data/amazon/productGraph.html) — Digital Music 5-core (real star ratings)
2. [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences) — Amazon / Yelp / IMDB

A stratified `data/reviews_sample.csv` is committed; full data is regenerated locally (gitignored). Offline synthetic fallback exists if downloads fail. See `data/SOURCE.txt`.

Star ratings are mapped to ternary labels:

- 1–2 → **negative**
- 3 → **neutral**
- 4–5 → **positive**

---

## Results

Metrics below are from a real `scripts/train.py` run on the SNAP + UCI review sample
(**n=8983**, train=7186, test=1797).
Best model by macro-F1: **`tfidf_logreg`**.

| Model | Accuracy | F1 macro | F1 weighted |
|---|---:|---:|---:|
| `tfidf_logreg` | 0.7206 | 0.6666 | 0.7317 |
| `tfidf_linearsvc` | 0.7451 | 0.6519 | 0.7381 |

NMF topic diversity: **0.775** · LDA perplexity: **1648.0**

Artifacts:

- `reports/metrics.json`
- `reports/figures/model_comparison.svg`
- `reports/figures/cm_*.svg`
- `reports/figures/nmf_topics.svg`

Pipeline:

1. Stratified 80/20 train/test split  
2. Fit `tfidf_logreg` and `tfidf_linearsvc`  
3. Pick best by **macro-F1**  
4. Fit NMF (8 topics) + LDA; report diversity / perplexity  

---

## NLP methods (resume bullets)

- Built a reproducible review NLP pipeline with cleaning, **TF-IDF n-gram** features, and dual linear classifiers (**LogReg**, **LinearSVC**).
- Applied **NMF** and **LDA** for unsupervised theme discovery with qualitative top-word inspection and topic-diversity scoring.
- Shipped a **Streamlit** product demo: live sentiment + confidence, topic assignment, and metric visualizations.
- Emphasized evaluation hygiene: stratified split, macro/weighted F1, confusion matrices saved as SVG.

---

## License

Code in this repository is provided for portfolio / educational use. Public SNAP/UCI datasets remain subject to their respective terms; see `data/SOURCE.txt`.
