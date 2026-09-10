# Models

Trained sklearn pipelines are produced by `python scripts/train.py`.

Artifacts written locally:
- `sentiment_pipeline.joblib` (best model)
- `sentiment_pipeline.joblib.b64` (+ optional `.part*` shards for text-only remotes)
- `nmf_topics.joblib` / `lda_topics.joblib`

Decode shards:
```bash
python scripts/decode_model.py
```

The Streamlit app auto-decodes `.b64` / `.part*` if the `.joblib` is missing.
