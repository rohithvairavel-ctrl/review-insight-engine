#!/usr/bin/env python3
"""Decode models/sentiment_pipeline.joblib.b64[.part*] -> .joblib"""
import base64
from pathlib import Path
root = Path(__file__).resolve().parents[1] / "models"
out = root / "sentiment_pipeline.joblib"
b64 = root / "sentiment_pipeline.joblib.b64"
parts = sorted(root.glob("sentiment_pipeline.joblib.b64.part*"))
if b64.exists():
    data = b64.read_text(encoding="utf-8")
elif parts:
    data = "".join(p.read_text(encoding="utf-8") for p in parts)
else:
    raise SystemExit("Missing sentiment_pipeline.joblib.b64 or .part* files")
out.write_bytes(base64.b64decode(data))
print(f"Wrote {out} ({out.stat().st_size} bytes)")
