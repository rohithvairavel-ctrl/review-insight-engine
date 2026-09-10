#!/usr/bin/env python3
"""Decode data/reviews_sample.csv.b64.part* -> reviews_sample.csv if CSV missing."""
import base64
from pathlib import Path
data = Path(__file__).resolve().parents[1] / "data"
out = data / "reviews_sample.csv"
if out.exists():
    print(f"Already present: {out}")
    raise SystemExit(0)
parts = sorted(data.glob("reviews_sample.csv.b64.part*"))
if not parts:
    raise SystemExit("Missing reviews_sample.csv and b64 parts; run scripts/download_data.py")
out.write_bytes(base64.b64decode("".join(p.read_text(encoding="utf-8") for p in parts)))
print(f"Wrote {out} ({out.stat().st_size} bytes)")
