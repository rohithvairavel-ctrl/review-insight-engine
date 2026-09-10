"""Text cleaning and tokenization utilities."""

from __future__ import annotations

import re
import string
from typing import Iterable, List

# Lightweight stopword list (avoids hard NLTK download dependency at import time)
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can did do does doing down during each
    few for from further had has have having he her here hers herself him himself
    his how i if in into is it its itself just me more most my myself no nor not
    now of off on once only or other our ours ourselves out over own s same she
    should so some such t than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what
    when where which while who whom why will with you your yours yourself
    yourselves
    """.split()
)


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """Normalize review text: lowercase, strip HTML/URLs/punct, collapse whitespace."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)  # HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # URLs
    text = re.sub(r"\S+@\S+", " ", text)  # emails
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    else:
        tokens = [t for t in tokens if len(t) > 1]
    return " ".join(tokens)


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Return token list after cleaning."""
    cleaned = clean_text(text, remove_stopwords=remove_stopwords)
    return cleaned.split() if cleaned else []


def clean_series(texts: Iterable[str], remove_stopwords: bool = False) -> List[str]:
    return [clean_text(t, remove_stopwords=remove_stopwords) for t in texts]
