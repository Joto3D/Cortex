"""Pick which learned skill an assignment means, offline.

"go cut some wood" → "chop trees". Uses MobileCLIP's text encoder (already on your
Mac) when available, plus word overlap, so no API key is needed.
"""
from __future__ import annotations

import re

import numpy as np

from .skill import Skill

_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return {w.rstrip("s") for w in _WORD.findall(text.lower()) if len(w) > 2}


def choose_skill(text: str, skills: list[Skill], encoder=None) -> Skill | None:
    if not skills:
        return None
    text = text.strip()
    if not text:
        return skills[0]
    for s in skills:
        if s.name.lower() == text.lower():
            return s

    overlap = np.array([len(_words(text) & _words(s.name)) for s in skills], np.float32)
    score = overlap.copy()
    if encoder is not None:
        try:
            emb = encoder.encode_text([text] + [s.name for s in skills])
            score = score + emb[1:] @ emb[0]  # cosine similarity (embeddings are normalised)
        except Exception:
            pass
    return skills[int(np.argmax(score))]
