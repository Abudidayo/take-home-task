"""Mock embeddings.

The real system would call a small embedding model (e.g. text-embedding-3-small)
on filename + first page + folder path — a few pennies for the whole corpus.
Here we fake it deterministically with a bag-of-words vector over a governance
vocabulary so the prototype runs offline and tests are reproducible.

The *interface* is what matters and is what the pipeline depends on:
    embed(text) -> vector
    cosine(a, b) -> float
    anchor_similarity(text) -> float in [0, 1]

Swapping in a real embedding provider means replacing only this module.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List

# Vocabulary that a governing document tends to use.
_VOCAB = [
    "policy", "procedure", "scope", "purpose", "responsibilities", "compliance",
    "effective", "version", "revision", "agreement", "obligations", "liability",
    "termination", "staff", "company", "risk", "assessment", "handbook",
    "governance", "approved", "management", "confidential", "terms",
]

# Archetypal governing-document phrasings. In production these are seeded from
# confirmed governing docs across businesses and grow over time.
_ANCHORS = [
    "this document sets out the company policy purpose scope responsibilities",
    "agreement between the parties obligations liability termination effective date",
    "standard operating procedure version revision history approved by management",
]

_WORD = re.compile(r"[a-z]+")


def _bow(text: str) -> Dict[str, float]:
    counts: Dict[str, float] = {}
    for w in _WORD.findall(text.lower()):
        if w in _VOCAB:
            counts[w] = counts.get(w, 0.0) + 1.0
    return counts


def embed(text: str) -> Dict[str, float]:
    """Sparse embedding: L2-normalised bag-of-governance-words."""
    bow = _bow(text)
    norm = math.sqrt(sum(v * v for v in bow.values())) or 1.0
    return {k: v / norm for k, v in bow.items()}


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) & set(b)
    return sum(a[k] * b[k] for k in keys)  # already normalised


_ANCHOR_VECS: List[Dict[str, float]] = [embed(a) for a in _ANCHORS]


def anchor_similarity(text: str) -> float:
    """Max cosine similarity of `text` to any governance anchor, in [0, 1]."""
    if not text:
        return 0.0
    v = embed(text)
    return max((cosine(v, a) for a in _ANCHOR_VECS), default=0.0)
