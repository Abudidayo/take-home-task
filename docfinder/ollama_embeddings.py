"""Real local embeddings, backed by Ollama running nomic-embed-text.

Drop-in replacement for the mock `embeddings` module: exposes the same
`anchor_similarity(text) -> float` in [0, 1] that the scorer consumes, so
`signals.score_file` and the funnel stay unchanged. Selected together with the
Ollama LLM backend via `run_demo.py --backend ollama`.

Standard library only (urllib). Requires the model:

    ollama pull nomic-embed-text

Calibration note: dense embeddings sit on a compressed cosine scale (unrelated
text still scores ~0.4). Measured on sample documents, governing docs land near
0.70-0.76 and noise near 0.44-0.53, so we linearly rescale raw cosine from that
band into [0, 1]. In production the band would be fit on labelled data rather
than hard-coded.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List

# Same governance archetypes as the mock. Grown from confirmed docs in prod.
_ANCHORS = [
    "this document sets out the company policy purpose scope responsibilities",
    "agreement between the parties obligations liability termination effective date",
    "standard operating procedure version revision history approved by management",
]

# Rescale band (see calibration note above).
_LO, _HI = 0.45, 0.75


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


@dataclass
class OllamaEmbedder:
    model: str = "nomic-embed-text"
    host: str = "http://localhost:11434"
    timeout: float = 60.0
    calls: int = 0
    _anchors: List[List[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # nomic uses task prefixes; anchors are the "query" side.
        self._anchors = [self._embed(a, "search_query: ") for a in _ANCHORS]

    def _embed(self, text: str, prefix: str) -> List[float]:
        self.calls += 1
        payload = json.dumps({"model": self.model, "prompt": prefix + text}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/embeddings", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))["embedding"]

    def anchor_similarity(self, text: str) -> float:
        """Max cosine of `text` against the governance anchors, rescaled to
        [0, 1]. Empty text (media/scanned) returns 0."""
        if not text:
            return 0.0
        try:
            v = self._embed(text, "search_document: ")
        except (urllib.error.URLError, TimeoutError, ValueError):  # pragma: no cover
            return 0.0
        raw = max((_cosine(v, a) for a in self._anchors), default=0.0)
        return max(0.0, min(1.0, (raw - _LO) / (_HI - _LO)))
