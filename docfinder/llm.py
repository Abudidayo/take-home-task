"""Mock LLM verifier — the expensive oracle at the bottom of the funnel.

In production this is a real LLM call that reads a candidate and returns a
classification + extracted metadata, costing real tokens. It is the final
arbiter (there is no human in the loop), so we let a generous number of
candidates reach it — but never all 50k.

Here we simulate it deterministically: it returns the ground-truth label with
accuracy `accuracy`, and it records a token cost proportional to the text it
had to read. The pipeline treats it as a black box behind `verify`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List

from .models import DriveFile


@dataclass
class LLMVerifier:
    accuracy: float = 0.97
    seed: int = 0
    calls: int = 0
    tokens: int = 0
    _log: List[str] = field(default_factory=list)

    def _rand(self, file_id: str) -> float:
        h = hashlib.sha256(f"{self.seed}:{file_id}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def verify(self, f: DriveFile) -> bool:
        """Return the model's judgement of whether `f` is a governing document.

        Costs are accounted so the demo can compare funnel spend against the
        naive 'read everything' baseline.
        """
        self.calls += 1
        # ~1 token per 4 chars, capped as if we only read the first pages.
        self.tokens += min(len(f.text), 6000) // 4 + 20
        self._log.append(f.id)
        # Simulate an imperfect but strong classifier.
        correct = self._rand(f.id) < self.accuracy
        return f.is_governing if correct else (not f.is_governing)

    @staticmethod
    def naive_token_cost(files: List[DriveFile]) -> int:
        """Tokens the baseline would spend reading *every* text-bearing file."""
        return sum(min(len(f.text), 6000) // 4 + 20 for f in files if f.text)
