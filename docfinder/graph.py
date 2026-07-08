"""Graph rescue — the highest-leverage recall move.

A governing document with terrible metadata (bad name, junk folder, thin text)
scores near zero on every cheap signal. The one thing that can still save it,
for free, is its *neighbours*:

  * folder co-location — a file sitting among three confirmed policies is
    probably a policy;
  * incoming references — a file cited by confirmed governing docs is probably
    governing.

We propagate "governing-ness" through these edges so a weak-metadata file
inherits some signal from its strong neighbours. This is what lifts the hard
cases above the budget cutoff, and it costs nothing (no reads, no LLM).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .models import DriveFile


def propagate(
    files: List[DriveFile],
    base_scores: Dict[str, float],
    folder_weight: float = 0.5,
    reference_weight: float = 0.6,
    iterations: int = 2,
) -> Dict[str, float]:
    """Return boosted scores. A file's score is lifted (never lowered) toward
    the strength of its folder siblings and the files that reference it.

    The boost is capped so a lone strong neighbour can rescue a file, but noise
    folders full of low scores contribute ~nothing.
    """
    by_folder: Dict[str, List[str]] = defaultdict(list)
    incoming: Dict[str, List[str]] = defaultdict(list)
    for f in files:
        by_folder[f.path].append(f.id)
        for ref in f.references:
            incoming[ref].append(f.id)

    scores = dict(base_scores)
    for _ in range(iterations):
        updated: Dict[str, float] = {}
        for f in files:
            own = scores[f.id]

            siblings = [scores[i] for i in by_folder[f.path] if i != f.id]
            folder_signal = max(siblings) if siblings else 0.0

            citers = [scores[i] for i in incoming.get(f.id, [])]
            ref_signal = max(citers) if citers else 0.0

            boost = folder_weight * folder_signal + reference_weight * ref_signal
            # A file can be lifted toward, but not past, a blend of its
            # neighbours; it never *loses* its own score.
            updated[f.id] = max(own, min(1.0, own + boost * (1.0 - own)))
        scores = updated
    return scores
