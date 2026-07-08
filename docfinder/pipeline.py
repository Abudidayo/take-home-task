"""The funnel.

    Stage 0  hard-drop media (the ONLY recall-unsafe cut)
    Stage 1-3  score every survivor on cheap signals + embedding similarity
    Graph      propagate 'governing-ness' to rescue weak-metadata files
    Rank       sort by boosted score (we filter almost nothing — we rank)
    Verify     spend the LLM budget top-down on the ranked head
    Audit      spend a reserved slice on a random tail sample -> recall estimate

The output is not a hard "here are the N". It is a ranked, LLM-verified set
plus a *measured* recall estimate with a confidence interval — the only honest
recall signal available with no human in the loop.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import embeddings, graph, signals
from .llm import LLMVerifier
from .models import DriveFile


@dataclass
class PipelineResult:
    surfaced: List[DriveFile]                 # LLM-verified governing docs
    scores: Dict[str, float]                  # boosted score per file id
    n_total: int
    n_after_media: int
    n_verified: int
    n_audited: int
    # measured recall (LLM treated as oracle, since it is the final arbiter)
    recall_estimate: float
    recall_ci: Tuple[float, float]
    estimated_missed: float
    # cost accounting
    funnel_tokens: int
    naive_tokens: int
    # ground-truth eval (available only with a labelled corpus; None otherwise)
    true_recall: Optional[float] = None
    true_missed: List[DriveFile] = field(default_factory=list)

    @property
    def cost_ratio(self) -> float:
        return self.naive_tokens / max(self.funnel_tokens, 1)

    def summary(self) -> str:
        lines = [
            f"files scanned            : {self.n_total}",
            f"after media drop         : {self.n_after_media}",
            f"LLM verifications (head) : {self.n_verified}",
            f"LLM audit calls (tail)   : {self.n_audited}",
            f"surfaced as governing    : {len(self.surfaced)}",
            f"recall estimate          : {self.recall_estimate:.1%} "
            f"(95% CI {self.recall_ci[0]:.1%}-{self.recall_ci[1]:.1%})",
            f"estimated missed         : {self.estimated_missed:.1f}",
            f"funnel token cost        : {self.funnel_tokens:,}",
            f"naive token cost         : {self.naive_tokens:,}",
            f"cost reduction           : {self.cost_ratio:.0f}x cheaper",
        ]
        if self.true_recall is not None:
            lines.append(f"TRUE recall (eval)       : {self.true_recall:.1%} "
                         f"({len(self.true_missed)} truly missed)")
        return "\n".join(lines)


def _wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _debias(observed_rate: float, acc: float) -> float:
    """Rogan-Gladen correction for an imperfect classifier.

    The tail audit uses the LLM, which has a false-positive rate of (1 - acc).
    On a tail that is almost entirely non-governing, the *observed* positive
    rate is dominated by that error, not by real leakage. Inverting the
    classifier's confusion recovers the true prevalence:

        true = (observed - fpr) / (sensitivity + specificity - 1)

    with sensitivity = specificity = acc. Clamped to [0, 1]. In production `acc`
    is calibrated on a small gold set rather than assumed.
    """
    denom = 2 * acc - 1
    if denom <= 0:
        return observed_rate
    return max(0.0, min(1.0, (observed_rate - (1 - acc)) / denom))


def run(
    files: List[DriveFile],
    *,
    max_llm_calls: int = 200,
    audit_fraction: float = 0.2,
    verify_floor: float = 0.05,
    llm_accuracy: float = 0.97,
    seed: int = 0,
    evaluate: bool = True,
    verifier=None,
    embedder=None,
    progress=None,
) -> PipelineResult:
    """Run the funnel over `files` under a hard LLM-call budget.

    `max_llm_calls` stands in for the token budget: a slice (`audit_fraction`)
    is reserved for the tail audit, the rest is spent verifying the ranked head.

    `verifier` is the expensive oracle; any object with `verify(f) -> bool` and
    `.tokens` works. `embedder` is any object with `anchor_similarity(text)`.
    Both default to the deterministic mocks; pass `OllamaVerifier` /
    `OllamaEmbedder` to run against real local models. `llm_accuracy` is the
    assumed accuracy used to debias the audit (calibrated on a gold set in
    production), independent of the verifier's real behaviour.
    """
    rng = random.Random(seed)
    if verifier is None:
        verifier = LLMVerifier(accuracy=llm_accuracy, seed=seed)
    if embedder is None:
        embedder = embeddings
    n_total = len(files)

    # ---- Stage 0: hard-drop media (recall-safe) --------------------------
    survivors = [f for f in files if not f.is_media]
    n_after_media = len(survivors)

    # ---- Stage 1-3: score every survivor (no filtering) ------------------
    base_scores: Dict[str, float] = {}
    for i, f in enumerate(survivors, 1):
        sim = embedder.anchor_similarity(f.text)
        base_scores[f.id] = signals.score_file(f, sim)
        if progress:
            progress("scoring", i, n_after_media)

    # ---- Graph rescue ----------------------------------------------------
    scores = graph.propagate(survivors, base_scores)

    # ---- Rank ------------------------------------------------------------
    ranked = sorted(survivors, key=lambda f: scores[f.id], reverse=True)

    # ---- Budget split ----------------------------------------------------
    audit_budget = int(max_llm_calls * audit_fraction)
    verify_budget = max_llm_calls - audit_budget

    head = [f for f in ranked if scores[f.id] >= verify_floor][:verify_budget]
    head_ids = {f.id for f in head}
    tail = [f for f in ranked if f.id not in head_ids]

    # ---- Select the NEAR-MISS BAND to audit -----------------------------
    # A leaked governing doc is far likelier to sit just below the cutoff than
    # at score ~0. Auditing the whole tail uniformly is statistically hopeless:
    # governing docs are a needle, and the LLM's own false-positive rate would
    # swamp them. So we audit only the near-miss band (top of the tail) and,
    # under the score-monotonicity assumption (governing density falls with
    # score), treat the deep tail beyond the band as clean.
    band = tail[: min(len(tail), max(4 * audit_budget, 1))]
    audit_sample = band if len(band) <= audit_budget else rng.sample(band, audit_budget)

    verify_total = len(head) + len(audit_sample)
    _done = 0

    def _tick():
        nonlocal _done
        _done += 1
        if progress:
            progress("verifying", _done, verify_total)

    # ---- Verify the head -------------------------------------------------
    surfaced = []
    for f in head:
        ok = verifier.verify(f)
        _tick()
        if ok:
            surfaced.append(f)
    n_verified = len(head)

    # ---- Audit the near-miss band ---------------------------------------
    leaks = 0
    for f in audit_sample:
        if verifier.verify(f):
            leaks += 1
        _tick()
    n_audited = len(audit_sample)

    observed_rate = (leaks / n_audited) if n_audited else 0.0
    # Debias the audit against the LLM's own false-positive rate.
    leak_rate = _debias(observed_rate, llm_accuracy)
    estimated_missed = leak_rate * len(band)
    raw_lo, raw_hi = _wilson(leaks, n_audited) if n_audited else (0.0, 0.0)
    lo_rate, hi_rate = _debias(raw_lo, llm_accuracy), _debias(raw_hi, llm_accuracy)

    found = len(surfaced)
    def _recall(missed: float) -> float:
        denom = found + missed
        return found / denom if denom else 1.0

    recall_estimate = _recall(estimated_missed)
    # more missing -> lower recall, so hi_rate maps to the lower recall bound
    recall_ci = (_recall(hi_rate * len(band)), _recall(lo_rate * len(band)))

    # ---- Optional ground-truth evaluation --------------------------------
    true_recall = None
    true_missed: List[DriveFile] = []
    if evaluate:
        gov = [f for f in files if f.is_governing]
        surfaced_ids = {f.id for f in surfaced}
        true_hits = [f for f in gov if f.id in surfaced_ids]
        true_missed = [f for f in gov if f.id not in surfaced_ids]
        true_recall = len(true_hits) / len(gov) if gov else 1.0

    return PipelineResult(
        surfaced=surfaced,
        scores=scores,
        n_total=n_total,
        n_after_media=n_after_media,
        n_verified=n_verified,
        n_audited=n_audited,
        recall_estimate=recall_estimate,
        recall_ci=recall_ci,
        estimated_missed=estimated_missed,
        funnel_tokens=verifier.tokens,
        naive_tokens=LLMVerifier.naive_token_cost(files),
        true_recall=true_recall,
        true_missed=true_missed,
    )
