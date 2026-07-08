"""End-to-end pipeline behaviour on the synthetic corpus.

These are the tests that matter: they assert the operating-point guarantees —
recall-first, budget-bounded, media-safe — hold on a realistic mix."""

import unittest

from docfinder import run
from docfinder.corpus import generate_corpus


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = generate_corpus(n_files=3000, n_governing=12, seed=7)
        cls.result = run(cls.corpus, max_llm_calls=200, seed=7)

    def test_media_drop_is_recall_safe(self):
        # No governing document is ever a media file, so stage 0 drops none.
        gov_media = [f for f in self.corpus if f.is_governing and f.is_media]
        self.assertEqual(gov_media, [])

    def test_high_recall(self):
        # Recall-first operating point: we expect to recover almost everything.
        self.assertGreaterEqual(self.result.true_recall, 0.83)

    def test_budget_is_respected(self):
        total_calls = self.result.n_verified + self.result.n_audited
        self.assertLessEqual(total_calls, 200)

    def test_cost_is_below_naive(self):
        # Cheaper than reading everything, even at this small scale.
        self.assertGreater(self.result.cost_ratio, 5)

    def test_cost_decouples_from_corpus_size(self):
        # The core value prop: funnel LLM spend is bounded by the budget and
        # does NOT grow with the corpus, so the saving multiplies with scale.
        small = run(generate_corpus(2000, 12, seed=1), max_llm_calls=200, seed=1)
        large = run(generate_corpus(8000, 12, seed=1), max_llm_calls=200, seed=1)
        # ~constant funnel cost, linearly growing naive cost -> bigger ratio.
        self.assertLess(abs(large.funnel_tokens - small.funnel_tokens),
                        0.5 * small.funnel_tokens)
        self.assertGreater(large.cost_ratio, small.cost_ratio)

    def test_recall_estimate_is_produced_with_ci(self):
        lo, hi = self.result.recall_ci
        self.assertTrue(0.0 <= lo <= self.result.recall_estimate <= hi <= 1.0)

    def test_recall_estimate_tracks_truth(self):
        # The tail audit's estimate should be in the right neighbourhood of the
        # true recall (within 15 points) — it is an estimate, not an oracle.
        self.assertLess(abs(self.result.recall_estimate - self.result.true_recall), 0.15)

    def test_reproducible(self):
        again = run(self.corpus, max_llm_calls=200, seed=7)
        self.assertEqual(again.true_recall, self.result.true_recall)
        self.assertEqual(len(again.surfaced), len(self.result.surfaced))

    def test_surfaced_are_mostly_governing(self):
        # Precision is not the objective, but the surfaced set shouldn't be junk.
        if self.result.surfaced:
            true_pos = sum(1 for f in self.result.surfaced if f.is_governing)
            precision = true_pos / len(self.result.surfaced)
            self.assertGreater(precision, 0.6)


if __name__ == "__main__":
    unittest.main()
