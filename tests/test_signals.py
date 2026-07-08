"""Signals should rank a governing doc above an invoice, and never crash on
empty text. These are pure-function tests — no LLM, no corpus randomness."""

import unittest
from datetime import datetime

from docfinder import embeddings, signals
from docfinder.models import DriveFile


def _file(**kw) -> DriveFile:
    base = dict(
        id="x", name="doc", mime_type="application/pdf", size=1000,
        path="/", owner="a@a.co", last_modifying_user="a@a.co",
        created_time=datetime(2020, 1, 1), modified_time=datetime(2020, 1, 1),
        revision_count=1, shared=False,
    )
    base.update(kw)
    return DriveFile(**base)


class TestSignals(unittest.TestCase):
    def test_policy_scores_above_invoice(self):
        policy = _file(
            name="Data Protection Policy.docx",
            mime_type="application/vnd.google-apps.document",
            path="/HR/Policies", revision_count=12, shared=True, editors=4,
            modified_time=datetime(2021, 6, 1),
            text="Table of Contents\n1. Purpose\n2. Scope\nEffective Date: 01/2023\nVersion 2.0",
        )
        invoice = _file(
            name="invoice_1234.pdf", path="/Misc",
            text="Invoice Number: INV-1234\nAmount Due: £500\nPayment due within 30 days",
        )
        ps = signals.score_file(policy, embeddings.anchor_similarity(policy.text))
        inv = signals.score_file(invoice, embeddings.anchor_similarity(invoice.text))
        self.assertGreater(ps, 0.5)
        self.assertLess(inv, 0.2)
        self.assertGreater(ps, inv)

    def test_empty_text_is_safe(self):
        f = _file(name="IMG.jpg", mime_type="image/jpeg", text="")
        self.assertEqual(signals.structure_signal(f), 0.0)
        self.assertEqual(embeddings.anchor_similarity(f.text), 0.0)
        # still returns a valid probability
        s = signals.score_file(f, 0.0)
        self.assertTrue(0.0 <= s <= 1.0)

    def test_edit_dynamics_living_vs_static(self):
        living = _file(revision_count=20, shared=True, editors=4,
                       modified_time=datetime(2022, 1, 1))
        static = _file(revision_count=1, shared=False, editors=1)
        self.assertGreater(signals.edit_dynamics_signal(living),
                           signals.edit_dynamics_signal(static))


if __name__ == "__main__":
    unittest.main()
