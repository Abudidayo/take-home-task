"""Graph rescue must lift a weak-metadata file that sits among strong ones,
and must never lower a file's own score."""

import unittest
from datetime import datetime

from docfinder import graph
from docfinder.models import DriveFile


def _f(fid, path, refs=None):
    return DriveFile(
        id=fid, name=fid, mime_type="application/pdf", size=1000, path=path,
        owner="a@a.co", last_modifying_user="a@a.co",
        created_time=datetime(2020, 1, 1), modified_time=datetime(2020, 1, 1),
        revision_count=1, shared=False, references=refs or [],
    )


class TestGraph(unittest.TestCase):
    def test_folder_colocation_rescues_weak_file(self):
        files = [_f("weak", "/Policies"),
                 _f("strong1", "/Policies"),
                 _f("strong2", "/Policies")]
        base = {"weak": 0.02, "strong1": 0.9, "strong2": 0.85}
        boosted = graph.propagate(files, base)
        self.assertGreater(boosted["weak"], base["weak"])
        self.assertGreater(boosted["weak"], 0.1)  # lifted meaningfully

    def test_incoming_reference_rescues(self):
        files = [_f("weak", "/Junk"),
                 _f("strong", "/Policies", refs=["weak"])]
        base = {"weak": 0.01, "strong": 0.95}
        boosted = graph.propagate(files, base)
        self.assertGreater(boosted["weak"], base["weak"])

    def test_never_lowers_own_score(self):
        files = [_f("a", "/Junk"), _f("b", "/Junk")]
        base = {"a": 0.7, "b": 0.0}
        boosted = graph.propagate(files, base)
        self.assertGreaterEqual(boosted["a"], base["a"])

    def test_noise_folder_does_not_inflate(self):
        files = [_f(f"n{i}", "/Junk") for i in range(5)]
        base = {f"n{i}": 0.0 for i in range(5)}
        boosted = graph.propagate(files, base)
        for i in range(5):
            self.assertLess(boosted[f"n{i}"], 0.05)


if __name__ == "__main__":
    unittest.main()
