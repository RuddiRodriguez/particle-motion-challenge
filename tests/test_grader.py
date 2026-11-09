"""Adversarial submissions and data separation, using only a temporary directory."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from prepare import prepare


class GraderContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.public, cls.private = cls.root / "public", cls.root / "private"
        prepare(cls.public, cls.private, 91)
        with (cls.private / "answers.csv").open() as file:
            cls.truth = [(r["track_id"], r["label"]) for r in csv.DictReader(file)]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def score(self, predictions):
        submission = self.root / "submission.csv"
        with submission.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["track_id", "label"])
            writer.writerows(predictions)
        raw = subprocess.check_output([sys.executable, "grade.py", submission, "--answers", self.private / "answers.csv"])
        return json.loads(raw)

    def test_perfect_even_when_shuffled(self):
        report = self.score(reversed(self.truth))
        self.assertTrue(report["passed"])
        self.assertEqual(report["score"], 1.0)

    def test_collapsing_to_one_class(self):
        report = self.score((key, "directed") for key, _ in self.truth)
        self.assertFalse(report["passed"])
        self.assertAlmostEqual(report["score"], 1 / 6, places=6)
        self.assertEqual(report["checks"]["recall_diffusive"]["value"], 0)
        self.assertEqual(report["checks"]["recall_directed"]["value"], 1)

    def test_invalid_submissions_fail_closed(self):
        for case in (self.truth[:-1], self.truth + self.truth[:1], [("alien", "diffusive")] + self.truth[1:],
                     [(self.truth[0][0], "unrecognized")] + self.truth[1:]):
            with self.subTest(first=case[0], count=len(case)):
                report = self.score(case)
                self.assertEqual(report["score"], 0)
                self.assertFalse(any(c["passed"] for c in report["checks"].values()))
                self.assertTrue(report["error"])

    def test_blind_seed_changes_test_only(self):
        public, private = self.root / "other-public", self.root / "other-private"
        prepare(public, private, 92)
        self.assertEqual((public / "train_tracks.csv").read_bytes(), (self.public / "train_tracks.csv").read_bytes())
        self.assertNotEqual((public / "test_tracks.csv").read_bytes(), (self.public / "test_tracks.csv").read_bytes())
        self.assertNotEqual((private / "answers.csv").read_bytes(), (self.private / "answers.csv").read_bytes())
        self.assertEqual(sorted(p.name for p in public.iterdir()), ["DATA_LICENSE.txt", "test_tracks.csv", "train_labels.csv", "train_tracks.csv"])
        with (public / "test_tracks.csv").open() as file:
            self.assertEqual(next(csv.reader(file)), ["track_id", "frame", "x", "y"])
