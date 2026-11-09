"""Simulate independent particle tracks. The answer key belongs to the evaluator."""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np


LABELS = ("diffusive", "directed", "confined")


def split_rng(seed, split):
    digest = hashlib.sha256(f"motion-v1:{split}:{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "big"))


def trajectory(rng, label, length, noisy):
    # Speed scale, direction and initial camera position vary independently of class.
    step = rng.uniform(0.4, 1.8)
    angle = rng.uniform(0, 2 * np.pi)
    drift = step * rng.uniform(0.45, 1.0) * np.array([np.cos(angle), np.sin(angle)])
    pull = rng.uniform(0.16, 0.38)
    points = np.zeros((length, 2))
    for t, kick in enumerate(rng.normal(0, step, (length - 1, 2)), 1):
        previous = points[t - 1]
        if label == "directed":
            kick = kick + drift
        elif label == "confined":
            kick = kick - pull * previous
        points[t] = previous + kick
    error = rng.uniform(0.45, 0.75) if noisy else rng.uniform(0.05, 0.18)
    return points + rng.uniform(-100, 100, 2) + rng.normal(0, error * step, points.shape)


def write_split(split, seed, copies, public, private):
    rng = split_rng(seed, split)
    cases = [(label, length, noisy) for label in LABELS for length in (32, 80)
             for noisy in (False, True) for _ in range(copies)]
    rng.shuffle(cases)  # IDs and row order must not encode motion type.
    is_train = split == "train"
    label_path = public / "train_labels.csv" if is_train else private / "answers.csv"
    with (public / f"{split}_tracks.csv").open("w", newline="") as tracks, label_path.open("w", newline="") as labels:
        frames, keys = csv.writer(tracks), csv.writer(labels)
        frames.writerow(["track_id", "frame", "x", "y"])
        keys.writerow(["track_id", "label"] if is_train else ["track_id", "label", "length", "noise"])
        for idx, (label, length, noisy) in enumerate(cases):
            track_id = f"{split}-{idx:04d}"
            points = trajectory(rng, label, length, noisy)
            frames.writerows((track_id, frame, f"{x:.6f}", f"{y:.6f}") for frame, (x, y) in enumerate(points))
            keys.writerow([track_id, label] if is_train else [track_id, label, length, "high" if noisy else "low"])


def prepare(public, private, test_seed):
    if private.resolve().is_relative_to(public.resolve()):
        raise ValueError("The evaluator's answer key must be outside the contestant directory.")
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    write_split("train", 127, 75, public, private)
    write_split("test", test_seed, 25, public, private)
    (public / "DATA_LICENSE.txt").write_bytes(Path(__file__).with_name("DATA_LICENSE.txt").read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, default=Path("work/public"))
    parser.add_argument("--private", type=Path, default=Path("work/private"))
    parser.add_argument("--test-seed", type=int, default=9021, help="Public demo seed; use an evaluator-only seed for blind grading.")
    args = parser.parse_args()
    prepare(args.public, args.private, args.test_seed)
