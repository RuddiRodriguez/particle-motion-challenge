"""Score one prediction per track and emit exactly one JSON object."""
import argparse
import csv
import json
from pathlib import Path

LABELS = ("diffusive", "directed", "confined")
LIMITS = {
    "macro_f1": 0.72,
    **{f"recall_{label}": 0.60 for label in LABELS},
    "short_track_macro_f1": 0.62,
    "high_noise_macro_f1": 0.62,
}


def recall(rows, label):
    correct = sum(r["label"] == r["prediction"] == label for r in rows)
    return correct / sum(r["label"] == label for r in rows)


def macro_f1(rows):
    # Each class has equal weight, even if the model never predicts it.
    scores = []
    for label in LABELS:
        correct = sum(r["label"] == r["prediction"] == label for r in rows)
        actual = sum(r["label"] == label for r in rows)
        predicted = sum(r["prediction"] == label for r in rows)
        scores.append(2 * correct / (actual + predicted))
    return sum(scores) / len(scores)


def read_submission(path, expected):
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file, strict=True)
        if next(reader, None) != ["track_id", "label"]:
            raise ValueError("Expected CSV header: track_id,label")
        predictions = {}
        for line, row in enumerate(reader, 2):
            if len(row) != 2:
                raise ValueError(f"Row {line}: expected two fields.")
            track_id, label = row
            if track_id not in expected or track_id in predictions:
                raise ValueError(f"Row {line}: unexpected or repeated track ID.")
            if label not in LABELS:
                raise ValueError(f"Row {line}: expected one of {', '.join(LABELS)}.")
            predictions[track_id] = label
        if len(predictions) != len(expected):
            raise ValueError(f"Missing {len(expected) - len(predictions)} test tracks.")
        return predictions


def grade(submission, answers):
    with answers.open(newline="") as file:
        rows = list(csv.DictReader(file))
    checks = {"complete_submission": {"passed": False}}
    checks.update({name: {"value": None, "minimum": limit, "passed": False} for name, limit in LIMITS.items()})
    try:
        predictions = read_submission(submission, {r["track_id"] for r in rows})
    except (ValueError, OSError, csv.Error) as error:
        return {"score": 0.0, "passed": False, "error": str(error), "checks": checks}

    checks["complete_submission"]["passed"] = True
    for row in rows:
        row["prediction"] = predictions[row["track_id"]]
    metrics = {
        "macro_f1": macro_f1(rows),
        **{f"recall_{label}": recall(rows, label) for label in LABELS},
        "short_track_macro_f1": macro_f1([r for r in rows if int(r["length"]) == 32]),
        "high_noise_macro_f1": macro_f1([r for r in rows if r["noise"] == "high"]),
    }
    for name, value in metrics.items():
        checks[name].update(value=round(value, 6), passed=value >= LIMITS[name])
    return {"score": round(metrics["macro_f1"], 6), "passed": all(c["passed"] for c in checks.values()), "checks": checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--answers", type=Path, default=Path("work/private/answers.csv"))
    args = parser.parse_args()
    print(json.dumps(grade(args.submission, args.answers), sort_keys=True, allow_nan=False))
