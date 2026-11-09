# Particle motion challenge

Given noisy two-dimensional positions over time, classify each previously unseen particle track as **diffusive**, **directed**, or **confined**. One submission row represents a whole track. A track that moves in one direction for a few frames can still be diffusive: use the shape over time.

## Run the benchmark

From a checkout of this repository, with an active Python 3.10 environment on Linux:

```sh
python -m pip install -c constraints-linux.txt . && python run.py
```

This command generates the data, executes [the reference notebook](reference.ipynb) without a browser, writes predictions, grades them twice and fails if the JSON differs or any check fails. Read `work/report.json` for the score and `work/reference.executed.ipynb` for the executed notebook. On macOS, install with `python -m pip install .` and run the same script. No external dataset, network service, GPU or API key is needed after installation.

## Data and submission

The simulator creates 900 independent training tracks and 300 independent test tracks, balanced by class, length (32 or 80 frames) and observation noise (low or high). Diffusive particles take random steps; directed particles also drift; confined particles are pulled back toward their starting region. Starting position, direction, and motion scale vary. These are simulations, not microscopy measurements.

| File | Columns | Who sees it |
| --- | --- | --- |
| `train_tracks.csv`, `test_tracks.csv` | `track_id,frame,x,y`: opaque ID, zero-based frame, floating-point position in arbitrary units | Agent |
| `train_labels.csv` | `track_id,label`: one of `diffusive,directed,confined` | Agent |
| `answers.csv` | `track_id,label,length,noise` | Evaluator only |

Write a CSV with header `track_id,label`, exactly one row per test ID in any order. Example of the format (not a verified answer):

```csv
track_id,label
test-0000,directed
```

## Grading contract

```sh
python prepare.py --public work/public --private work/private
python grade.py work/submission.csv --answers work/private/answers.csv
```

The grader emits a single JSON object: `score` (overall macro F1, 0–1), `passed`, and seven named `checks`. A valid submission receives its measured score even if it misses a threshold. A malformed submission returns score zero, failed checks and an error message. Read the JSON for pass/fail; a scored submission exits normally.

| Check | Pass condition |
| --- | --- |
| Complete submission | Exact header, all and only expected IDs once, one allowed label per ID |
| Overall macro F1 | ≥ 0.72 |
| Diffusive recall | ≥ 0.60 |
| Directed recall | ≥ 0.60 |
| Confined recall | ≥ 0.60 |
| Short-track macro F1 (32 frames) | ≥ 0.62 |
| High-noise macro F1 | ≥ 0.62 |

Recall is the fraction of a class identified correctly. Macro F1 averages each class's F1 (the harmonic mean of precision and recall). Limits apply to unrounded values; JSON displays six decimal places. All seven checks must pass. A constant classifier gets just 0.166667 macro F1 on this balanced test.

**Benchmark:** The reference random forest scored **0.892973 macro F1** on the public test split (seed 9021); all seven checks passed. It uses displacement at several time intervals, alignment between steps and path shape. It never reads the answer key. This score describes only these simulated tracks.

**For a blind evaluation:** Run `prepare.py --test-seed YOUR_PRIVATE_INTEGER` in the evaluator's environment, using the same integer for repeat runs. Give the contestant only the public directory and the task instructions. Run the grader separately; do not mount the private directory or expose the seed, notebook or reference predictions to the contestant. The default seed is a public regression fixture, not a hidden leaderboard. Changing the test seed leaves the training set unchanged.

## Licensing and verification

All observations come from the equations in [prepare.py](prepare.py). Generated datasets and answer keys are dedicated under [CC0 1.0](DATA_LICENSE.txt); code and notebook are [MIT licensed](LICENSE). No Kaggle files or third-party tracks are used.

[GitHub Actions](https://github.com/RuddiRodriguez/particle-motion-challenge/actions/workflows/challenge.yml) installs the package on Ubuntu/Python 3.10, tests invalid submissions, runs the entire notebook twice from scratch and compares SHA-256 hashes of the datasets, predictions and grade reports. The run also publishes the executed notebook, predictions and JSON report.
