"""Generate the synthetic sample datasets in this folder.

The compounds and values are made up. They mimic the shape of a real QSAR
dataset (one ID column, one target column, then molecular descriptors) so the
app can be tried without private research data. Standard library only.

Run from the repository root:  python examples/generate_sample_data.py
"""

import csv
import random
import statistics
from pathlib import Path

N_COMPOUNDS = 80
SEED = 7

rng = random.Random(SEED)


def clip(value, low, high):
    return max(low, min(high, value))


rows = []
for i in range(1, N_COMPOUNDS + 1):
    mol_wt = clip(rng.gauss(380, 70), 180, 600)
    d = {
        "MolWt": mol_wt,
        # Almost a copy of MolWt: the inter-R² pre-treatment filter should drop it.
        "MolMR": mol_wt * 0.27 + rng.gauss(0, 0.8),
        "LogP": clip(rng.gauss(3.0, 1.2), -1, 7),
        "TPSA": clip(rng.gauss(85, 25), 20, 160),
        "HBD": rng.randint(0, 4),
        "HBA": rng.randint(1, 9),
        "RotatableBonds": rng.randint(0, 11),
        "AromaticRings": rng.randint(0, 4),
        "FractionCSP3": rng.uniform(0.05, 0.7),
        "HeavyAtoms": round(mol_wt / 13.5 + rng.gauss(0, 2)),
        # Nearly constant: the variance pre-treatment filter should drop it.
        "Halogens": 0.01 if rng.random() < 0.03 else 0.0,
    }
    # Activity depends on three descriptors plus noise, so a 3-term equation
    # (the app's default equation length) can recover it.
    pic50 = (
        5.0
        + 0.55 * d["LogP"]
        - 0.018 * d["TPSA"]
        + 0.35 * d["AromaticRings"]
        + rng.gauss(0, 0.25)
    )
    rows.append((f"CMPD-{i:03d}", round(pic50, 2), {k: round(v, 3) for k, v in d.items()}))

names = list(rows[0][2])
median = statistics.median(r[1] for r in rows)
out = Path(__file__).resolve().parent

with open(out / "sample_regression.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ID", "pIC50", *names])
    for cid, pic50, d in rows:
        w.writerow([cid, pic50, *d.values()])

with open(out / "sample_classification.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ID", "Activity", *names])
    for cid, pic50, d in rows:
        w.writerow([cid, "Active" if pic50 >= median else "Inactive", *d.values()])

print(f"Wrote {N_COMPOUNDS} compounds to {out}")
