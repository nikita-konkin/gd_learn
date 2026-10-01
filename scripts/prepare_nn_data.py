"""Write the two-moons sample the network playground trains on.

It is built with scikit-learn:

    features, target = make_moons(n_samples=1000, noise=0.2, random_state=0)
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=0.3, random_state=0, stratify=target)

The playground ships the result as a CSV instead, so the page does not have to
download scikit-learn to draw a thousand points. Row order inside each part is
kept: the gradient check uses the first fifty training rows.

Run from the repository root:

    python scripts/prepare_nn_data.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "nn_playground" / "data" / "moons.csv"


def moons_split():
    features, target = make_moons(n_samples=1000, noise=0.2, random_state=0)
    return train_test_split(features, target, test_size=0.3, random_state=0, stratify=target)


def main() -> None:
    train_x, test_x, train_y, test_y = moons_split()
    frames = []
    for part, matrix, labels in (("train", train_x, train_y), ("test", test_x, test_y)):
        frames.append(pd.DataFrame({"x1": matrix[:, 0], "x2": matrix[:, 1], "label": labels, "part": part}))
    table = pd.concat(frames, ignore_index=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # repr-precision floats: the CSV must read back bit for bit.
    table.to_csv(OUTPUT, index=False, float_format="%.17g")
    print(f"{OUTPUT.relative_to(ROOT)}: {len(table)} rows")


if __name__ == "__main__":
    main()
