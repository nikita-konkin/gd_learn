"""Write the diabetes table the model-complexity playground regularises.

Notebooks load it with ``sklearn.datasets.load_diabetes(as_frame=True)``. The
playground ships it as a CSV so the page does not depend on scikit-learn's
bundled data files being present in the browser build. The table is the one
distributed with scikit-learn (BSD-3-Clause), from Efron, Hastie, Johnstone and
Tibshirani, "Least Angle Regression", Annals of Statistics, 2004.

Run from the repository root:

    python scripts/prepare_fit_data.py
"""

from __future__ import annotations

from pathlib import Path

from sklearn.datasets import load_diabetes

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fit_playground" / "data" / "diabetes.csv"


def diabetes_table():
    features, target = load_diabetes(return_X_y=True, as_frame=True)
    return features.assign(target=target)


def main() -> None:
    table = diabetes_table()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # repr-precision floats: the CSV must read back bit for bit.
    table.to_csv(OUTPUT, index=False, float_format="%.17g")
    print(f"{OUTPUT.relative_to(ROOT)}: {len(table)} rows, {table.shape[1] - 1} features")


if __name__ == "__main__":
    main()
