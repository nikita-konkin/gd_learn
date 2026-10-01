"""Write the model scores the metric playground thresholds.

Both models train on synthetic data. Training happens here, once, with
scikit-learn; the page receives only the held-out labels and
the scores, and computes every metric from them with NumPy. That keeps
scikit-learn out of the browser for this page, and the tests check the NumPy
metrics against scikit-learn's.

Run from the repository root:

    python scripts/prepare_metric_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "metric_playground" / "data"
FAILURES = DATA / "failures_scores.csv"
RARE_CLASS = DATA / "rare_scores.csv"


def failure_scores() -> pd.DataFrame:
    """Two per cent failures, a class-weighted logistic regression."""
    generator = np.random.default_rng(0)
    size = 5000
    labels = (generator.random(size) < 0.02).astype(int)
    observations = generator.normal(size=(size, 4)) + labels[:, None] * 1.2
    train_x, test_x, train_y, test_y = train_test_split(
        observations, labels, test_size=0.3, random_state=42, stratify=labels
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(train_x, train_y)
    return pd.DataFrame({"label": test_y, "score": model.predict_proba(test_x)[:, 1]})


def rare_class_scores() -> pd.DataFrame:
    """Three per cent rare class, a random forest trained three ways."""
    features, target = make_classification(
        n_samples=6000, n_features=12, n_informative=5, weights=[0.97, 0.03], flip_y=0.01, random_state=42
    )
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=0.3, random_state=42, stratify=target
    )
    plain = RandomForestClassifier(n_estimators=200, random_state=0).fit(train_x, train_y)
    weighted = RandomForestClassifier(n_estimators=200, random_state=0, class_weight="balanced_subsample").fit(
        train_x, train_y
    )
    generator = np.random.default_rng(0)
    rare_index = np.where(train_y == 1)[0]
    common_index = generator.choice(np.where(train_y == 0)[0], size=len(rare_index) * 3, replace=False)
    selected = np.concatenate([rare_index, common_index])
    undersampled = RandomForestClassifier(n_estimators=200, random_state=0).fit(train_x[selected], train_y[selected])
    return pd.DataFrame(
        {
            "label": test_y,
            "plain": plain.predict_proba(test_x)[:, 1],
            "weighted": weighted.predict_proba(test_x)[:, 1],
            "undersampled": undersampled.predict_proba(test_x)[:, 1],
        }
    )


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for path, table in ((FAILURES, failure_scores()), (RARE_CLASS, rare_class_scores())):
        # repr-precision floats: the CSV must read back bit for bit.
        table.to_csv(path, index=False, float_format="%.17g")
        print(f"{path.relative_to(ROOT)}: {len(table)} rows, {int(table['label'].sum())} positive")


if __name__ == "__main__":
    main()
