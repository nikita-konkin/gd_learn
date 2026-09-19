"""Grouping the corpus without being told the answers, and the map that follows.

Two negative results of the lab live here, and both are worth keeping rather
than tuning away. Clustering does not recover the content types, and swapping
in better vectors shifts the agreement without fixing it — because the task is
mis-stated, not the method. Clustering finds the groups that are in the data;
the content types are groups a person decided on. Nothing makes those the same.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tm_playground.compat import patch_pyarrow_stub

patch_pyarrow_stub()

from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

# Straight from the lab, so the numbers here are the numbers students get.
N_CLUSTERS = 4
CLUSTER_SEED = 0
N_INIT = 10

REPRESENTATION_LABELS = {
    "char_ngrams": "символьные n-граммы (свой корпус)",
    "lsa": "LSA, 60 измерений",
    "embeddings": "готовые эмбеддинги",
}


def cluster(matrix, n_clusters: int = N_CLUSTERS, seed: int = CLUSTER_SEED) -> np.ndarray:
    """Cluster assignments from k-means, one per segment."""
    return KMeans(n_clusters=n_clusters, random_state=seed, n_init=N_INIT).fit_predict(matrix)


def agreement(labels, assignments: np.ndarray) -> float:
    """Adjusted Rand index: 1.0 reproduces the labelling, 0.0 is chance."""
    return float(adjusted_rand_score(labels, assignments))


def agreement_across_seeds(
    matrix,
    labels,
    seeds=range(8),
    n_clusters: int = N_CLUSTERS,
) -> list[float]:
    """The same measurement repeated from different starting points.

    On 160 segments the spread across seeds is as large as the differences
    people quote between representations, which is the reason a single ARI
    printed to three decimals should not be read as a result.
    """
    return [agreement(labels, cluster(matrix, n_clusters, seed)) for seed in seeds]


def agreement_table(
    matrices: dict[str, object],
    labels,
    n_clusters: int = N_CLUSTERS,
    seeds=range(8),
) -> pd.DataFrame:
    """ARI per representation: the headline seed, and the spread behind it."""
    rows = []
    for name, matrix in matrices.items():
        across = agreement_across_seeds(matrix, labels, seeds, n_clusters)
        rows.append(
            {
                "representation": name,
                "ari": round(agreement(labels, cluster(matrix, n_clusters, CLUSTER_SEED)), 3),
                "ari_low": round(min(across), 3),
                "ari_high": round(max(across), 3),
            }
        )
    return pd.DataFrame(rows)


def crosstab(labels, assignments: np.ndarray) -> pd.DataFrame:
    """Which content types landed in which cluster."""
    return pd.crosstab(pd.Series(assignments, name="кластер"), pd.Series(list(labels), name="тип"))


def corpus_map(matrix, seed: int = CLUSTER_SEED) -> np.ndarray:
    """Two coordinates per segment, for looking at the corpus on a plane.

    Two components keep only part of the variation, so points overlapping here
    are not necessarily indistinguishable in the space the model actually used.
    The map raises hypotheses; it does not settle them.
    """
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    return PCA(n_components=2, random_state=seed).fit_transform(dense)
