"""The ways of measuring how close two segments are.

Four of them, and the point of the lab is that they disagree:

``levenshtein``   counts single-character edits, so it feels word order and
                  length directly;
``char_ngrams``   TF-IDF over character n-grams inside word boundaries, compared
                  by cosine, so it ignores order and normalises away length;
``lsa``           the same n-grams squeezed to 60 dimensions, which is as far
                  towards meaning as 160 short segments can carry you;
``embeddings``    vectors from a model trained on billions of words, computed
                  ahead of time and shipped as a file.

None of them is the right one. The right one is whichever gets the cheaper
errors for the project at hand, and a person picks it.
"""

from __future__ import annotations

from tm_playground.compat import patch_pyarrow_stub

patch_pyarrow_stub()

import numpy as np  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import Normalizer  # noqa: E402

from tm_playground.corpus import EMBEDDING_MODEL  # noqa: E402

LEVENSHTEIN = "levenshtein"
CHAR_NGRAMS = "char_ngrams"
LSA = "lsa"
EMBEDDINGS = "embeddings"

MEASURES = (LEVENSHTEIN, CHAR_NGRAMS, LSA, EMBEDDINGS)

# Code keys above, what the student reads below.
MEASURE_LABELS = {
    LEVENSHTEIN: "Левенштейн",
    CHAR_NGRAMS: "символьные n-граммы",
    LSA: "LSA",
    EMBEDDINGS: "готовые эмбеддинги",
}
MEASURE_HELP = {
    LEVENSHTEIN: "Считает правки посимвольно. Чувствителен к порядку слов и к длине.",
    CHAR_NGRAMS: "Косинус между TF-IDF по символьным n-граммам. Порядок игнорирует.",
    LSA: "Те же n-граммы, сжатые до 60 измерений усечённым сингулярным разложением.",
    EMBEDDINGS: f"Косинус между векторами модели {EMBEDDING_MODEL}.",
}
MEASURE_UNITS = {LEVENSHTEIN: "%", CHAR_NGRAMS: "", LSA: "", EMBEDDINGS: ""}

# Both taken from the lab unchanged, so the numbers here are the lab's numbers.
NGRAM_RANGE = (3, 5)
LSA_COMPONENTS = 60
LSA_SEED = 0

# Below this percentage the industry does not show a suggestion at all: fixing
# someone else's unsuitable translation costs more than translating afresh.
# It is a decision about the price of a mistake, not a property of the measure.
DEFAULT_THRESHOLD = 75.0


def levenshtein(first: str, second: str) -> int:
    """Fewest insertions, deletions and substitutions turning one into the other.

    Written out rather than imported: it is six lines, and the shape of the
    percentage it produces only makes sense once you have seen them.
    """
    previous_row = list(range(len(second) + 1))
    for i, character in enumerate(first, start=1):
        current_row = [i]
        for j, other in enumerate(second, start=1):
            insertion = current_row[j - 1] + 1
            deletion = previous_row[j] + 1
            substitution = previous_row[j - 1] + (character != other)
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row
    return previous_row[-1]


def match_percent(first: str, second: str) -> float:
    """The CAT-tool match percentage: 100 means the strings are identical.

    Dividing by the longer string is what makes a short segment found inside a
    long query score badly however well it matches.
    """
    if not first and not second:
        return 100.0
    return (1 - levenshtein(first, second) / max(len(first), len(second))) * 100


def build_char_vectorizer() -> TfidfVectorizer:
    """Character n-grams inside word boundaries — robust to Russian inflection."""
    return TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM_RANGE)


def build_lsa():
    """Character n-grams squeezed to a few dozen latent dimensions."""
    return make_pipeline(
        build_char_vectorizer(),
        TruncatedSVD(n_components=LSA_COMPONENTS, random_state=LSA_SEED),
        Normalizer(copy=False),
    )


def cosine(vector: np.ndarray, matrix) -> np.ndarray:
    """Cosine similarity of one row against every row of a matrix.

    Wrapped so that callers say what they mean and stay clear of the
    two-dimensional input that ``cosine_similarity`` insists on.
    """
    return cosine_similarity(vector, matrix)[0]
