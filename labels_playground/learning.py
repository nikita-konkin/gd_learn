"""Learning curves: is the model short of ideas or short of data?

Lab 1, section 12 draws one curve. Two on the same axes answer a sharper
question: how many labelled segments is a better feature set worth? Read it
horizontally — the size at which the better features match what the worse ones
reach on the whole corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

from labels_playground.compat import patch_pyarrow_stub

# Has to run before sklearn is first touched: in the browser every call that
# inspects its input fails otherwise.
patch_pyarrow_stub()

import numpy as np  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

WORDS, CHARACTERS = "words", "characters"
FEATURE_SETS = (WORDS, CHARACTERS)
FEATURE_LABELS = {WORDS: "слова, TF-IDF", CHARACTERS: "символьные n-граммы 2–4"}

LOGISTIC, BAYES, NEIGHBOURS = "logistic", "bayes", "neighbours"
CLASSIFIERS = (LOGISTIC, BAYES, NEIGHBOURS)
# Lab 1, section 9 compares exactly these three, on character n-grams.
CLASSIFIER_LABELS = {
    LOGISTIC: "логистическая регрессия",
    BAYES: "наивный Байес",
    NEIGHBOURS: "ближайшие соседи (k = 5)",
}
# The curves also need words, and on words the nearest-neighbour score is not a
# measurement. A quarter of the held-out sentences share no word with the
# training part; their vectors are empty, equidistant from every training row,
# and which five rows count as nearest comes down to rounding. Reversing the
# training rows moves the score by 0.025, the browser's arithmetic by 0.05.
CURVE_CLASSIFIERS = (LOGISTIC, BAYES)

FOLDS = 5
TRAIN_FRACTIONS = np.linspace(0.2, 1.0, 6)
# Lab 1 shuffles before cutting subsets: the corpus is sorted by type, and an
# unshuffled first subset holds one class only, which fails the fit.
SHUFFLE_SEED = 0


def build_vectorizer(feature_set: str) -> TfidfVectorizer:
    if feature_set == CHARACTERS:
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    return TfidfVectorizer()


def build_classifier(name: str):
    if name == BAYES:
        return MultinomialNB()
    if name == NEIGHBOURS:
        return KNeighborsClassifier(n_neighbors=5)
    return LogisticRegression(max_iter=2000)


@dataclass(frozen=True)
class Curve:
    feature_set: str
    classifier: str
    sizes: np.ndarray
    train: np.ndarray  # mean accuracy on the data the model was fitted on
    test: np.ndarray  # mean accuracy on the held-out fold
    test_spread: np.ndarray  # standard deviation across folds

    @property
    def final(self) -> float:
        return float(self.test[-1])


def learning_curve_for(texts, labels, feature_set: str, classifier: str) -> Curve:
    sizes, train, test = learning_curve(
        make_pipeline(build_vectorizer(feature_set), build_classifier(classifier)),
        texts,
        labels,
        cv=FOLDS,
        train_sizes=TRAIN_FRACTIONS,
        scoring="accuracy",
        shuffle=True,
        random_state=SHUFFLE_SEED,
    )
    return Curve(
        feature_set=feature_set,
        classifier=classifier,
        sizes=sizes.astype(int),
        train=train.mean(axis=1),
        test=test.mean(axis=1),
        test_spread=test.std(axis=1),
    )


def catch_up_size(better: Curve, worse: Curve) -> int | None:
    """Fewest segments at which ``better`` matches ``worse`` on all the data.

    The horizontal gap between two learning curves, in labelled segments: what
    a better idea about features is worth in annotation work. ``None`` when the
    better curve never gets there, which the interface must then say plainly.
    """
    target = worse.final
    for size, accuracy in zip(better.sizes, better.test, strict=True):
        if accuracy >= target:
            return int(size)
    return None


def empty_vectors(texts, labels, feature_set: str) -> int:
    """Held-out sentences with no feature the training part knows, over all folds.

    Such a sentence becomes a zero vector, and a model can only give every one
    of them the same answer. The folds are those of the curves at full size, so
    the count explains their right-hand end.
    """
    texts, labels = np.asarray(texts), np.asarray(labels)
    empty = 0
    for train, test in StratifiedKFold(FOLDS).split(texts, labels):
        vectorizer = build_vectorizer(feature_set).fit(texts[train])
        empty += int((vectorizer.transform(texts[test]).getnnz(axis=1) == 0).sum())
    return empty


def baseline_accuracy(labels) -> float:
    """Always answer the most frequent class. On a balanced corpus that is 1/4."""
    zeros = np.zeros((len(labels), 1))
    return float(cross_val_score(DummyClassifier(strategy="most_frequent"), zeros, labels, cv=FOLDS).mean())
