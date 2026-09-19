"""Scoring the classifier, and working out which segments it gets wrong."""

from __future__ import annotations

from dataclasses import dataclass

from vec_playground.compat import patch_pyarrow_stub

# Has to run before sklearn is first touched: in the browser every call that
# inspects its input fails otherwise.
patch_pyarrow_stub()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.model_selection import cross_val_predict, cross_val_score  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from vec_playground.features import FeatureSettings, build_vectorizer  # noqa: E402

CLASSIFIERS = ("логистическая регрессия", "наивный Байес", "линейный SVM")

FOLDS = 5


def build_classifier(name: str):
    if name == "наивный Байес":
        return MultinomialNB()
    if name == "линейный SVM":
        return LinearSVC()
    return LogisticRegression(max_iter=2000)


def build_pipeline(settings: FeatureSettings, classifier: str):
    return make_pipeline(build_vectorizer(settings), build_classifier(classifier))


@dataclass
class Score:
    accuracy: float
    deviation: float
    features: int


def evaluate(texts, labels, settings: FeatureSettings, classifier: str) -> Score:
    """Accuracy over five folds, the way the lab measures it.

    A single split of 160 segments is too noisy: the spread between folds is
    comparable to the difference between configurations, so it has to be shown
    alongside the mean rather than hidden behind it.
    """
    pipeline = build_pipeline(settings, classifier)
    scores = cross_val_score(pipeline, texts, labels, cv=FOLDS, scoring="accuracy")
    matrix = build_vectorizer(settings).fit_transform(texts)
    return Score(accuracy=float(scores.mean()), deviation=float(scores.std()), features=int(matrix.shape[1]))


def baseline_accuracy(labels) -> float:
    """Always answer the most frequent class. On a balanced corpus that is 1/4."""
    dummy = DummyClassifier(strategy="most_frequent")
    zeros = np.zeros((len(labels), 1))
    scores = cross_val_score(dummy, zeros, labels, cv=FOLDS, scoring="accuracy")
    return float(scores.mean())


def predictions(texts, labels, settings: FeatureSettings, classifier: str) -> np.ndarray:
    """Out-of-fold predictions: each segment is answered by a model that never saw it."""
    return cross_val_predict(build_pipeline(settings, classifier), texts, labels, cv=FOLDS)


def confusion(labels, predicted, classes: list[str]) -> pd.DataFrame:
    matrix = confusion_matrix(labels, predicted, labels=classes)
    return pd.DataFrame(matrix, index=classes, columns=classes)


def mistakes(texts, labels, predicted) -> pd.DataFrame:
    """The segments the model got wrong, with the text itself."""
    frame = pd.DataFrame({"текст": texts, "правильно": labels, "модель сказала": predicted})
    return frame[frame["правильно"] != frame["модель сказала"]].reset_index(drop=True)


def flipped(texts, labels, before: np.ndarray, after: np.ndarray) -> pd.DataFrame:
    """Segments whose answer changed between two configurations.

    This is exactly what a mean accuracy hides: two feature sets scoring the
    same number can be wrong about entirely different segments.
    """
    frame = pd.DataFrame(
        {
            "текст": list(texts),
            "правильно": list(labels),
            "было": list(before),
            "стало": list(after),
        }
    )
    changed = frame[frame["было"] != frame["стало"]].copy()
    changed["итог"] = np.where(
        changed["стало"] == changed["правильно"],
        "исправлено",
        np.where(changed["было"] == changed["правильно"], "испорчено", "всё ещё неверно"),
    )
    return changed.reset_index(drop=True)


def top_features(texts, labels, settings: FeatureSettings, top: int = 8) -> pd.DataFrame:
    """The heaviest-weighted features for each class.

    Always computed with logistic regression, whichever classifier is selected:
    weights mean different things to naive Bayes and to an SVM, while the
    question "what is the model looking at" is one question.
    """
    vectorizer = build_vectorizer(settings)
    matrix = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=2000).fit(matrix, labels)
    names = np.asarray(vectorizer.get_feature_names_out())

    rows = []
    for index, class_name in enumerate(model.classes_):
        weights = model.coef_[index]
        best = np.argsort(weights)[-top:][::-1]
        rows.append({"класс": class_name, "признаки": ", ".join(repr(str(n)) for n in names[best])})
    return pd.DataFrame(rows)


@dataclass
class SweepPoint:
    analyzer: str
    ngram_max: int
    accuracy: float
    deviation: float
    features: int


def sweep_ngrams(texts, labels, classifier: str, max_n: int = 6) -> list[SweepPoint]:
    """Accuracy as a function of n-gram length, for words and for characters.

    This is the headline result of lab 1 drawn as a curve: word features hit a
    ceiling early, character features keep climbing past it.
    """
    points = []
    for analyzer, start in (("слова", 1), ("символы внутри слов", 2)):
        for ngram_max in range(start, max_n + 1):
            settings = FeatureSettings(analyzer=analyzer, ngram_min=start, ngram_max=ngram_max)
            score = evaluate(texts, labels, settings, classifier)
            points.append(
                SweepPoint(
                    analyzer=analyzer,
                    ngram_max=ngram_max,
                    accuracy=score.accuracy,
                    deviation=score.deviation,
                    features=score.features,
                )
            )
    return points


# The set from section 8 of the lab. The numbers in the comments are what the
# lab documents for logistic regression, and the tests check them.
LAB_CONFIGURATIONS = (
    ("TF-IDF по умолчанию", FeatureSettings()),  # 0.53
    ("без нижнего регистра", FeatureSettings(lowercase=False)),
    ("слова + биграммы", FeatureSettings(ngram_max=2)),
    ("min_df=2 (без редких)", FeatureSettings(min_df=2)),
    ("стемминг", FeatureSettings(stemming=True)),  # 0.60
    ("символьные 3-5", FeatureSettings(analyzer="символы внутри слов", ngram_min=3, ngram_max=5)),
    ("символьные 2-4", FeatureSettings(analyzer="символы внутри слов", ngram_min=2, ngram_max=4)),  # 0.74
)


def lab_comparison(texts, labels, classifier: str) -> pd.DataFrame:
    rows = []
    for name, settings in LAB_CONFIGURATIONS:
        score = evaluate(texts, labels, settings, classifier)
        rows.append(
            {
                "конфигурация": name,
                "точность": round(score.accuracy, 3),
                "разброс": round(score.deviation, 3),
                "признаков": score.features,
            }
        )
    return pd.DataFrame(rows)
