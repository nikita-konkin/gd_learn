"""Оценка качества классификации и разбор того, на чём модель ошибается."""

from __future__ import annotations

from dataclasses import dataclass

from vec_playground.compat import patch_pyarrow_stub

# Должно отработать до первого обращения к sklearn: в браузере иначе падает
# любой вызов, разбирающий входные данные.
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
    """Точность по 5-кратной кросс-валидации — как в лабораторной работе.

    Одно разбиение на 160 сегментах слишком шумное: разброс между частями
    сравним с разницей между конфигурациями, и его тоже надо показывать.
    """
    pipeline = build_pipeline(settings, classifier)
    scores = cross_val_score(pipeline, texts, labels, cv=FOLDS, scoring="accuracy")
    matrix = build_vectorizer(settings).fit_transform(texts)
    return Score(accuracy=float(scores.mean()), deviation=float(scores.std()), features=int(matrix.shape[1]))


def baseline_accuracy(labels) -> float:
    """Самый частый класс. На сбалансированном корпусе это 1/4."""
    dummy = DummyClassifier(strategy="most_frequent")
    zeros = np.zeros((len(labels), 1))
    scores = cross_val_score(dummy, zeros, labels, cv=FOLDS, scoring="accuracy")
    return float(scores.mean())


def predictions(texts, labels, settings: FeatureSettings, classifier: str) -> np.ndarray:
    """Предсказания вне обучения — каждый сегмент предсказан моделью, его не видевшей."""
    return cross_val_predict(build_pipeline(settings, classifier), texts, labels, cv=FOLDS)


def confusion(labels, predicted, classes: list[str]) -> pd.DataFrame:
    matrix = confusion_matrix(labels, predicted, labels=classes)
    return pd.DataFrame(matrix, index=classes, columns=classes)


def mistakes(texts, labels, predicted) -> pd.DataFrame:
    """Сегменты, на которых модель ошиблась — с самим текстом."""
    frame = pd.DataFrame({"текст": texts, "правильно": labels, "модель сказала": predicted})
    return frame[frame["правильно"] != frame["модель сказала"]].reset_index(drop=True)


def flipped(texts, labels, before: np.ndarray, after: np.ndarray) -> pd.DataFrame:
    """Сегменты, у которых ответ изменился между двумя конфигурациями.

    Средняя точность прячет как раз это: два набора признаков с одинаковым
    числом могут ошибаться на совершенно разных сегментах.
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
    """Признаки с наибольшим весом для каждого класса.

    Считается по логистической регрессии независимо от выбранного
    классификатора: у наивного Байеса и SVM веса означают разное, а вопрос
    «на что модель смотрит» один.
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
    """Точность как функция длины n-граммы — для слов и для символов.

    Это и есть главный результат работы № 1 в виде кривой: словарные признаки
    упираются в потолок рано, символьные продолжают расти.
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


# Набор из раздела 8 лабораторной работы. Значения в комментариях — то, что
# работа документирует для логистической регрессии; тесты это проверяют.
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
