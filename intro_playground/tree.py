"""A decision tree on six features a linguist would think of.

The intro lecture's slide «Дерево решений: правила, которые можно прочесть»
trains this tree; the figure script that draws it lives in the course repo
(``Лекции/tex/собрать_иллюстрации.py``, function ``дерево``). The features and
the model settings here are that script's, so the accuracy is the slide's.

The features are computed with Python's ``re``, never with pandas' ``.str``
methods. Under pandas 3 strings are backed by pyarrow, whose regular expressions
treat ``\\b`` as an ASCII word boundary: next to a Cyrillic letter it never
matches, and two of the six features silently become all zeros.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from playground_common.compat import patch_pyarrow_stub

# Has to run before sklearn is first touched: in the browser every call that
# inspects its input fails otherwise.
patch_pyarrow_stub()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

INFINITIVE = re.compile(r"^[А-ЯЁ][а-яё]+(?:ть|ться)\b")
PLACEHOLDER = re.compile(r"\{|%[sd]|<")
POLITE_YOU = re.compile(r"\b[Вв](?:ы|ам|ас|аш\w*)\b")
LEGAL_MARKERS = re.compile(r"астоящ|торон|обязат")

WORDS = "words"
FINAL_PERIOD = "final_period"
FEATURES = (WORDS, FINAL_PERIOD, "infinitive", "placeholder", "polite_you", "legal_markers")
FEATURE_LABELS = {
    WORDS: "слов в сегменте",
    FINAL_PERIOD: "кончается точкой",
    "infinitive": "начинается с инфинитива",
    "placeholder": "есть плейсхолдер",
    "polite_you": "есть «вы», «ваш»",
    "legal_markers": "есть «настоящий», «стороны»",
}

DEPTHS = tuple(range(1, 7))
LECTURE_DEPTH = 3
FOLDS = 5
# Lab 1, section 9: character n-grams 2-4 with logistic regression. The number
# the tree is weighed against on the slide «Что мы узнали из дерева».
NGRAM_ACCURACY = 0.744


def feature_table(corpus: pd.DataFrame) -> pd.DataFrame:
    """One column per feature, one row per segment; all but the word count are 0/1."""
    ru = [str(text).strip() for text in corpus["ru_ref"]]
    en = [str(text) for text in corpus["en"]]
    return pd.DataFrame(
        {
            WORDS: [len(text.split()) for text in ru],
            FINAL_PERIOD: [int(text.endswith(".")) for text in ru],
            "infinitive": [int(bool(INFINITIVE.match(text))) for text in ru],
            "placeholder": [int(bool(PLACEHOLDER.search(text))) for text in en],
            "polite_you": [int(bool(POLITE_YOU.search(text))) for text in ru],
            "legal_markers": [int(bool(LEGAL_MARKERS.search(text))) for text in ru],
        },
        index=corpus.index,
    )


def build_tree(depth: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(max_depth=depth, random_state=0)


def tree_accuracy(features: pd.DataFrame, labels, depth: int) -> float:
    """Mean accuracy over five folds, as the figure script measures it."""
    return float(cross_val_score(build_tree(depth), features, labels, cv=FOLDS).mean())


def depth_sweep(features: pd.DataFrame, labels) -> pd.DataFrame:
    return pd.DataFrame(
        {"depth": list(DEPTHS), "accuracy": [tree_accuracy(features, labels, depth) for depth in DEPTHS]}
    )


@dataclass(frozen=True)
class Node:
    """One box of the drawn tree: a question, or a leaf with its verdict."""

    level: int
    branch: str  # "нет" / "да" from the parent; empty for the root
    question: str | None
    verdict: str | None
    segments: int
    purity: float  # share of the node's segments that belong to its majority type


def _question(feature: str, threshold: float) -> str:
    label = FEATURE_LABELS[feature]
    label = label[:1].upper() + label[1:]
    return f"{label}?" if threshold < 1 else f"{label} больше {threshold:.0f}?"


def tree_nodes(tree: DecisionTreeClassifier, feature_names: list[str]) -> list[Node]:
    """The fitted tree, depth first, left ("нет") before right ("да")."""
    structure, classes = tree.tree_, list(tree.classes_)
    nodes: list[Node] = []

    def visit(index: int, level: int, branch: str) -> None:
        # value holds class weights (fractions in recent sklearn); only the proportions matter.
        counts = structure.value[index][0]
        leaf = structure.children_left[index] == -1
        question = None if leaf else _question(feature_names[structure.feature[index]], structure.threshold[index])
        nodes.append(
            Node(
                level=level,
                branch=branch,
                question=question,
                verdict=classes[int(np.argmax(counts))] if leaf else None,
                segments=int(structure.n_node_samples[index]),
                purity=float(counts.max() / counts.sum()),
            )
        )
        if not leaf:
            visit(structure.children_left[index], level + 1, "нет")
            visit(structure.children_right[index], level + 1, "да")

    visit(0, 0, "")
    return nodes


def used_features(tree: DecisionTreeClassifier, feature_names: list[str]) -> list[str]:
    """Features the tree actually asks about, in the order it first asks."""
    seen: list[str] = []
    for index in tree.tree_.feature:
        if index >= 0 and feature_names[index] not in seen:
            seen.append(feature_names[index])
    return seen
