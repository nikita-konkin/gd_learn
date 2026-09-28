"""The decision tree of the intro lecture, checked against its slides.

The figure script in the course repo (``Лекции/tex/собрать_иллюстрации.py``,
``дерево``) trains the tree the slide «Дерево решений: правила, которые можно
прочесть» draws, and the slide after it quotes its accuracy, 0.631.
"""

import pandas as pd
import pytest

from intro_playground import app as intro_app
from intro_playground.corpus import load_corpus
from intro_playground.tree import (
    FEATURES,
    FINAL_PERIOD,
    INFINITIVE,
    LECTURE_DEPTH,
    build_tree,
    depth_sweep,
    feature_table,
    tree_accuracy,
    tree_nodes,
    used_features,
)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def features(corpus):
    return feature_table(corpus)


def test_every_feature_fires_somewhere(features):
    """Under pandas 3's pyarrow regex, \\b next to Cyrillic never matches and two features went all-zero."""
    assert features.sum().to_dict() == {
        "words": 949,
        "final_period": 120,
        "infinitive": 21,
        "placeholder": 3,
        "polite_you": 8,
        "legal_markers": 12,
    }


def test_pandas_string_regex_is_the_trap_the_module_avoids(corpus):
    """Documents why the features use Python's re: pandas' .str may run a different regex engine."""
    python_re = sum(bool(INFINITIVE.match(text.strip())) for text in corpus["ru_ref"])
    via_pandas = int(pd.Series(corpus["ru_ref"]).str.strip().str.match(INFINITIVE.pattern).sum())

    assert python_re == 21
    assert via_pandas in (0, 21)  # 0 under pandas 3 with pyarrow strings, 21 under pandas 2


def test_the_lecture_tree_scores_the_slides_accuracy(corpus, features):
    assert tree_accuracy(features, corpus["type"], LECTURE_DEPTH) == pytest.approx(0.631, abs=0.0005)


def test_deeper_is_not_better_for_long(corpus, features):
    sweep = depth_sweep(features, corpus["type"]).set_index("depth")["accuracy"].round(3)

    assert sweep.to_dict() == {1: 0.5, 2: 0.562, 3: 0.631, 4: 0.644, 5: 0.625, 6: 0.619}


def test_the_tree_learns_what_the_slide_says_it_learns(corpus, features):
    """Interface does not end in a full stop; legal markers mean legal text; long means documentation."""
    tree = build_tree(LECTURE_DEPTH).fit(features, corpus["type"])
    nodes = tree_nodes(tree, list(features.columns))

    assert nodes[0].question == "Кончается точкой?"
    assert (nodes[1].branch, nodes[1].verdict, nodes[1].segments, nodes[1].purity) == ("нет", "интерфейс", 40, 1.0)
    leaves = {node.verdict for node in nodes if node.verdict is not None}
    assert leaves == {"интерфейс", "маркетинг", "юридический", "документация"}
    assert used_features(tree, list(features.columns)) == ["final_period", "words", "legal_markers"]


def test_without_the_full_stop_the_tree_falls_back_on_the_infinitive(corpus, features):
    rest = features.drop(columns=[FINAL_PERIOD])
    tree = build_tree(LECTURE_DEPTH).fit(rest, corpus["type"])

    assert tree_accuracy(rest, corpus["type"], LECTURE_DEPTH) == pytest.approx(0.506, abs=0.0005)
    assert "infinitive" in used_features(tree, list(rest.columns))


def test_rules_render_as_a_nested_list(corpus, features):
    tree = build_tree(2).fit(features, corpus["type"])
    lines = intro_app._rules_markdown(tree_nodes(tree, list(features.columns))).splitlines()

    assert lines[0] == "- **Кончается точкой?**"
    assert lines[1] == "  - нет → интерфейс — 40 сегментов, 100% этого типа"
    assert lines[2].startswith("  - да → **Слов в сегменте больше 6?**")


def test_the_feature_list_is_the_slides_six():
    assert len(FEATURES) == 6
