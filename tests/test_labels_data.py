"""What the playground ships, checked against where it came from.

The four data files are written offline by ``scripts/prepare_labels_data.py``.
These tests pin them to the numbers the lab notebooks print when executed, and
recompute whatever is cheap enough to recompute here, so a stale file fails CI
instead of teaching a number the lab no longer produces.
"""

import re

import numpy as np
import pandas as pd
import pytest

from labels_playground.corpus import load_annotations, load_corpus, load_newsgroups_curve, load_remedies
from mt_playground.corpus import load_corpus as load_scored_mt_corpus


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def annotations():
    return load_annotations()


def test_the_corpus_is_the_course_corpus(corpus):
    assert len(corpus) == 160
    assert corpus["type"].value_counts().to_dict() == {
        "интерфейс": 40,
        "документация": 40,
        "маркетинг": 40,
        "юридический": 40,
    }


def test_stored_scores_match_the_metrics_the_mt_playground_verifies(corpus):
    """The mt playground's BLEU is checked against lab 3; this file must not drift from it."""
    scored = load_scored_mt_corpus().set_index("id")
    stored = corpus.set_index("id")

    np.testing.assert_allclose(stored["bleu"], scored.loc[stored.index, "BLEU"], atol=1e-6)
    np.testing.assert_allclose(stored["chrf"], scored.loc[stored.index, "chrF"], atol=1e-6)


def test_the_median_is_the_one_section_7_cuts_at(corpus):
    assert corpus["bleu"].median() == pytest.approx(0.366, abs=0.001)


def test_the_annotations_are_the_labs_worked_example(annotations):
    assert list(annotations["id"]) == [
        "s002", "s012", "s028", "s083", "s115", "s003", "s026", "s114", "s006", "s065",
    ]  # fmt: skip
    assert annotations["severity"].value_counts().to_dict() == {
        "критическая": 5,
        "серьёзная": 3,
        "незначительная": 1,
        "нет": 1,
    }
    assert not annotations["ru_mt"].isna().any(), "every annotated id must exist in the corpus"


def test_the_second_annotator_departs_on_exactly_the_two_disputed_segments(annotations):
    disputed = annotations[annotations["category"] != annotations["category_second"]]

    pairs = zip(disputed["category"], disputed["category_second"], strict=True)
    assert dict(zip(disputed["id"], pairs, strict=True)) == {
        "s003": ("точность", "терминология"),
        "s006": ("терминология", "стиль"),
    }


def test_the_teachers_key_is_not_shipped():
    """The lab 3 error key is not given to students. Nothing here may carry it."""
    from labels_playground.corpus import DATA_DIR

    names = {path.name for path in DATA_DIR.iterdir()}
    assert names == {"corpus.csv", "mqm_examples.csv", "remedies.csv", "newsgroups_curve.csv"}


def test_the_remedies_are_what_lab_1_prints():
    remedies = load_remedies().set_index("remedy")

    assert remedies.loc["hyperparameters", "after"] == pytest.approx(0.744, abs=0.0005)
    assert remedies.loc["hyperparameters", "gain"] == pytest.approx(0.0, abs=0.0005)
    assert remedies.loc["feature_union", "after"] == pytest.approx(0.725, abs=0.0005)
    assert remedies.loc["feature_union", "gain"] == pytest.approx(-0.019, abs=0.0005)
    assert remedies.loc["more_data", "before"] == pytest.approx(0.546, abs=0.0005)
    assert remedies.loc["more_data", "after"] == pytest.approx(0.830, abs=0.0005)


def test_the_newsgroups_curve_is_what_lab_1_prints():
    curve = load_newsgroups_curve()

    assert list(curve["train_size"]) == [160, 320, 640, 1280, 1800]
    assert list(curve["accuracy"]) == pytest.approx([0.546, 0.732, 0.786, 0.821, 0.830], abs=0.0005)


def test_the_feature_union_recomputes_to_the_stored_number(corpus):
    """Cheap enough to redo here, so the stored value is checked, not trusted."""
    nltk = pytest.importorskip("nltk")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline, make_union

    stemmer = nltk.stem.snowball.SnowballStemmer("russian")
    token = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

    def tokenize_and_stem(text):
        return [stemmer.stem(word) for word in token.findall(text.lower())]

    union = make_union(
        TfidfVectorizer(tokenizer=tokenize_and_stem, token_pattern=None),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)),
    )
    score = cross_val_score(
        make_pipeline(union, LogisticRegression(max_iter=2000)), corpus["ru_ref"], corpus["type"], cv=5
    ).mean()

    stored = load_remedies().set_index("remedy").loc["feature_union", "after"]
    assert score == pytest.approx(stored, abs=0.0005)


def test_mqm_examples_file_has_english_keys():
    """Column keys are English; the values are the lab's Russian labels."""
    from labels_playground.corpus import MQM_PATH

    assert list(pd.read_csv(MQM_PATH).columns) == ["id", "category", "severity", "comment", "category_second"]
