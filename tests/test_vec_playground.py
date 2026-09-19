import pytest

from vec_playground.corpus import CONTENT_TYPES, texts_and_labels
from vec_playground.evaluation import (
    CLASSIFIERS,
    baseline_accuracy,
    build_classifier,
    confusion,
    evaluate,
    flipped,
    lab_comparison,
    mistakes,
    predictions,
    sweep_ngrams,
    top_features,
)
from vec_playground.features import FeatureSettings, build_vectorizer, tokenize_and_stem, tokenize_words

LOGISTIC = "логистическая регрессия"


@pytest.fixture(scope="module")
def data():
    return texts_and_labels()


@pytest.fixture(scope="module")
def lab_table(data):
    return lab_comparison(*data, LOGISTIC)


# --- corpus ----------------------------------------------------------------


def test_corpus_is_balanced(data):
    texts, labels = data

    assert len(texts) == 160
    assert set(labels.unique()) == set(CONTENT_TYPES)
    assert labels.value_counts().unique().tolist() == [40]


def test_baseline_is_one_class_in_four(data):
    assert baseline_accuracy(data[1]) == pytest.approx(0.25, abs=0.01)


# --- features --------------------------------------------------------------


def test_tokenizer_lowercases_and_splits_on_punctuation():
    assert tokenize_words("Изменения сохранены. Сохранить?") == ["изменения", "сохранены", "сохранить"]


def test_stemming_merges_inflected_forms():
    stems = tokenize_and_stem("Сохранить сохранены сохранение")

    assert len(set(stems)) < 3, "forms of one word have to collapse together"


def test_stemming_is_ignored_for_character_ngrams():
    """A tokenizer does not apply to character n-grams, and the knob must not lie."""
    settings = FeatureSettings(analyzer="символы внутри слов", ngram_min=2, ngram_max=4, stemming=True)

    assert build_vectorizer(settings).tokenizer is None


def test_stemming_reaches_the_vectorizer_for_words():
    assert build_vectorizer(FeatureSettings(stemming=True)).tokenizer is tokenize_and_stem


def test_ngram_range_is_never_inverted():
    """The slider must not allow min > max."""
    settings = FeatureSettings(ngram_min=3, ngram_max=1)

    assert build_vectorizer(settings).ngram_range == (3, 3)


def test_describe_mentions_the_active_switches():
    described = FeatureSettings(stemming=True, min_df=2, use_idf=False).describe()

    assert "стемминг" in described
    assert "min_df=2" in described
    assert "счётчики" in described


# --- reproducing the lab's numbers -----------------------------------------


def test_default_tfidf_matches_the_lab(data):
    """Lab 1 documents 0.53 for word TF-IDF."""
    assert evaluate(*data, FeatureSettings(), LOGISTIC).accuracy == pytest.approx(0.53, abs=0.006)


def test_stemming_matches_the_lab(data):
    """Lab 1: 0.60 with stemming."""
    assert evaluate(*data, FeatureSettings(stemming=True), LOGISTIC).accuracy == pytest.approx(0.60, abs=0.006)


def test_character_ngrams_match_the_lab(data):
    """Lab 1: character n-grams reach 0.74, the best result in the lab."""
    settings = FeatureSettings(analyzer="символы внутри слов", ngram_min=2, ngram_max=4)

    assert evaluate(*data, settings, LOGISTIC).accuracy == pytest.approx(0.74, abs=0.006)


def test_lab_table_covers_every_documented_configuration(lab_table):
    names = set(lab_table["конфигурация"])

    assert {"TF-IDF по умолчанию", "стемминг", "символьные 2-4"} <= names
    assert (lab_table["точность"] > 0.25).all(), "every configuration has to beat the baseline"


def test_character_ngrams_win_the_comparison(lab_table):
    best = lab_table.loc[lab_table["точность"].idxmax()]

    assert best["конфигурация"] == "символьные 2-4"


def test_character_ngrams_cost_far_more_features(lab_table):
    by_name = dict(zip(lab_table["конфигурация"], lab_table["признаков"], strict=True))

    assert by_name["символьные 2-4"] > 5 * by_name["TF-IDF по умолчанию"]


# --- the n-gram curve ------------------------------------------------------


@pytest.fixture(scope="module")
def sweep(data):
    return sweep_ngrams(*data, LOGISTIC, max_n=5)


def test_sweep_covers_both_analyzers(sweep):
    assert {point.analyzer for point in sweep} == {"слова", "символы внутри слов"}


def test_characters_beat_words_at_their_best(sweep):
    """The lab's headline result: character features pass above word features."""
    words = max(point.accuracy for point in sweep if point.analyzer == "слова")
    characters = max(point.accuracy for point in sweep if point.analyzer == "символы внутри слов")

    assert characters > words


def test_feature_count_grows_with_ngram_length(sweep):
    characters = [p for p in sweep if p.analyzer == "символы внутри слов"]
    counts = [point.features for point in sorted(characters, key=lambda p: p.ngram_max)]

    assert counts == sorted(counts)


# --- looking at the mistakes -----------------------------------------------


@pytest.fixture(scope="module")
def predicted(data):
    return predictions(*data, FeatureSettings(), LOGISTIC)


def test_confusion_matrix_is_square_and_totals_the_corpus(data, predicted):
    matrix = confusion(data[1], predicted, list(CONTENT_TYPES))

    assert matrix.shape == (4, 4)
    assert matrix.values.sum() == 160


def test_mistakes_lists_only_wrong_answers(data, predicted):
    wrong = mistakes(data[0], data[1], predicted)

    assert (wrong["правильно"] != wrong["модель сказала"]).all()
    assert 0 < len(wrong) < 160


def test_flipped_reports_what_changed_and_how(data, predicted):
    characters = FeatureSettings(analyzer="символы внутри слов", ngram_min=2, ngram_max=4)
    after = predictions(*data, characters, LOGISTIC)

    changed = flipped(data[0], data[1], predicted, after)
    assert not changed.empty, "changing the features has to move something"
    assert set(changed["итог"]) <= {"исправлено", "испорчено", "всё ещё неверно"}
    assert (changed["было"] != changed["стало"]).all()


def test_switching_to_character_ngrams_fixes_more_than_it_breaks(data, predicted):
    characters = FeatureSettings(analyzer="символы внутри слов", ngram_min=2, ngram_max=4)
    changed = flipped(data[0], data[1], predicted, predictions(*data, characters, LOGISTIC))
    counts = changed["итог"].value_counts()

    assert counts.get("исправлено", 0) > counts.get("испорчено", 0)


# --- everything else -------------------------------------------------------


def test_top_features_returns_one_row_per_class(data):
    table = top_features(*data, FeatureSettings())

    assert len(table) == 4
    assert set(table["класс"]) == set(CONTENT_TYPES)


@pytest.mark.parametrize("name", CLASSIFIERS)
def test_every_classifier_beats_the_baseline(data, name):
    assert build_classifier(name) is not None
    assert evaluate(*data, FeatureSettings(stemming=True), name).accuracy > 0.25
