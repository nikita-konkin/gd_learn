import pytest

from mt_playground.checks import NEG_RU_MORPHOLOGY, formal_checks, placeholders
from mt_playground.corpus import blind_spots, coverage_table, load_corpus, segment_label
from mt_playground.metrics import bleu, chrf, edit_distance, ter, tokenize
from mt_playground.mutations import (
    MUTATIONS,
    break_placeholder,
    drop_last_sentence,
    drop_negation,
    shuffle_words,
    swap_synonym,
)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


# --- measures --------------------------------------------------------------


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Сохранить изменения!") == ["сохранить", "изменения"]


def test_identical_strings_score_perfectly():
    assert bleu("Сохранить изменения", "Сохранить изменения") == pytest.approx(1.0)
    assert chrf("Сохранить изменения", "Сохранить изменения") == pytest.approx(1.0)
    assert ter("Сохранить изменения", "Сохранить изменения") == pytest.approx(0.0)


def test_empty_input_scores_zero():
    assert bleu("", "Сохранить") == 0.0
    assert chrf("", "Сохранить") == 0.0


def test_bleu_punishes_lost_negation():
    with_negation = bleu("Не удалось подключиться", "Не удалось подключиться к серверу")
    without = bleu("Удалось подключиться", "Не удалось подключиться к серверу")
    assert without < with_negation


def test_chrf_is_more_forgiving_than_bleu_on_inflection():
    """Character n-grams survive a change of word form; word n-grams do not."""
    hypothesis, reference = "Сохранение изменений", "Сохранить изменения"
    assert chrf(hypothesis, reference) > bleu(hypothesis, reference)


def test_edit_distance_matches_known_values():
    assert edit_distance("кот", "код") == 1
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "") == 3


def test_ter_counts_edits_per_reference_word():
    assert ter("а б в", "а б г") == pytest.approx(1 / 3)


# --- formal checks ---------------------------------------------------------


def test_broken_placeholder_is_detected():
    fired = formal_checks('Delete file "{name}"?', 'Удалить файл "(имя}"?')
    assert "плейсхолдеры" in fired


def test_clean_translation_fires_nothing():
    assert formal_checks("Save changes", "Сохранить изменения") == []


def test_number_mismatch_is_detected():
    assert "числа" in formal_checks("Try it for 30 days", "Попробуйте 14 дней")


def test_short_translation_is_detected():
    assert "подозрительно коротко" in formal_checks("A fairly long source string", "Кор")


def test_placeholders_are_extracted_in_order():
    assert placeholders("{a} then %s then <b>") == ["{a}", "%s", "<b>"]


# --- corpus: the numbers have to match the lab ------------------------------


def test_corpus_loads_all_segments(corpus):
    assert len(corpus) == 160
    assert set(corpus["type"].unique()) == {"интерфейс", "документация", "маркетинг", "юридический"}


def test_mean_scores_match_the_course(corpus):
    """Lab 3 documents a mean BLEU of 0.46 and chrF of 0.64."""
    assert corpus["BLEU"].mean() == pytest.approx(0.46, abs=0.005)
    assert corpus["chrF"].mean() == pytest.approx(0.64, abs=0.005)


def test_formal_checks_fire_on_eight_segments(corpus):
    """Lab 3: the checks flag 5 % of the corpus against 49 % for a BLEU threshold."""
    assert int(corpus["есть_замечания"].sum()) == 8

    below_median = (corpus["BLEU"] < corpus["BLEU"].median()).sum()
    assert below_median / len(corpus) == pytest.approx(0.49, abs=0.01)


def test_the_three_real_placeholder_breakages_are_caught(corpus):
    """О_данных.md: the model broke exactly three placeholders."""
    broken = corpus[corpus["проверки"].apply(lambda checks: "плейсхолдеры" in checks)]
    assert set(broken["id"]) == {"s002", "s012", "s028"}


def test_blind_spot_contains_the_broken_placeholder(corpus):
    """s002 is a broken string that BLEU lets through above the median."""
    missed = blind_spots(corpus, corpus["BLEU"].median())
    assert "s002" in set(missed["id"])


def test_morphology_rule_removes_only_the_false_positives():
    """«Невозможно» and «отсутствуют» are negations lab 3's rule cannot see."""
    base = load_corpus()
    fixed = load_corpus(NEG_RU_MORPHOLOGY)

    removed = set(base[base["есть_замечания"]]["id"]) - set(fixed[fixed["есть_замечания"]]["id"])
    assert removed == {"s010", "s031"}
    # s083 is a genuine lost sentence; it has to stay flagged.
    assert "s083" in set(fixed[fixed["есть_замечания"]]["id"])


def test_coverage_table_orders_tools_by_cost(corpus):
    table = coverage_table(corpus, corpus["BLEU"].median(), 0.70)
    by_tool = dict(zip(table["средство"], table["на проверку"], strict=True))

    formal = by_tool["формальные проверки"]
    by_bleu = by_tool[f"BLEU < {corpus['BLEU'].median():.2f}"]

    assert formal == 8
    assert by_bleu == 78
    # 49 % of the corpus against 5 % — nearly tenfold the manual work.
    assert by_bleu / formal == pytest.approx(9.75, abs=0.5)


def test_segment_label_truncates_long_sources(corpus):
    label = segment_label(corpus.iloc[0])
    assert label.startswith("s001 · интерфейс · ")
    assert all(len(segment_label(row)) < 80 for _, row in corpus.iterrows())


# --- edits -----------------------------------------------------------------


def test_break_placeholder_keeps_text_but_breaks_the_check():
    text = 'Удалить файл "{name}"?'
    broken, _ = break_placeholder(text)

    assert broken != text
    assert "плейсхолдеры" in formal_checks('Delete file "{name}"?', broken)


def test_break_placeholder_barely_moves_bleu():
    """The heart of the lab: a critical breakage the measure barely registers."""
    reference = 'Удалить файл "{name}"?'
    broken, _ = break_placeholder(reference)

    assert bleu(broken, reference) > 0.5


def test_mutations_return_none_when_not_applicable():
    assert break_placeholder("без плейсхолдеров") is None
    assert drop_negation("здесь нечего убирать") is None
    assert drop_last_sentence("Одно предложение") is None
    assert swap_synonym("нет знакомых слов") is None
    assert shuffle_words("одно") is None


def test_drop_negation_removes_the_negation():
    result, _ = drop_negation("Не удалось подключиться")
    assert result == "удалось подключиться"


def test_drop_last_sentence_keeps_the_first():
    result, _ = drop_last_sentence("Попробуйте бесплатно. Карта не нужна.")
    assert result == "Попробуйте бесплатно."


def test_synonym_swap_preserves_meaning_but_costs_bleu():
    reference = "Сохранить изменения"
    swapped, _ = swap_synonym(reference)

    assert swapped != reference
    assert formal_checks("Save changes", swapped) == []
    assert bleu(swapped, reference) < 1.0


def test_shuffle_words_keeps_the_same_multiset():
    result, _ = shuffle_words("а б в г")
    assert sorted(result.split()) == ["а", "б", "в", "г"]


def test_every_mutation_is_wired_with_a_label_and_hint():
    for label, mutate, hint in MUTATIONS:
        assert label and hint
        assert callable(mutate)
