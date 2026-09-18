import math

import pytest

from lm_playground.corpus import CONTENT_TYPES, corpus_statistics, load_texts, split
from lm_playground.evaluation import (
    copied_fraction,
    copied_mask,
    longest_copied_span,
    perplexity,
    sweep_orders,
)
from lm_playground.generation import generate, split_segments
from lm_playground.model import BOUNDARY, CharNgramLM
from lm_playground.sampling import apply_temperature, apply_top_k, apply_top_p, prepare


@pytest.fixture(scope="module")
def texts():
    return load_texts()


@pytest.fixture(scope="module")
def parts(texts):
    return split(texts, seed=0)


# --- корпус ----------------------------------------------------------------


def test_corpus_matches_the_localisation_set(texts):
    assert len(texts) == 160
    assert corpus_statistics(texts)["characters"] > 7000


def test_filtering_by_type_keeps_forty_segments_each():
    for content_type in CONTENT_TYPES:
        assert len(load_texts((content_type,))) == 40


def test_split_is_disjoint_and_deterministic(texts):
    train, heldout = split(texts, seed=0)

    assert set(train).isdisjoint(heldout)
    assert len(train) + len(heldout) == len(texts)
    assert split(texts, seed=0) == (train, heldout)


# --- модель ----------------------------------------------------------------


def test_order_must_be_positive():
    with pytest.raises(ValueError):
        CharNgramLM(0)


def test_distribution_is_a_probability_distribution(parts):
    model = CharNgramLM(3).fit(parts[0])
    distribution = model.distribution("Сохран")

    assert sum(distribution.values()) == pytest.approx(1.0)
    assert all(value > 0 for value in distribution.values()), "сглаживание обязано исключить нули"


def test_model_predicts_the_continuation_it_was_trained_on():
    model = CharNgramLM(4).fit(["Сохранить изменения"] * 5)
    distribution = model.distribution("Сохр")

    assert max(distribution, key=lambda key: distribution[key]) == "а"


def test_unseen_context_falls_back_instead_of_failing(parts):
    model = CharNgramLM(4).fit(parts[0])

    assert model.context_support("qqqq") == 0
    distribution = model.distribution("qqqq")
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_log_likelihood_is_negative_and_counts_every_character(parts):
    model = CharNgramLM(3).fit(parts[0])
    total, count = model.log_likelihood("Сохранить")

    assert count == len("Сохранить")
    assert total < 0


# --- сэмплирование ---------------------------------------------------------


def test_low_temperature_collapses_to_the_most_likely():
    distribution = {"а": 0.6, "б": 0.3, "в": 0.1}
    sharpened = apply_temperature(distribution, 0.001)

    assert sharpened == {"а": 1.0, "б": 0.0, "в": 0.0}


def test_high_temperature_flattens_the_distribution():
    distribution = {"а": 0.6, "б": 0.3, "в": 0.1}
    flattened = apply_temperature(distribution, 2.0)

    spread_before = max(distribution.values()) - min(distribution.values())
    spread_after = max(flattened.values()) - min(flattened.values())
    assert spread_after < spread_before
    assert sum(flattened.values()) == pytest.approx(1.0)


def test_temperature_one_changes_nothing():
    distribution = {"а": 0.6, "б": 0.3, "в": 0.1}

    assert apply_temperature(distribution, 1.0) == pytest.approx(distribution)


def test_top_k_keeps_exactly_k_survivors():
    distribution = {"а": 0.5, "б": 0.3, "в": 0.15, "г": 0.05}
    cut = apply_top_k(distribution, 2)

    assert sum(1 for value in cut.values() if value > 0) == 2
    assert cut["в"] == 0.0
    assert sum(cut.values()) == pytest.approx(1.0)


def test_top_k_zero_is_disabled():
    distribution = {"а": 0.5, "б": 0.3, "в": 0.2}

    assert apply_top_k(distribution, 0) == distribution


def test_top_p_adapts_the_survivor_count():
    confident = {"а": 0.95, "б": 0.03, "в": 0.02}
    unsure = {"а": 0.4, "б": 0.35, "в": 0.25}

    assert sum(1 for v in apply_top_p(confident, 0.9).values() if v > 0) == 1
    assert sum(1 for v in apply_top_p(unsure, 0.9).values() if v > 0) == 3


def test_prepare_applies_temperature_before_cutting():
    distribution = {"а": 0.5, "б": 0.3, "в": 0.2}
    prepared = prepare(distribution, temperature=0.5, top_k=2, top_p=1.0)

    assert prepared["в"] == 0.0
    assert sum(prepared.values()) == pytest.approx(1.0)


# --- генерация -------------------------------------------------------------


def test_generation_is_deterministic_for_a_seed(parts):
    model = CharNgramLM(3).fit(parts[0])

    assert generate(model, length=80, seed=7).text == generate(model, length=80, seed=7).text


def test_generation_respects_the_requested_length(parts):
    model = CharNgramLM(3).fit(parts[0])

    assert len(generate(model, length=120, seed=1).text) == 120


def test_generation_keeps_the_requested_number_of_steps(parts):
    model = CharNgramLM(3).fit(parts[0])
    generation = generate(model, length=50, seed=1, keep_steps=4)

    assert len(generation.steps) == 4
    assert all(sum(step.distribution.values()) == pytest.approx(1.0) for step in generation.steps)


def test_split_segments_drops_empty_pieces():
    assert split_segments(f"один{BOUNDARY}{BOUNDARY}два{BOUNDARY}") == ["один", "два"]


# --- измерения -------------------------------------------------------------


def test_perplexity_is_near_one_when_the_text_is_memorised():
    model = CharNgramLM(6).fit(["Сохранить изменения"] * 20)

    assert perplexity(model, ["Сохранить изменения"]) < 2.0


def test_perplexity_is_finite_on_unseen_text(parts):
    """Ради этого и сделано сглаживание."""
    model = CharNgramLM(5).fit(parts[0])

    assert math.isfinite(perplexity(model, ["совершенно невиданный текст"]))


def test_longest_copied_span_finds_the_verbatim_piece():
    assert longest_copied_span("xxСохранить измененияyy", "Сохранить изменения") == "Сохранить изменения"


def test_copied_mask_marks_only_long_matches():
    generated = "Сохранить изменения и что-то ещё"
    mask = copied_mask(generated, "Сохранить изменения")

    assert all(mask[: len("Сохранить изменения")])
    assert not mask[-1]


def test_copied_fraction_is_zero_for_unrelated_text():
    assert copied_fraction("абвгдеёжзийклмн", "Сохранить изменения") == 0.0


def test_copied_fraction_is_one_for_a_verbatim_copy():
    source = "Сохранить изменения"
    assert copied_fraction(source, source) == pytest.approx(1.0)


# --- главный вывод работы --------------------------------------------------


@pytest.fixture(scope="module")
def sweep(parts):
    return sweep_orders(parts[0], parts[1], range(1, 9), seed=3)


def test_training_perplexity_falls_as_context_grows(sweep):
    values = [result.train_perplexity for result in sweep]

    assert values == sorted(values, reverse=True)
    assert values[-1] < 2.0, "на длинном контексте модель просто помнит корпус"


def test_heldout_perplexity_has_a_minimum_in_the_middle(sweep):
    """Кривая переобучения: качество на невиданном тексте разворачивается вверх."""
    values = [result.heldout_perplexity for result in sweep]
    best = values.index(min(values))

    assert 0 < best < len(values) - 1, "минимум обязан быть внутри диапазона"
    assert values[-1] > min(values), "после минимума перплексия растёт"


def test_copying_grows_with_context(sweep):
    shares = [result.copied_fraction for result in sweep]

    assert shares[0] < 0.1, "на коротком контексте списывать нечего"
    assert shares[-1] > 0.9, "на длинном контексте текст почти целиком из корпуса"


def test_the_best_generalising_model_still_copies_little(sweep):
    """Связка, ради которой playground существует.

    Порядок с лучшей отложенной перплексией списывает единицы процентов, а
    самый «складный» на вид — почти всё.
    """
    best = min(sweep, key=lambda result: result.heldout_perplexity)
    greediest = sweep[-1]

    assert best.copied_fraction < 0.2
    assert greediest.copied_fraction > 4 * best.copied_fraction
