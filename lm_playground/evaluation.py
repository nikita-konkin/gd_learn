"""Measurements: how much the model predicts, and how much it copies.

The playground's main point rests on two numbers that pull apart. As the order
n grows the text looks better and better — and a larger and larger share of it
turns out to be copied verbatim from the training corpus. Held-out perplexity
shows where the model stopped generalising and started memorising.
"""

from __future__ import annotations

from dataclasses import dataclass

from lm_playground.generation import generate
from lm_playground.model import CharNgramLM


def perplexity(model: CharNgramLM, texts: list[str]) -> float:
    """Read perplexity as "how many characters the model is wavering between".

    1.0 means always certain; the size of the alphabet means it knows nothing.
    """
    total_log = 0.0
    total_count = 0
    for text in texts:
        log_likelihood, count = model.log_likelihood(text)
        total_log += log_likelihood
        total_count += count
    if total_count == 0:
        return float("nan")
    return float(2 ** (-total_log / total_count))


def longest_copied_span(generated: str, source: str) -> str:
    """The longest run of generated text taken verbatim from the corpus.

    A naive grow-and-search: instant at teaching volumes, and readable without
    commentary, unlike a suffix automaton.
    """
    best = ""
    for start in range(len(generated)):
        # Worth continuing only while the span still occurs at all.
        length = len(best) + 1
        while start + length <= len(generated) and generated[start : start + length] in source:
            best = generated[start : start + length]
            length += 1
    return best


def copied_mask(generated: str, source: str, minimum: int = 10) -> list[bool]:
    """Per character: is it inside a verbatim match of at least ``minimum``.

    The threshold stops every «не» and « и » from counting as a match. The mask
    is returned separately from the fraction because the interface uses it to
    highlight the copied runs in the text itself.
    """
    covered = [False] * len(generated)
    for start in range(len(generated)):
        length = minimum
        matched = 0
        while start + length <= len(generated) and generated[start : start + length] in source:
            matched = length
            length += 1
        for index in range(start, start + matched):
            covered[index] = True
    return covered


def copied_fraction(generated: str, source: str, minimum: int = 10) -> float:
    """Share of characters inside verbatim matches of at least ``minimum``."""
    if not generated:
        return 0.0
    covered = copied_mask(generated, source, minimum)
    return sum(covered) / len(generated)


@dataclass
class OrderResult:
    order: int
    train_perplexity: float
    heldout_perplexity: float
    copied_fraction: float
    longest_copied: int


def sweep_orders(
    train_texts: list[str],
    heldout_texts: list[str],
    orders: range,
    sample_length: int = 300,
    seed: int = 0,
) -> list[OrderResult]:
    """Run the model at every order — this is the overfitting curve.

    Training perplexity falls towards one, held-out perplexity turns back
    upwards, and the copied share climbs monotonically. Three curves, one plot.
    """
    source = "\n".join(train_texts)
    results = []
    for order in orders:
        model = CharNgramLM(order).fit(train_texts)
        generated = generate(model, length=sample_length, seed=seed).text
        results.append(
            OrderResult(
                order=order,
                train_perplexity=perplexity(model, train_texts),
                heldout_perplexity=perplexity(model, heldout_texts),
                copied_fraction=copied_fraction(generated, source),
                longest_copied=len(longest_copied_span(generated, source)),
            )
        )
    return results
