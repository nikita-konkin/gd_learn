"""How a distribution becomes text.

The three knobs every language model has, from an n-gram one to a transformer.
They are applied in the order real implementations use: temperature reshapes
the distribution, top-k and top-p cut off its tail, and only then is the die
cast.
"""

from __future__ import annotations

import random

# Below this temperature we simply take the most likely character: raising
# probabilities to the power 1/T is numerically meaningless for a tiny T.
GREEDY_TEMPERATURE = 0.01


def apply_temperature(distribution: dict[str, float], temperature: float) -> dict[str, float]:
    """T < 1 sharpens the distribution, T > 1 flattens it.

    As T approaches 0 only the single most likely character survives: the text
    becomes predictable and starts looping. At a large T the distribution tends
    towards uniform and the text turns into noise.
    """
    if temperature <= GREEDY_TEMPERATURE:
        best = max(distribution, key=lambda key: distribution[key])
        return {character: (1.0 if character == best else 0.0) for character in distribution}

    reshaped = {
        character: probability ** (1.0 / temperature)
        for character, probability in distribution.items()
    }
    total = sum(reshaped.values())
    if total <= 0:
        return dict.fromkeys(distribution, 1.0 / len(distribution))
    return {character: value / total for character, value in reshaped.items()}


def apply_top_k(distribution: dict[str, float], k: int) -> dict[str, float]:
    """Keep the k most likely characters and zero the rest."""
    if k <= 0 or k >= len(distribution):
        return dict(distribution)

    kept = sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:k]
    total = sum(value for _, value in kept)
    if total <= 0:
        return dict(distribution)
    survivors = dict(kept)
    return {
        character: survivors.get(character, 0.0) / total for character in distribution
    }


def apply_top_p(distribution: dict[str, float], p: float) -> dict[str, float]:
    """Nucleus sampling: keep the smallest set whose mass adds up to p.

    Unlike top-k, the size of that set adapts to the distribution: where the
    model is confident one or two characters survive, where it is not, a dozen.
    """
    if p >= 1.0:
        return dict(distribution)

    ordered = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    kept: dict[str, float] = {}
    accumulated = 0.0
    for character, probability in ordered:
        kept[character] = probability
        accumulated += probability
        if accumulated >= p:
            break

    total = sum(kept.values())
    if total <= 0:
        return dict(distribution)
    return {character: kept.get(character, 0.0) / total for character in distribution}


def prepare(
    distribution: dict[str, float],
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> dict[str, float]:
    """Temperature, then top-k, then top-p — in that order."""
    prepared = apply_temperature(distribution, temperature)
    prepared = apply_top_k(prepared, top_k)
    return apply_top_p(prepared, top_p)


def sample(distribution: dict[str, float], generator: random.Random) -> str:
    """Cast the die over the prepared distribution."""
    characters = list(distribution)
    weights = [distribution[character] for character in characters]
    return generator.choices(characters, weights=weights, k=1)[0]
