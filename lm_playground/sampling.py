"""Как из распределения получается текст.

Три ручки, которые есть у любой языковой модели — от n-граммной до
трансформера. Порядок применения тот же, что в реальных реализациях:
температура меняет форму распределения, top-k и top-p отрезают хвост,
и только потом бросается жребий.
"""

from __future__ import annotations

import random

# Ниже этой температуры считаем, что выбирается просто самый вероятный символ:
# возводить вероятности в степень 1/T при крошечном T бессмысленно численно.
GREEDY_TEMPERATURE = 0.01


def apply_temperature(distribution: dict[str, float], temperature: float) -> dict[str, float]:
    """T < 1 заостряет распределение, T > 1 разглаживает.

    При T → 0 остаётся один самый вероятный символ: текст становится
    предсказуемым и зацикливается. При большом T распределение стремится к
    равномерному, и текст превращается в шум.
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
    """Оставляет k самых вероятных символов, остальным — ноль."""
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
    """Nucleus sampling: оставляет минимальный набор с суммарной массой p.

    В отличие от top-k размер набора подстраивается под распределение: там, где
    модель уверена, остаётся один-два символа, где не уверена — десяток.
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
    """Температура, затем top-k, затем top-p — в этом порядке."""
    prepared = apply_temperature(distribution, temperature)
    prepared = apply_top_k(prepared, top_k)
    return apply_top_p(prepared, top_p)


def sample(distribution: dict[str, float], generator: random.Random) -> str:
    """Бросок жребия по подготовленному распределению."""
    characters = list(distribution)
    weights = [distribution[character] for character in characters]
    return generator.choices(characters, weights=weights, k=1)[0]
