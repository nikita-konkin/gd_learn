"""Порождение текста символ за символом."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from lm_playground.model import BOUNDARY, CharNgramLM
from lm_playground.sampling import prepare, sample


@dataclass
class GenerationStep:
    """Один шаг генерации — чтобы показать, из чего складывался выбор."""

    context: str
    chosen: str
    distribution: dict[str, float]
    support: int  # сколько раз контекст встречался в обучении


@dataclass
class Generation:
    text: str
    steps: list[GenerationStep] = field(default_factory=list)


def generate(
    model: CharNgramLM,
    length: int = 240,
    prompt: str = "",
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 0,
    keep_steps: int = 0,
) -> Generation:
    """Сгенерировать текст, при желании сохранив разбор первых шагов.

    `keep_steps` управляет тем, сколько шагов запомнить для показа
    распределения: хранить все 240 незачем, а первые несколько объясняют
    механику.
    """
    generator = random.Random(seed)
    context = prompt
    produced: list[str] = []
    steps: list[GenerationStep] = []

    for index in range(length):
        raw = model.distribution(context)
        prepared = prepare(raw, temperature=temperature, top_k=top_k, top_p=top_p)
        character = sample(prepared, generator)

        if index < keep_steps:
            steps.append(
                GenerationStep(
                    context=context[-model.order :],
                    chosen=character,
                    distribution=prepared,
                    support=model.context_support(context),
                )
            )

        produced.append(character)
        context += character

    return Generation(text="".join(produced), steps=steps)


def split_segments(text: str) -> list[str]:
    """Разбить сгенерированный поток на сегменты по границам."""
    return [part.strip() for part in text.split(BOUNDARY) if part.strip()]
