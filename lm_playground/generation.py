"""Generating text one character at a time."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from lm_playground.model import BOUNDARY, CharNgramLM
from lm_playground.sampling import prepare, sample


@dataclass
class GenerationStep:
    """One step of generation, kept so the choice behind it can be shown."""

    context: str
    chosen: str
    distribution: dict[str, float]
    support: int  # how many times the context occurred in training


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
    """Generate text, optionally keeping a breakdown of the first steps.

    ``keep_steps`` says how many steps to remember for showing the
    distribution: there is no reason to keep all 240, and the first few explain
    the mechanics.
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
    """Split the generated stream into segments on the boundary character."""
    return [part.strip() for part in text.split(BOUNDARY) if part.strip()]
