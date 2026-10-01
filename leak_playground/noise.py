"""The data the experiment runs on: noise, with an optional dose of real signal.

By default the data is pure noise and nothing else:

    generator = np.random.default_rng(1)
    noise_x = generator.normal(size=(300, 2000))
    noise_y = generator.integers(0, 2, 300)

``make_data(NoiseSettings())`` returns exactly those two arrays — same seed,
same draws, same order — so the default page is the same for everyone.

The ``signal`` setting is the playground's one addition. At its default of 0.0
nothing is added, so the arrays are untouched; above 0.0 the first
``INFORMATIVE`` columns are shifted for one class, which puts genuine signal
into the data without drawing from the generator again. That matters: another
draw would change every later value and the numbers would move for a setting
the student did not touch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# How many columns carry the signal, when there is any. Fixed rather than
# exposed: a second slider over the same cause is what made the sample
# repository's seed slider useless.
INFORMATIVE = 5


@dataclass(frozen=True)
class NoiseSettings:
    """Everything that decides the data, and nothing that decides the evaluation.

    Frozen and hashable, so ``lru_cache`` can key on it in the browser.
    """

    observations: int = 300
    features: int = 2000
    seed: int = 1
    signal: float = 0.0


def make_data(settings: NoiseSettings) -> tuple[np.ndarray, np.ndarray]:
    """The matrix and the labels. With ``signal == 0`` there is nothing to predict."""
    generator = np.random.default_rng(settings.seed)
    matrix = generator.normal(size=(settings.observations, settings.features))
    labels = generator.integers(0, 2, settings.observations)

    if settings.signal:
        informative = min(INFORMATIVE, settings.features)
        matrix[labels == 1, :informative] += settings.signal

    return matrix, labels
