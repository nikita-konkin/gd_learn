"""A character n-gram language model.

Works out how likely each next character is after a context of ``order``
characters. It is not a neural network, and that is the point: the whole model
fits in a dictionary of counts, it can be printed and read by eye, and the
questions it exists to answer — what a next-token distribution is, what
temperature does to one, how "fluent" differs from "memorised" — are the same
questions a transformer raises.

Probabilities are smoothed by Jelinek-Mercer interpolation: the estimate from a
context of length n is mixed with the estimate from length n-1, and so on down
to the uniform distribution. Without it any unseen character would have
probability zero and the held-out perplexity would be infinite.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable

# Weight of the longest context in the interpolation. 0.7 is the usual teaching
# choice: the highest order dominates without ever taking all the mass.
LAMBDA = 0.7

BOUNDARY = "\n"


class CharNgramLM:
    """A character n-gram model with interpolation smoothing."""

    def __init__(self, order: int = 3):
        if order < 1:
            raise ValueError("order должен быть не меньше 1")
        self.order = order
        self.counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        self.vocabulary: tuple[str, ...] = ()
        self.training_text = ""

    def fit(self, texts: Iterable[str]) -> CharNgramLM:
        """Training here is counting. There is no gradient descent in it."""
        joined = BOUNDARY.join(str(text) for text in texts) + BOUNDARY
        self.training_text = joined
        self.vocabulary = tuple(sorted(set(joined)))

        padded = BOUNDARY * self.order + joined
        for index in range(self.order, len(padded)):
            following = padded[index]
            # Count every order from 0 to order at once: the backoff needs them all.
            for back in range(self.order + 1):
                context = tuple(padded[index - back : index])
                self.counts[context][following] += 1
        return self

    def _probability(self, context: tuple[str, ...], character: str) -> float:
        if not context:
            counts = self.counts[()]
            total = sum(counts.values())
            uniform = 1.0 / len(self.vocabulary)
            if total == 0:
                return uniform
            unigram = counts.get(character, 0) / total
            return LAMBDA * unigram + (1 - LAMBDA) * uniform

        lower = self._probability(context[1:], character)
        counts = self.counts.get(context)
        if not counts:
            return lower
        total = sum(counts.values())
        maximum_likelihood = counts.get(character, 0) / total
        return LAMBDA * maximum_likelihood + (1 - LAMBDA) * lower

    def distribution(self, context: str) -> dict[str, float]:
        """The next-character distribution — the thing that gets sampled from."""
        if not self.vocabulary:
            raise RuntimeError("модель не обучена")

        trimmed = tuple((BOUNDARY * self.order + context)[-self.order :])
        raw = {character: self._probability(trimmed, character) for character in self.vocabulary}
        total = sum(raw.values())
        return {character: value / total for character, value in raw.items()}

    def context_support(self, context: str) -> int:
        """How many times this context occurred in the training text.

        Zero means the model has backed off to a shorter context. It is worth
        being able to see when the "model" in fact knows nothing any more.
        """
        trimmed = tuple((BOUNDARY * self.order + context)[-self.order :])
        return sum(self.counts.get(trimmed, Counter()).values())

    def log_likelihood(self, text: str) -> tuple[float, int]:
        """Total log2 likelihood of the text, and how many characters it has."""
        padded = BOUNDARY * self.order + str(text)
        total = 0.0
        count = 0
        for index in range(self.order, len(padded)):
            context = tuple(padded[index - self.order : index])
            probability = self._probability(context, padded[index])
            total += math.log2(max(probability, 1e-12))
            count += 1
        return total, count
