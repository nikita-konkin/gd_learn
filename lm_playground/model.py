"""Символьная n-граммная языковая модель.

Считает, с какой вероятностью после контекста из `order` символов идёт каждый
следующий символ. Это не нейросеть, и в этом смысл: вся модель умещается в
словарь счётчиков, её можно распечатать и прочитать глазами, а вопросы, ради
которых модель тут стоит — что такое распределение следующего токена,
что делает температура, чем «складно» отличается от «выучено наизусть» —
одинаковы и для неё, и для трансформера.

Вероятности сглажены интерполяцией Йелинека-Мерсера: оценка по контексту
длины n смешивается с оценкой по контексту длины n-1 и так далее до
равномерного распределения. Без этого любой невиданный символ давал бы
нулевую вероятность и бесконечную перплексию на отложенной выборке.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable

# Вес старшего контекста при интерполяции. 0.7 — обычный учебный выбор:
# старший порядок доминирует, но никогда не забирает всю массу.
LAMBDA = 0.7

BOUNDARY = "\n"


class CharNgramLM:
    """Модель символьных n-грамм с интерполяционным сглаживанием."""

    def __init__(self, order: int = 3):
        if order < 1:
            raise ValueError("order должен быть не меньше 1")
        self.order = order
        self.counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        self.vocabulary: tuple[str, ...] = ()
        self.training_text = ""

    def fit(self, texts: Iterable[str]) -> CharNgramLM:
        """Обучение — это подсчёт. Никакого градиентного спуска здесь нет."""
        joined = BOUNDARY.join(str(text) for text in texts) + BOUNDARY
        self.training_text = joined
        self.vocabulary = tuple(sorted(set(joined)))

        padded = BOUNDARY * self.order + joined
        for index in range(self.order, len(padded)):
            following = padded[index]
            # Считаем сразу все порядки от 0 до order: они нужны для отката.
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
        """Распределение следующего символа — то самое, из чего потом сэмплируют."""
        if not self.vocabulary:
            raise RuntimeError("модель не обучена")

        trimmed = tuple((BOUNDARY * self.order + context)[-self.order :])
        raw = {character: self._probability(trimmed, character) for character in self.vocabulary}
        total = sum(raw.values())
        return {character: value / total for character, value in raw.items()}

    def context_support(self, context: str) -> int:
        """Сколько раз этот контекст встретился в обучающем тексте.

        Ноль означает, что модель откатилась на более короткий контекст:
        полезно видеть, когда «модель» на самом деле уже ничего не знает.
        """
        trimmed = tuple((BOUNDARY * self.order + context)[-self.order :])
        return sum(self.counts.get(trimmed, Counter()).values())

    def log_likelihood(self, text: str) -> tuple[float, int]:
        """Суммарный log2 правдоподобия текста и число символов в нём."""
        padded = BOUNDARY * self.order + str(text)
        total = 0.0
        count = 0
        for index in range(self.order, len(padded)):
            context = tuple(padded[index - self.order : index])
            probability = self._probability(context, padded[index])
            total += math.log2(max(probability, 1e-12))
            count += 1
        return total, count
