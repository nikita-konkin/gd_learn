"""Строковые метрики качества перевода.

`bleu` и `chrf` повторяют реализацию из лабораторной работы № 3 буква в
букву: те же сглаживание, штраф за краткость и усреднение F-меры. Значения
на учебном корпусе совпадают с задокументированными (BLEU 0.46, chrF 0.64).

`ter` в лабораторной работе нет — это третья метрика, добавленная, чтобы
студент видел, что метрики расходятся не только с человеком, но и между
собой.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Слова в нижнем регистре, без пунктуации."""
    return TOKEN.findall(str(text).lower())


def ngrams(sequence: Sequence, n: int) -> Counter:
    """Счётчик n-грамм."""
    return Counter(tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1))


def bleu(hypothesis: str, reference: str, max_n: int = 4, smoothing: float = 1.0) -> float:
    """Упрощённый BLEU для одного предложения, значение от 0 до 1."""
    hyp, ref = tokenize(hypothesis), tokenize(reference)
    if not hyp or not ref:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams, ref_ngrams = ngrams(hyp, n), ngrams(ref, n)
        total = sum(hyp_ngrams.values())
        if total == 0:  # гипотеза короче, чем n
            continue
        # сглаживание: без него одна нулевая точность обнуляет весь BLEU
        matched = sum(min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items())
        precisions.append((matched + smoothing) / (total + smoothing))

    if not precisions:
        return 0.0

    geometric_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    penalty = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / max(len(hyp), 1))
    return float(geometric_mean * penalty)


def chrf(hypothesis: str, reference: str, max_n: int = 6, beta: float = 2.0) -> float:
    """chrF: F-мера по символьным n-граммам, значение от 0 до 1."""
    hyp = " ".join(tokenize(hypothesis))
    ref = " ".join(tokenize(reference))
    if not hyp or not ref:
        return 0.0

    f_scores = []
    for n in range(1, max_n + 1):
        hyp_ngrams, ref_ngrams = ngrams(hyp, n), ngrams(ref, n)
        if not hyp_ngrams or not ref_ngrams:
            continue
        matched = sum(min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items())
        precision = matched / sum(hyp_ngrams.values())
        recall = matched / sum(ref_ngrams.values())
        if precision + recall == 0:
            f_scores.append(0.0)
        else:
            f_scores.append((1 + beta**2) * precision * recall / (beta**2 * precision + recall))

    return float(sum(f_scores) / len(f_scores)) if f_scores else 0.0


def edit_distance(source: Sequence, target: Sequence) -> int:
    """Расстояние Левенштейна по элементам последовательности."""
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous = list(range(len(target) + 1))
    for i, source_item in enumerate(source, start=1):
        current = [i]
        for j, target_item in enumerate(target, start=1):
            current.append(
                min(
                    previous[j] + 1,  # удаление
                    current[j - 1] + 1,  # вставка
                    previous[j - 1] + (source_item != target_item),  # замена
                )
            )
        previous = current
    return previous[-1]


def ter(hypothesis: str, reference: str) -> float:
    """Упрощённый TER: правок на слово эталона, без сдвигов блоков.

    В отличие от BLEU и chrF, меньше — лучше. Значение больше 1 означает,
    что правок нужно больше, чем слов в эталоне.
    """
    hyp, ref = tokenize(hypothesis), tokenize(reference)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(edit_distance(hyp, ref) / len(ref))
