"""Измерения: насколько модель предсказывает и насколько она списывает.

Главный вывод playground'а держится на двух числах, которые расходятся.
С ростом порядка n текст на вид становится всё лучше — и всё большая его
часть оказывается дословно скопированной из обучающего корпуса. Перплексия
на отложенной выборке показывает, где модель перестала обобщать и начала
запоминать.
"""

from __future__ import annotations

from dataclasses import dataclass

from lm_playground.generation import generate
from lm_playground.model import CharNgramLM


def perplexity(model: CharNgramLM, texts: list[str]) -> float:
    """Перплексию читают как «между сколькими символами модель колеблется».

    1.0 — модель уверена всегда; размер алфавита — модель не знает ничего.
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
    """Самый длинный кусок сгенерированного текста, дословно взятый из корпуса.

    Наивный поиск с наращиванием: на учебных объёмах он мгновенный, а читается
    без комментариев, в отличие от суффиксного автомата.
    """
    best = ""
    for start in range(len(generated)):
        # Продолжать имеет смысл только пока кусок вообще встречается.
        length = len(best) + 1
        while start + length <= len(generated) and generated[start : start + length] in source:
            best = generated[start : start + length]
            length += 1
    return best


def copied_mask(generated: str, source: str, minimum: int = 10) -> list[bool]:
    """Для каждого символа: попал ли он в дословное совпадение длиной от `minimum`.

    Порог нужен, чтобы не считать совпадением каждое «не» и « и ». Маска
    возвращается отдельно от доли, потому что интерфейс ею же подсвечивает
    списанные куски прямо в тексте.
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
    """Доля символов, попавших в дословные совпадения длиной от `minimum`."""
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
    """Прогнать модель по всем порядкам — это и есть кривая переобучения.

    Обучающая перплексия падает до единицы, отложенная разворачивается вверх,
    а доля списанного растёт монотонно. Три кривые на одном графике.
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
