"""Formal checks on a translation.

These reproduce section 8 of lab 3. They are not a measure: a check either
fires or it does not. On the teaching corpus they fire on 8 segments out of
160 — 5 % against the 49 % a BLEU threshold flags.
"""

from __future__ import annotations

import re

PLACEHOLDER = re.compile(r"\{[^}]*\}|%[sd]|<[^>]+>")
NUMBER = re.compile(r"\d+")
NEG_EN = re.compile(r"\b(not|no|never|cannot|can't|don't|doesn't|unable|without|nor)\b", re.I)

# The rule from lab 3, to the letter.
NEG_RU = re.compile(r"\b(не|нет|ни|нельзя|без|никогда|никаких|отсутствует|запрещ\w*)\b", re.I)

# The same rule, but allowing for Russian morphology. On the teaching corpus the
# original gives two false alarms: «Невозможно подключиться», where the negation
# is fused into the stem and `\bне\b` cannot see it, and «Результаты
# отсутствуют», where only the form «отсутствует» is hardcoded. The author had
# already written `запрещ\w*`, so inflection was on their mind — this just
# carries the same idea through.
NEG_RU_MORPHOLOGY = re.compile(
    r"\b(не|нет|ни|нельзя|без|никогда|никаких|ничего|отсутств\w*|невозможн\w*|запрещ\w*)\b",
    re.I,
)

# This order is the order they are listed in the interface.
CHECK_NAMES = (
    "плейсхолдеры",
    "числа",
    "потеряно отрицание",
    "пусто",
    "подозрительно коротко",
)

CHECK_EXPLANATIONS = {
    "плейсхолдеры": "набор плейсхолдеров в переводе не совпадает с оригиналом — строка не соберётся",
    "числа": "числа в переводе не совпадают с числами оригинала",
    "потеряно отрицание": "в оригинале есть отрицание, в переводе его нет",
    "пусто": "перевод пуст",
    "подозрительно коротко": "перевод короче 45 % длины оригинала — возможна потеря текста",
}


def formal_checks(source: str, translation: str, negation_rule: re.Pattern = NEG_RU) -> list[str]:
    """The checks that fired. An empty list means nothing to complain about.

    ``negation_rule`` defaults to the lab's own, so the corpus-wide numbers come
    out identical to it. Pass ``NEG_RU_MORPHOLOGY`` to watch the two false
    alarms disappear.
    """
    source, translation = str(source), str(translation)
    fired = []
    if set(PLACEHOLDER.findall(source)) != set(PLACEHOLDER.findall(translation)):
        fired.append("плейсхолдеры")
    if sorted(NUMBER.findall(source)) != sorted(NUMBER.findall(translation)):
        fired.append("числа")
    if NEG_EN.search(source) and not negation_rule.search(translation):
        fired.append("потеряно отрицание")
    if not translation.strip():
        fired.append("пусто")
    if len(translation) < 0.45 * len(source):
        fired.append("подозрительно коротко")
    return fired


def placeholders(text: str) -> list[str]:
    """Placeholders in order of appearance, for highlighting in the interface."""
    return PLACEHOLDER.findall(str(text))
