"""Формальные проверки перевода.

Повторяют раздел 8 лабораторной работы № 3. Это не метрика: проверка либо
сработала, либо нет. На учебном корпусе срабатывают на 8 сегментах из 160 —
5 % против 49 % у порога по BLEU.
"""

from __future__ import annotations

import re

PLACEHOLDER = re.compile(r"\{[^}]*\}|%[sd]|<[^>]+>")
NUMBER = re.compile(r"\d+")
NEG_EN = re.compile(r"\b(not|no|never|cannot|can't|don't|doesn't|unable|without|nor)\b", re.I)

# Правило из лабораторной работы № 3, буква в букву.
NEG_RU = re.compile(r"\b(не|нет|ни|нельзя|без|никогда|никаких|отсутствует|запрещ\w*)\b", re.I)

# То же правило, но с учётом русской морфологии. На учебном корпусе исходное
# правило даёт две ложные тревоги: «Невозможно подключиться» (отрицание сращено
# с основой, `\bне\b` его не видит) и «Результаты отсутствуют» (в правиле
# захардкожена только форма «отсутствует»). Автор уже писал `запрещ\w*`, то есть
# про словоизменение знал — здесь то же самое доведено до конца.
NEG_RU_MORPHOLOGY = re.compile(
    r"\b(не|нет|ни|нельзя|без|никогда|никаких|ничего|отсутств\w*|невозможн\w*|запрещ\w*)\b",
    re.I,
)

# Порядок задаёт порядок вывода в интерфейсе.
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
    """Возвращает список сработавших проверок. Пустой список — нареканий нет.

    `negation_rule` по умолчанию совпадает с лабораторной работой, поэтому
    значения по корпусу воспроизводятся один в один. Передайте
    `NEG_RU_MORPHOLOGY`, чтобы увидеть, как две ложные тревоги исчезают.
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
    """Плейсхолдеры в порядке появления — для подсветки в интерфейсе."""
    return PLACEHOLDER.findall(str(text))
