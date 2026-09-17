"""Готовые правки перевода — кнопки «сломай это» для интерфейса.

Смысл кнопок в контрасте. Одни правки ломают строку так, что продукт не
соберётся, почти не задев BLEU. Другие сохраняют смысл полностью, но роняют
BLEU вдвое. Пока студент правит текст руками, он этот контраст ищет; кнопка
показывает его за один клик.
"""

from __future__ import annotations

import re

from mt_playground.checks import PLACEHOLDER

# Синонимы из словаря локализации: замена сохраняет смысл, но меняет форму.
SYNONYMS = {
    "сохранить": "записать",
    "удалить": "стереть",
    "файл": "документ",
    "настройки": "параметры",
    "изменения": "правки",
    "аккаунт": "учётная запись",
    "пароль": "код доступа",
    "ошибка": "сбой",
    "загрузка": "скачивание",
    "отменить": "прервать",
    "продолжить": "далее",
    "выбрать": "указать",
    "создать": "добавить",
    "закрыть": "свернуть",
    "поиск": "розыск",
}

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def break_placeholder(text: str) -> tuple[str, str] | None:
    """Ломает первую открывающую скобку плейсхолдера — ровно та ошибка,
    которую модель `opus-mt-en-ru` сделала сама в сегменте s002."""
    match = PLACEHOLDER.search(text)
    if match is None or not match.group().startswith("{"):
        return None
    broken = "(" + match.group()[1:]
    return text[: match.start()] + broken + text[match.end() :], "плейсхолдер сломан"


def drop_negation(text: str) -> tuple[str, str] | None:
    """Убирает первое «не» — смысл меняется на противоположный."""
    replaced, count = re.subn(r"\bне\s+", "", text, count=1, flags=re.IGNORECASE)
    if count == 0:
        return None
    return replaced, "отрицание потеряно"


def drop_last_sentence(text: str) -> tuple[str, str] | None:
    """Выбрасывает последнее предложение — так модель потеряла s083."""
    sentences = [s for s in SENTENCE_SPLIT.split(text.strip()) if s]
    if len(sentences) < 2:
        return None
    return " ".join(sentences[:-1]), "предложение потеряно"


def swap_synonym(text: str) -> tuple[str, str] | None:
    """Меняет одно слово на синоним: смысл сохранён, форма другая."""
    for word, synonym in SYNONYMS.items():
        pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)
        match = pattern.search(text)
        if match is None:
            continue
        replacement = synonym.capitalize() if match.group()[0].isupper() else synonym
        return pattern.sub(replacement, text, count=1), f"«{match.group()}» → «{replacement}»"
    return None


def shuffle_words(text: str) -> tuple[str, str] | None:
    """Переставляет два соседних слова: те же слова, другой порядок."""
    words = text.split()
    if len(words) < 2:
        return None
    middle = len(words) // 2
    words[middle - 1], words[middle] = words[middle], words[middle - 1]
    return " ".join(words), "два слова переставлены"


# Порядок задаёт порядок кнопок в интерфейсе.
MUTATIONS = (
    ("Сломать плейсхолдер", break_placeholder, "критическая поломка, почти незаметная для BLEU"),
    ("Убрать отрицание", drop_negation, "смысл на противоположный"),
    ("Потерять предложение", drop_last_sentence, "часть текста исчезает"),
    ("Заменить синонимом", swap_synonym, "смысл сохранён, BLEU падает"),
    ("Переставить слова", shuffle_words, "те же слова, другой порядок"),
)
