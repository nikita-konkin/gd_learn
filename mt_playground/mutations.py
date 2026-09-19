"""Ready-made edits — the "break it" buttons in the interface.

The point of the buttons is the contrast. Some edits break the string so badly
the product will not build, and barely move BLEU. Others preserve the meaning
completely and halve BLEU. Editing by hand, a student goes looking for that
contrast; a button shows it in one click.
"""

from __future__ import annotations

import re

from mt_playground.checks import PLACEHOLDER

# Synonyms from the localisation glossary: swapping one keeps the meaning and
# changes the form.
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
    """Break the first opening brace of a placeholder.

    Exactly the mistake ``opus-mt-en-ru`` made by itself on segment s002.
    """
    match = PLACEHOLDER.search(text)
    if match is None or not match.group().startswith("{"):
        return None
    broken = "(" + match.group()[1:]
    return text[: match.start()] + broken + text[match.end() :], "плейсхолдер сломан"


def drop_negation(text: str) -> tuple[str, str] | None:
    """Drop the first «не», which reverses the meaning."""
    replaced, count = re.subn(r"\bне\s+", "", text, count=1, flags=re.IGNORECASE)
    if count == 0:
        return None
    return replaced, "отрицание потеряно"


def drop_last_sentence(text: str) -> tuple[str, str] | None:
    """Drop the last sentence — this is how the model lost s083."""
    sentences = [s for s in SENTENCE_SPLIT.split(text.strip()) if s]
    if len(sentences) < 2:
        return None
    return " ".join(sentences[:-1]), "предложение потеряно"


def swap_synonym(text: str) -> tuple[str, str] | None:
    """Swap one word for a synonym: meaning kept, form changed."""
    for word, synonym in SYNONYMS.items():
        pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)
        match = pattern.search(text)
        if match is None:
            continue
        replacement = synonym.capitalize() if match.group()[0].isupper() else synonym
        return pattern.sub(replacement, text, count=1), f"«{match.group()}» → «{replacement}»"
    return None


def shuffle_words(text: str) -> tuple[str, str] | None:
    """Swap two neighbouring words: the same words in a different order."""
    words = text.split()
    if len(words) < 2:
        return None
    middle = len(words) // 2
    words[middle - 1], words[middle] = words[middle], words[middle - 1]
    return " ".join(words), "два слова переставлены"


# This order is the order of the buttons in the interface.
MUTATIONS = (
    ("Сломать плейсхолдер", break_placeholder, "критическая поломка, почти незаметная для BLEU"),
    ("Убрать отрицание", drop_negation, "смысл на противоположный"),
    ("Потерять предложение", drop_last_sentence, "часть текста исчезает"),
    ("Заменить синонимом", swap_synonym, "смысл сохранён, BLEU падает"),
    ("Переставить слова", shuffle_words, "те же слова, другой порядок"),
)
