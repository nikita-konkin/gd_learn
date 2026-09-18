"""Как текст превращается в числа.

Ровно те ручки, которые в лабораторной работе № 1 перебираются вручную по
списку конфигураций. Значения по умолчанию совпадают с `TfidfVectorizer()`
без аргументов, поэтому «всё по умолчанию» в интерфейсе даёт ту же точность,
что первая строка таблицы в работе.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

TOKEN = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

ANALYZERS = ("слова", "символы внутри слов", "символы подряд")
ANALYZER_CODES = {
    "слова": "word",
    "символы внутри слов": "char_wb",
    "символы подряд": "char",
}

_stemmer = SnowballStemmer("russian")


def tokenize_words(text: str) -> list[str]:
    """Слова из букв и цифр, приведённые к нижнему регистру."""
    return TOKEN.findall(str(text).lower())


def tokenize_and_stem(text: str) -> list[str]:
    """То же, но каждое слово усечено до основы.

    «Сохранить» и «сохранены» становятся одним признаком — для русского это
    даёт больше, чем любая замена классификатора.
    """
    return [_stemmer.stem(word) for word in tokenize_words(text)]


@dataclass(frozen=True)
class FeatureSettings:
    """Настройки векторизатора — то, что студент крутит в сайдбаре."""

    analyzer: str = "слова"
    ngram_min: int = 1
    ngram_max: int = 1
    lowercase: bool = True
    stemming: bool = False
    min_df: int = 1
    use_idf: bool = True
    sublinear_tf: bool = False

    def describe(self) -> str:
        parts = [self.analyzer, f"n-граммы {self.ngram_min}-{self.ngram_max}"]
        if self.stemming and self.analyzer == "слова":
            parts.append("стемминг")
        if not self.lowercase:
            parts.append("регистр сохранён")
        if self.min_df > 1:
            parts.append(f"min_df={self.min_df}")
        if not self.use_idf:
            parts.append("счётчики")
        if self.sublinear_tf:
            parts.append("sublinear_tf")
        return ", ".join(parts)


def build_vectorizer(settings: FeatureSettings) -> TfidfVectorizer:
    """Собрать `TfidfVectorizer` из настроек.

    Стемминг подключается только для словарного анализатора: усекать основы у
    символьных n-грамм бессмысленно, и `scikit-learn` в этом случае токенизатор
    попросту игнорирует — лучше не делать вид, что ручка работает.
    """
    analyzer = ANALYZER_CODES[settings.analyzer]
    parameters = {
        "analyzer": analyzer,
        "ngram_range": (settings.ngram_min, max(settings.ngram_min, settings.ngram_max)),
        "lowercase": settings.lowercase,
        "min_df": settings.min_df,
        "use_idf": settings.use_idf,
        "sublinear_tf": settings.sublinear_tf,
    }
    if analyzer == "word" and settings.stemming:
        parameters["tokenizer"] = tokenize_and_stem
        parameters["token_pattern"] = None
    return TfidfVectorizer(**parameters)
