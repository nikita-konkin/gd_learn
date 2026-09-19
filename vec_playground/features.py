"""How text turns into numbers.

Exactly the knobs that lab 1 works through by hand, one configuration at a
time. The defaults match ``TfidfVectorizer()`` with no arguments, so leaving
everything alone in the interface reproduces the first row of the lab's table.
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
    """Words of letters and digits, lowercased."""
    return TOKEN.findall(str(text).lower())


def tokenize_and_stem(text: str) -> list[str]:
    """The same, with each word cut back to its stem.

    "Сохранить" and "сохранены" collapse into one feature, which for Russian
    buys more than any change of classifier.
    """
    return [_stemmer.stem(word) for word in tokenize_words(text)]


@dataclass(frozen=True)
class FeatureSettings:
    """The vectoriser settings — what the student turns in the sidebar."""

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
    """Assemble a ``TfidfVectorizer`` from the settings.

    Stemming is wired up only for the word analyzer. Stemming character
    n-grams is meaningless, and ``scikit-learn`` silently ignores the tokenizer
    in that case — better not to pretend the knob does something.
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
