"""Looking a segment up in the translation memory, by each measure in turn.

This is a nearest-neighbour search. The whole question is what counts as
distance between two texts, and the answers disagree — which is the point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tm_playground.measures import (
    CHAR_NGRAMS,
    EMBEDDINGS,
    LEVENSHTEIN,
    LSA,
    build_char_vectorizer,
    build_lsa,
    cosine,
    match_percent,
)


@dataclass(frozen=True)
class Hit:
    """One suggestion: where it sits in the memory and how close it scored."""

    rank: int
    index: int
    segment: str
    source: str
    content_type: str
    score: float


def rank_descending(scores: np.ndarray, top_n: int) -> list[int]:
    """Row numbers of the ``top_n`` best scores, ties going to the earlier row.

    A stable sort of the negated scores, rather than reversing an ascending
    one, which would hand ties to the *later* row and quietly disagree with the
    lab on every exact draw.
    """
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    return [int(index) for index in order[:top_n]]


class SearchIndex:
    """The translation memory with every representation of it fitted once.

    Fitting the vectorizers is the expensive part and it does not depend on the
    query, so it happens here and the app caches the whole object.
    """

    def __init__(
        self,
        corpus: pd.DataFrame,
        queries: pd.DataFrame,
        segment_vectors: np.ndarray,
        query_vectors: np.ndarray,
    ):
        self.corpus = corpus.reset_index(drop=True)
        self.segments = self.corpus["ru_ref"].astype(str).tolist()
        self.queries = queries.reset_index(drop=True)
        self.segment_vectors = segment_vectors
        self.query_vectors = query_vectors

        self._char = build_char_vectorizer()
        self._char_matrix = self._char.fit_transform(self.segments)
        self._lsa = build_lsa()
        self._lsa_matrix = self._lsa.fit_transform(self.segments)

        self._query_rows = {
            str(text).strip(): row for row, text in enumerate(self.queries["text"])
        }

    def representations(self) -> dict[str, object]:
        """Every vector view of the memory, keyed by the measure that built it.

        Clustering and the corpus map work on these directly: they ask about
        the memory as a whole rather than about one query against it.
        """
        return {
            CHAR_NGRAMS: self._char_matrix,
            LSA: self._lsa_matrix,
            EMBEDDINGS: self.segment_vectors,
        }

    # -- what each measure can and cannot answer ---------------------------

    def precomputed_row(self, query: str) -> int | None:
        """Which lab query this is, or None if the student typed their own."""
        return self._query_rows.get(str(query).strip())

    def supports(self, query: str, measure: str) -> bool:
        """Whether this measure can score this query at all.

        Only the pretrained embeddings ever say no, and only for text that is
        not one of the lab's queries: turning text into one of those vectors
        needs the model itself, which is half a gigabyte and is not in the
        browser. That is not a gap in the playground — it is what the lab means
        by calling the embedding step a stage of the pipeline rather than a
        line in a notebook.
        """
        if measure != EMBEDDINGS:
            return True
        return self.precomputed_row(query) is not None

    # -- scoring -----------------------------------------------------------

    def scores(self, query: str, measure: str) -> np.ndarray | None:
        """Score of ``query`` against every segment, or None if unsupported."""
        if measure == LEVENSHTEIN:
            return np.array([match_percent(query, segment) for segment in self.segments])
        if measure == CHAR_NGRAMS:
            return cosine(self._char.transform([query]), self._char_matrix)
        if measure == LSA:
            return cosine(self._lsa.transform([query]), self._lsa_matrix)
        if measure == EMBEDDINGS:
            row = self.precomputed_row(query)
            if row is None:
                return None
            return cosine(self.query_vectors[row : row + 1], self.segment_vectors)
        raise ValueError(f"unknown measure: {measure}")

    def search(self, query: str, measure: str, top_n: int = 3) -> list[Hit]:
        """The ``top_n`` closest memory segments by one measure."""
        scores = self.scores(query, measure)
        if scores is None:
            return []
        return [
            Hit(
                rank=position + 1,
                index=index,
                segment=self.segments[index],
                source=str(self.corpus["en"].iloc[index]),
                content_type=str(self.corpus["type"].iloc[index]),
                score=float(scores[index]),
            )
            for position, index in enumerate(rank_descending(scores, top_n))
        ]

    def best(self, query: str, measure: str) -> Hit | None:
        """The single closest segment, or None if the measure cannot score this."""
        hits = self.search(query, measure, top_n=1)
        return hits[0] if hits else None

    def rank_of(self, query: str, measure: str, index: int) -> int | None:
        """Where a particular segment lands in this measure's ranking, 1-based."""
        scores = self.scores(query, measure)
        if scores is None:
            return None
        return rank_descending(scores, len(self.segments)).index(index) + 1


def comparison_table(index: SearchIndex) -> pd.DataFrame:
    """Best answer from each measure for every query in the lab, side by side.

    The central exercise of the lab is reading the rows where the first two
    columns disagree, so ``agree`` compares exactly those two: the
    character-counting measure against the order-blind one.
    """
    rows = []
    for _, query in index.queries.iterrows():
        text = str(query["text"])
        by_measure = {measure: index.best(text, measure) for measure in (LEVENSHTEIN, CHAR_NGRAMS, EMBEDDINGS)}
        rows.append(
            {
                "query": text,
                "note": str(query["note"]),
                "levenshtein_hit": by_measure[LEVENSHTEIN].segment,
                "levenshtein_score": round(by_measure[LEVENSHTEIN].score, 1),
                "char_ngrams_hit": by_measure[CHAR_NGRAMS].segment,
                "char_ngrams_score": round(by_measure[CHAR_NGRAMS].score, 3),
                "embeddings_hit": by_measure[EMBEDDINGS].segment if by_measure[EMBEDDINGS] else "",
                "embeddings_score": round(by_measure[EMBEDDINGS].score, 3) if by_measure[EMBEDDINGS] else float("nan"),
                "agree": by_measure[LEVENSHTEIN].segment == by_measure[CHAR_NGRAMS].segment,
            }
        )
    return pd.DataFrame(rows)


def divergences(table: pd.DataFrame) -> pd.DataFrame:
    """Only the rows where the two surface measures picked different segments."""
    return table[~table["agree"]].reset_index(drop=True)


def threshold_sweep(index: SearchIndex, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """How many queries get a suggestion at all, as the cut-off moves.

    ``agreeing`` counts the offered suggestions the order-blind measure would
    also have picked. It is not a measure of correctness — nothing here knows
    which suggestion is genuinely useful, that judgement is the translator's —
    but a suggestion both measures arrive at independently is the safer bet,
    and watching the two lines separate is the honest version of the question.
    """
    if thresholds is None:
        thresholds = np.arange(50, 100, 5, dtype=float)

    best_percent = []
    agreed = []
    for text in index.queries["text"].astype(str):
        best = index.best(text, LEVENSHTEIN)
        best_percent.append(best.score)
        agreed.append(best.segment == index.best(text, CHAR_NGRAMS).segment)

    best_percent = np.array(best_percent)
    agreed = np.array(agreed)
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "offered": [int((best_percent >= t).sum()) for t in thresholds],
            "agreeing": [int((agreed & (best_percent >= t)).sum()) for t in thresholds],
            "total": len(best_percent),
        }
    )


def suggestions(index: SearchIndex, threshold: float) -> pd.DataFrame:
    """Every query with its best match, and whether the cut-off lets it through."""
    rows = []
    for _, query in index.queries.iterrows():
        text = str(query["text"])
        best = index.best(text, LEVENSHTEIN)
        rows.append(
            {
                "query": text,
                "note": str(query["note"]),
                "suggestion": best.segment,
                "percent": round(best.score, 1),
                "offered": best.score >= threshold,
            }
        )
    return pd.DataFrame(rows).sort_values("percent", ascending=False).reset_index(drop=True)
