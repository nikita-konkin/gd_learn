"""String-based measures of translation quality.

``bleu`` and ``chrf`` reproduce lab 3's implementation to the letter: the same
smoothing, the same brevity penalty, the same averaging of the F-measure. On
the teaching corpus they give the documented numbers (BLEU 0.46, chrF 0.64).

``ter`` is not in the lab. It is a third measure, added so the student can see
that the measures disagree not only with a person but with each other.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercased words, punctuation dropped."""
    return TOKEN.findall(str(text).lower())


def ngrams(sequence: Sequence, n: int) -> Counter:
    """Counts of n-grams."""
    return Counter(tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1))


def bleu(hypothesis: str, reference: str, max_n: int = 4, smoothing: float = 1.0) -> float:
    """Simplified sentence-level BLEU, from 0 to 1."""
    hyp, ref = tokenize(hypothesis), tokenize(reference)
    if not hyp or not ref:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams, ref_ngrams = ngrams(hyp, n), ngrams(ref, n)
        total = sum(hyp_ngrams.values())
        if total == 0:  # the hypothesis is shorter than n
            continue
        # smoothing: without it a single zero precision zeroes all of BLEU
        matched = sum(min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items())
        precisions.append((matched + smoothing) / (total + smoothing))

    if not precisions:
        return 0.0

    geometric_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    penalty = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / max(len(hyp), 1))
    return float(geometric_mean * penalty)


def chrf(hypothesis: str, reference: str, max_n: int = 6, beta: float = 2.0) -> float:
    """chrF: an F-measure over character n-grams, from 0 to 1."""
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
    """Levenshtein distance over the elements of a sequence."""
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
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (source_item != target_item),  # substitution
                )
            )
        previous = current
    return previous[-1]


def ter(hypothesis: str, reference: str) -> float:
    """Simplified TER: edits per reference word, with no block shifts.

    Unlike BLEU and chrF, lower is better. Above 1 means the translation needs
    more edits than the reference has words.
    """
    hyp, ref = tokenize(hypothesis), tokenize(reference)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(edit_distance(hyp, ref) / len(ref))
