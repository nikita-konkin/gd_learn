"""Collect the data files the data-and-annotation playground ships.

Three sources, three kinds of provenance:

* the corpus comes from this repository and is scored with the BLEU and chrF
  that ``mt_playground`` already checks against the lab;
* the ten annotated segments and the second annotator are read out of the
  lab 3 notebook itself — the literal lists in its cells, parsed rather than
  copied by hand, so the playground cannot quietly disagree with the lab;
* the three remedies of lab 1, section 13, are recomputed with the lab's own
  code. One of them needs the 20 Newsgroups corpus (about 14 MB, fetched once
  and cached by scikit-learn), which is exactly why it is computed here and not
  in the browser.

    python -m scripts.prepare_labels_data
    python -m scripts.prepare_labels_data --course /path/to/course/folder

Run as a module from the repository root: it scores the corpus with
``mt_playground``'s metrics, and that package has to be importable.

The error key for lab 3 is never read. The ten segments used here are the ones
the lab publishes as its worked example; the key covers the segments students
annotate themselves and stays with the teacher.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from mt_playground.metrics import bleu, chrf

ROOT = Path(__file__).resolve().parents[1]
CORPUS_SOURCE = ROOT / "mt_playground" / "data" / "loc_corpus.csv"
COURSE = Path("/Users/w/MySIlver/дисциплины/Б.1.1.21 Машинное обучение и анализ данных/Основы МО МЛГ")
TARGET = ROOT / "labels_playground" / "data"

LAB3_PREFIX = "Л_р_№3_"

# Lab 1, section 13: the same grid, word tokenizer and newsgroup categories.
GRID = {
    "tfidfvectorizer__ngram_range": [(2, 4), (2, 5), (3, 5), (1, 4)],
    "tfidfvectorizer__sublinear_tf": [False, True],
    "tfidfvectorizer__min_df": [1, 2],
    "logisticregression__C": [1.0, 5.0, 20.0],
}
NEWSGROUPS = ["rec.autos", "sci.med", "talk.politics.guns", "comp.graphics"]
NEWSGROUP_SIZES = [160, 320, 640, 1280, 1800]
TOKEN = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def find_notebook(lab_dir: Path, prefix: str) -> Path:
    """The notebook whose name starts with ``prefix``.

    Listed from disk rather than spelled out: one of the course's notebook names
    is stored decomposed (NFD), so a literal can miss it on another system.
    """
    wanted = unicodedata.normalize("NFC", prefix)
    matches = [
        lab_dir / name
        for name in os.listdir(lab_dir)
        if unicodedata.normalize("NFC", name).startswith(wanted) and name.endswith(".ipynb")
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected one notebook starting with {prefix!r} in {lab_dir}, found {len(matches)}")
    return matches[0]


def _assigned_literals(source: str) -> dict[str, ast.expr]:
    """``name = <expr>`` at the top level of a cell, by name."""
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            found[node.targets[0].id] = node.value
    return found


def read_lab3_annotations(notebook: Path) -> pd.DataFrame:
    """The lab's worked example: ten segments, one annotation each, plus the
    second annotator's categories from the agreement cell."""
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    assignments: dict[str, ast.expr] = {}
    for cell in cells:
        if cell["cell_type"] == "code":
            assignments.update(_assigned_literals("".join(cell["source"])))

    # example = pd.DataFrame([{...}, ...]) — the list is the first argument.
    example_call = assignments["example"]
    if not (isinstance(example_call, ast.Call) and example_call.args):
        raise SystemExit("the lab's `example` is no longer a DataFrame built from a literal list")
    rows = ast.literal_eval(example_call.args[0])
    first = ast.literal_eval(assignments["annotator_1"])
    second = ast.literal_eval(assignments["annotator_2"])

    examples = pd.DataFrame(rows).rename(
        columns={"категория": "category", "серьёзность": "severity", "комментарий": "comment"}
    )
    # The agreement cell restates the example's categories as annotator 1. If
    # the two ever disagree, the lab has changed and this file must be rebuilt
    # by a person, not averaged over.
    if list(examples["category"]) != list(first):
        raise SystemExit("annotator_1 in the lab no longer matches the worked example")
    if len(second) != len(examples):
        raise SystemExit("annotator_2 has a different length from the worked example")
    examples["category_second"] = second
    return examples[["id", "category", "severity", "comment", "category_second"]]


def score_corpus(corpus_source: Path) -> pd.DataFrame:
    corpus = pd.read_csv(corpus_source)[["id", "type", "en", "ru_ref", "ru_mt"]]
    corpus["bleu"] = [round(bleu(h, r), 6) for h, r in zip(corpus["ru_mt"], corpus["ru_ref"], strict=True)]
    corpus["chrf"] = [round(chrf(h, r), 6) for h, r in zip(corpus["ru_mt"], corpus["ru_ref"], strict=True)]
    return corpus


def compute_remedies(corpus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lab 1, section 13, with the lab's own settings."""
    from nltk.stem.snowball import SnowballStemmer
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
    from sklearn.pipeline import make_pipeline, make_union

    warnings.filterwarnings("ignore", category=UserWarning)
    texts, labels = corpus["ru_ref"], corpus["type"]
    stemmer = SnowballStemmer("russian")

    def tokenize_and_stem(text: str) -> list[str]:
        return [stemmer.stem(word) for word in TOKEN.findall(text.lower())]

    def best_vectorizer():
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))

    base = cross_val_score(
        make_pipeline(best_vectorizer(), LogisticRegression(max_iter=2000)), texts, labels, cv=5, scoring="accuracy"
    ).mean()

    grid = GridSearchCV(
        make_pipeline(TfidfVectorizer(analyzer="char_wb"), LogisticRegression(max_iter=3000)),
        param_grid=GRID,
        cv=5,
        scoring="accuracy",
    ).fit(texts, labels)

    union = make_union(
        TfidfVectorizer(tokenizer=tokenize_and_stem, token_pattern=None),
        best_vectorizer(),
    )
    united = cross_val_score(
        make_pipeline(union, LogisticRegression(max_iter=2000)), texts, labels, cv=5, scoring="accuracy"
    ).mean()

    news = fetch_20newsgroups(
        subset="train", categories=NEWSGROUPS, remove=("headers", "footers", "quotes"), random_state=0
    )
    sizes, _, scores = learning_curve(
        make_pipeline(best_vectorizer(), LogisticRegression(max_iter=2000)),
        news.data,
        news.target,
        cv=5,
        train_sizes=NEWSGROUP_SIZES,
        scoring="accuracy",
        n_jobs=-1,
    )
    curve = pd.DataFrame({"train_size": sizes.astype(int), "accuracy": np.round(scores.mean(axis=1), 3)})

    remedies = pd.DataFrame(
        [
            {"remedy": "hyperparameters", "before": base, "after": grid.best_score_, "combinations": 48},
            {"remedy": "feature_union", "before": base, "after": united, "combinations": 1},
            {
                "remedy": "more_data",
                "before": float(curve["accuracy"].iloc[0]),
                "after": float(curve["accuracy"].iloc[-1]),
                "combinations": 1,
            },
        ]
    )
    remedies[["before", "after"]] = remedies[["before", "after"]].round(3)
    remedies["gain"] = (remedies["after"] - remedies["before"]).round(3)
    return remedies, curve


def build(corpus_source: Path = CORPUS_SOURCE, course: Path = COURSE, target: Path = TARGET) -> dict[str, int]:
    """Write the four data files and return their row counts."""
    target.mkdir(parents=True, exist_ok=True)

    corpus = score_corpus(corpus_source)
    corpus.to_csv(target / "corpus.csv", index=False)

    examples = read_lab3_annotations(find_notebook(course / "Л.р", LAB3_PREFIX))
    missing = set(examples["id"]) - set(corpus["id"])
    if missing:
        raise SystemExit(f"annotated segments missing from the corpus: {sorted(missing)}")
    examples.to_csv(target / "mqm_examples.csv", index=False)

    remedies, curve = compute_remedies(corpus)
    remedies.to_csv(target / "remedies.csv", index=False)
    curve.to_csv(target / "newsgroups_curve.csv", index=False)

    return {"corpus": len(corpus), "mqm_examples": len(examples), "remedies": len(remedies), "newsgroups": len(curve)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--course", type=Path, default=COURSE, help="the course folder, containing Л.р/")
    args = parser.parse_args()

    counts = build(course=args.course)
    for name, rows in counts.items():
        print(f"  {name:<14} {rows} rows")
    print(pd.read_csv(TARGET / "remedies.csv").to_string(index=False))


if __name__ == "__main__":
    main()
