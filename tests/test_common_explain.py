"""The shared explanation blocks, and the promise that every page carries them.

The glossary is the one place where a term is defined, so these tests guard the
two things that would quietly rot: an entry with an empty half, and a page that
asks for a term nobody wrote. The short instruction on top of every page is held
to its name: open, first, a few steps and under a hundred words.
"""

from __future__ import annotations

import pathlib

import pytest

from playground_common.explain import (
    DETAILS_TITLE,
    GLOSSARY,
    QUICK_START_TITLE,
    Term,
    how_to_read,
    quick_start,
    reference,
)
from tests.fake_streamlit import FakeStreamlit

ROOT = pathlib.Path(__file__).resolve().parents[1]
APPS = sorted(path for path in ROOT.glob("*_playground/app.py"))


def test_every_app_was_found():
    """Thirteen pages. A new one must not slip past the checks below."""
    assert len(APPS) == 13


@pytest.mark.parametrize("name,term", sorted(GLOSSARY.items()))
def test_every_term_is_filled_in(name: str, term: Term):
    assert term.title.strip(), name
    assert term.meaning.strip(), name
    # The meaning continues the title ("**Accuracy** — ..."), so it starts small
    # and ends with a full stop.
    assert term.meaning[0].islower(), name
    assert term.meaning.rstrip().endswith((".", "»")), name


def test_markdown_carries_the_title_the_formula_and_the_note():
    term = GLOSSARY["accuracy"]
    text = term.as_markdown()
    assert term.title in text
    assert "$$" in text

    plain = Term("Название", "значение без формулы.")
    assert "$$" not in plain.as_markdown()


def test_how_to_read_writes_its_four_sections():
    fake = FakeStreamlit()
    how_to_read(
        fake,
        what="что показано",
        controls="что менять",
        watch="куда смотреть",
        data="откуда данные",
    )

    assert fake.expanders == [DETAILS_TITLE]
    assert fake.open_expanders == [], "the long version is collapsed; quick_start is what stays open"
    (text,) = fake.markdowns
    for heading in ("Что здесь показано", "Что можно менять", "Куда смотреть", "Откуда данные"):
        assert heading in text
    for body in ("что показано", "что менять", "куда смотреть", "откуда данные"):
        assert body in text


def test_how_to_read_drops_the_data_line_when_there_is_none():
    fake = FakeStreamlit()
    how_to_read(fake, what="что", controls="чем", watch="куда")

    (text,) = fake.markdowns
    assert "Откуда данные" not in text


def test_quick_start_is_open_and_numbers_its_steps():
    fake = FakeStreamlit()
    quick_start(fake, purpose="Зачем страница.", steps=("Первое.", "Второе."), notice="одна мысль.")

    assert fake.open_expanders == [QUICK_START_TITLE]
    (text,) = fake.markdowns
    assert text.startswith("Зачем страница.")
    assert "1. Первое.\n2. Второе." in text
    assert text.endswith("**Главное:** одна мысль.")


def test_reference_renders_every_term_it_was_asked_for():
    fake = FakeStreamlit()
    reference(fake, "accuracy", "baseline", extra=(Term("Своё", "определение на одну страницу."),))

    assert fake.expanders == ["Метрики и формулы"]
    (text,) = fake.markdowns
    assert GLOSSARY["accuracy"].title in text
    assert GLOSSARY["baseline"].title in text
    assert "Своё" in text


def test_reference_refuses_a_term_nobody_wrote():
    with pytest.raises(KeyError):
        reference(FakeStreamlit(), "нет-такого-понятия")


QUICK_START_WORDS = 90


def _quick_start_call(path: pathlib.Path):
    import ast

    calls = [
        node
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "quick_start"
    ]
    assert len(calls) == 1, f"{path.parent.name}: expected one quick_start call, found {len(calls)}"
    return {keyword.arg: ast.literal_eval(keyword.value) for keyword in calls[0].keywords}


@pytest.mark.parametrize("path", APPS, ids=lambda path: path.parent.name)
def test_every_quick_start_is_short(path: pathlib.Path):
    """Two to four steps, and fewer words than it takes to lose a reader."""
    texts = _quick_start_call(path)
    assert 2 <= len(texts["steps"]) <= 4
    words = " ".join((texts["purpose"], *texts["steps"], texts["notice"])).replace("**", "").split()
    assert len(words) <= QUICK_START_WORDS, f"{path.parent.name}: {len(words)} words"
    # It follows «Главное:», so it continues a sentence.
    assert texts["notice"][:1].islower()


@pytest.mark.parametrize("path", APPS, ids=lambda path: path.parent.name)
def test_every_page_opens_with_the_two_blocks(path: pathlib.Path):
    """A cold visitor gets the short instruction first, then the long one and a collapsed glossary."""
    source = path.read_text(encoding="utf-8")
    assert "quick_start(" in source
    assert "how_to_read(" in source
    assert source.index("    quick_start(") < source.index("    how_to_read(")
    # nn_playground imports it under another name: `reference` is taken there by
    # the run it compares against.
    assert "reference(" in source or "glossary(" in source


@pytest.mark.parametrize("path", APPS, ids=lambda path: path.parent.name)
def test_every_term_a_page_asks_for_exists(path: pathlib.Path):
    """Catches a typo in a key, which otherwise surfaces only in the browser."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in {"reference", "glossary"}:
            continue
        for argument in node.args[1:]:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                assert argument.value in GLOSSARY, f"{path.parent.name}: {argument.value}"
