"""The sidebar links between the playgrounds."""

import re
from pathlib import Path

import pytest

from playground_common.links import PLAYGROUNDS, other_playgrounds
from scripts.build_site import APPS

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[([^\]]+)\]\(([^)]*)\)")


def test_every_published_app_is_listed_once():
    assert [slug for slug, _ in PLAYGROUNDS] == list(dict.fromkeys(slug for slug, _ in PLAYGROUNDS))
    assert {slug for slug, _ in PLAYGROUNDS} == {app.slug for app in APPS}


@pytest.mark.parametrize("current", [slug for slug, _ in PLAYGROUNDS], ids=lambda slug: slug or "root")
def test_each_page_links_to_every_other_page_and_not_to_itself(current):
    links = LINK.findall(other_playgrounds(current))

    assert [text for text, _ in links] == [text for slug, text in PLAYGROUNDS if slug != current]
    assert len(links) == len(PLAYGROUNDS) - 1


def test_links_are_relative_to_the_page():
    """The root page links down, the others link up and across."""
    from_root = dict(LINK.findall(other_playgrounds("")))
    from_vec = dict(LINK.findall(other_playgrounds("vec")))

    assert from_root["векторизация"] == "vec/"
    assert from_vec["градиентный спуск"] == "../"
    assert from_vec["память переводов"] == "../tm/"
    assert all(href for href in from_root.values())


@pytest.mark.parametrize("app", APPS, ids=lambda app: app.slug or "root")
def test_every_app_puts_the_line_in_its_sidebar(app):
    source = (ROOT / app.package / "app.py").read_text(encoding="utf-8")
    assert f'other_playgrounds("{app.slug}")' in source
