"""The registry of pages, courses and topics, and the sidebar line built from it."""

import re
from pathlib import Path

import pytest

from playground_common.links import (
    CATALOG,
    CATALOG_TEXT,
    COURSES,
    IST51,
    ML_BASICS,
    PAGES,
    PLAYGROUNDS,
    TOPICS,
    other_playgrounds,
    page_of,
)
from scripts.build_site import APPS

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[([^\]]+)\]\(([^)]*)\)")
PATHS = [path for path, _ in PLAYGROUNDS]


def _course_line(current):
    return other_playgrounds(current).split("\n\n")[0]


def _topic_line(current):
    return other_playgrounds(current).split("\n\n")[1]


def test_every_published_app_is_listed_once():
    assert PATHS == list(dict.fromkeys(PATHS))
    assert set(PATHS) == {app.slug for app in APPS}


def test_every_page_has_a_known_topic_and_every_topic_a_page():
    keys = [key for key, _ in TOPICS]

    assert len(keys) == len(set(keys))
    assert {page.topic for page in PAGES} == set(keys)


def test_every_page_belongs_to_exactly_one_course():
    for path in PATHS:
        assert sum(any(page.path == path for page in course.pages) for course in COURSES) == 1, path


def test_the_ist51_course_lives_under_its_own_directory():
    assert all(page.path.startswith(f"{IST51.home}/") for page in IST51.pages)
    assert not any(page.path.startswith(f"{IST51.home}/") for page in ML_BASICS.pages)


def test_catalog_titles_are_the_apps_titles():
    # The root app keeps its English <title>, published long before the catalog.
    for app in APPS:
        if app.slug:
            assert page_of(app.slug).title == app.title, app.slug


@pytest.mark.parametrize("current", PATHS, ids=lambda path: path or "root")
def test_the_course_line_links_to_every_other_page_of_the_course_and_not_to_itself(current):
    course = next(course for course in COURSES if any(page.path == current for page in course.pages))
    texts = [text for text, _ in LINK.findall(_course_line(current))]

    expected = [course.home_text] if course.home_text else []
    expected += [page.text for page in course.pages if page.path != current]
    assert texts == expected


@pytest.mark.parametrize("current", PATHS, ids=lambda path: path or "root")
def test_the_topic_line_names_the_topic_and_links_every_neighbour_on_it(current):
    page = page_of(current)
    line = _topic_line(current)
    titles = [text for text, _ in LINK.findall(line)]

    assert f"«{dict(TOPICS)[page.topic]}»" in line
    assert titles == [other.title for other in PAGES if other.topic == page.topic and other.path != current] + [
        CATALOG_TEXT
    ]


def test_links_are_relative_to_the_page():
    """The root page links down, the others up and across, at whatever depth they sit."""
    from_root = dict(LINK.findall(other_playgrounds("")))
    from_vec = dict(LINK.findall(other_playgrounds("vec")))
    from_leak = dict(LINK.findall(other_playgrounds("ml-practice/leak")))

    assert from_root["векторизация"] == "vec/"
    assert from_root[CATALOG_TEXT] == f"{CATALOG}/"
    assert from_vec["градиентный спуск"] == "../"
    assert from_vec["память переводов"] == "../tm/"
    assert from_leak["все площадки"] == "../"
    assert from_leak["сложность модели"] == "../fit/"
    assert from_leak["Метрики машинного перевода"] == "../../mt/"
    assert from_leak[CATALOG_TEXT] == f"../../{CATALOG}/"


def test_no_link_is_absolute():
    for current in PATHS:
        for _, href in LINK.findall(other_playgrounds(current)):
            assert not href.startswith(("/", "http")), (current, href)


@pytest.mark.parametrize("app", APPS, ids=lambda app: app.slug or "root")
def test_every_app_puts_the_line_in_its_sidebar(app):
    source = (ROOT / app.package / "app.py").read_text(encoding="utf-8")
    assert f'other_playgrounds("{app.slug}")' in source
