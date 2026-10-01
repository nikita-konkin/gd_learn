"""The pages on the site, the topics they belong to, and the sidebar line that links them.

The site carries two courses. «Основы машинного обучения» for the linguistics
master's programme came first: its gradient-descent page sits at the root, since
that URL was published before any other, and its other pages one directory down.
«Основы программирования систем ИИ на языке Python» (Б.1.2.2, profile ИСТ-51)
lives under ``ml-practice/`` — named for what its pages cover, not for the group that
studies them — with a static front page of its own.

Every page also belongs to one topic, and the topics, not the courses, are what
make the site one resource on machine learning: the catalog at ``topics/`` lists
every page by topic, and every sidebar names the page's topic and its
neighbours there, whichever course they come from.

``test_common_links`` checks the registry against ``scripts/build_site.APPS`` in
both directions, so a page cannot be published without a course and a topic, and
nothing can be listed before it is built.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass

# The catalog of every page by topic: a static page, built by scripts/build_site.py.
CATALOG = "topics"
CATALOG_TEXT = "все темы"

# (key, title), in the order a newcomer would meet them.
TOPICS = (
    ("models", "Задачи и модели"),
    ("data", "Данные"),
    ("training", "Обучение и оптимизация"),
    ("evaluation", "Оценка качества"),
    ("text", "Тексты"),
    ("pipelines", "Конвейеры и эксплуатация"),
)


@dataclass(frozen=True)
class Page:
    path: str  # from the site root; "" is the root itself
    text: str  # link text within the course's own sidebar line
    title: str  # the page's name in the catalog and in topic lines
    topic: str  # a key of TOPICS


@dataclass(frozen=True)
class Course:
    title: str
    lead: str  # how the course's sidebar line starts
    home: str  # path of the course's front page
    home_text: str | None  # link text for the front page, if it is not itself a playground
    pages: tuple[Page, ...]  # in course order


ML_BASICS = Course(
    title="Основы машинного обучения",
    lead="Другие playground'ы",
    home="",
    home_text=None,
    pages=(
        Page("", "градиентный спуск", "Градиентный спуск", "training"),
        Page("vec", "векторизация", "Векторизация текста", "text"),
        Page("tm", "память переводов", "Память переводов", "text"),
        Page("mt", "метрики перевода", "Метрики машинного перевода", "evaluation"),
        Page("lm", "языковая модель", "Языковая модель и температура", "text"),
        Page("labels", "данные и разметка", "Данные и разметка", "data"),
        Page("intro", "три задачи лекции", "Три задачи вводной лекции", "models"),
    ),
)

IST51 = Course(
    title="Основы программирования систем ИИ на языке Python",
    lead="Другие площадки",
    home="ml-practice",
    home_text="все площадки",
    pages=(
        Page("ml-practice/data", "конвейер данных", "Конвейер подготовки данных", "data"),
        Page("ml-practice/leak", "утечка данных", "Утечка данных", "evaluation"),
        Page("ml-practice/metric", "метрика и дисбаланс", "Метрика и дисбаланс", "evaluation"),
        Page("ml-practice/fit", "сложность модели", "Сложность модели", "models"),
        Page("ml-practice/nn", "сеть на NumPy", "Нейросеть на NumPy", "training"),
        Page("ml-practice/dag", "граф задач", "Граф задач и отказы", "pipelines"),
    ),
)

COURSES = (ML_BASICS, IST51)

PAGES = tuple(page for course in COURSES for page in course.pages)

# (path, link text) of every published page, in site order.
PLAYGROUNDS = tuple((page.path, page.text) for page in PAGES)


def _href(target: str, current: str) -> str:
    """Relative link from the page at ``current`` to the page at ``target``, both paths from the site root."""
    return posixpath.relpath(target or ".", current or ".") + "/"


def _link(text: str, target: str, current: str) -> str:
    return f"[{text}]({_href(target, current)})"


def course_of(path: str) -> Course:
    return next(course for course in COURSES if any(page.path == path for page in course.pages))


def page_of(path: str) -> Page:
    return next(page for page in PAGES if page.path == path)


def topic_title(key: str) -> str:
    return dict(TOPICS)[key]


def other_playgrounds(current: str) -> str:
    """Markdown for the sidebar: the rest of this course, then this page's topic across the site."""
    course = course_of(current)
    links = [_link(course.home_text, course.home, current)] if course.home_text else []
    links += [_link(page.text, page.path, current) for page in course.pages if page.path != current]

    topic = page_of(current).topic
    same_topic = [page for page in PAGES if page.topic == topic and page.path != current]
    neighbours = [_link(page.title, page.path, current) for page in same_topic]
    catalog = _link(CATALOG_TEXT, CATALOG, current)
    return (
        f"{course.lead}: " + " · ".join(links) + "\n\n"
        f"Тема «{topic_title(topic)}»: " + " · ".join([*neighbours, catalog])
    )
