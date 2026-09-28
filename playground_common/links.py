"""The playgrounds on the site, and the sidebar line that links each to the rest.

The order is the site's: the gradient-descent page first, at the root, then the
labs in course order, then the pages that follow the lectures. ``test_links``
checks it against ``scripts/build_site.APPS``, so a new app cannot be published
without appearing in every other app's sidebar.
"""

from __future__ import annotations

# (slug, link text). The root app has the empty slug.
PLAYGROUNDS = (
    ("", "градиентный спуск"),
    ("vec", "векторизация"),
    ("tm", "память переводов"),
    ("mt", "метрики перевода"),
    ("lm", "языковая модель"),
    ("labels", "данные и разметка"),
    ("intro", "три задачи лекции"),
)


def _href(target: str, current: str) -> str:
    """Relative link from the page at ``current`` to the page at ``target``."""
    prefix = "" if current == "" else "../"
    return prefix + (f"{target}/" if target else "")


def other_playgrounds(current: str) -> str:
    """Markdown for the sidebar: every playground except the current one."""
    links = [f"[{text}]({_href(slug, current)})" for slug, text in PLAYGROUNDS if slug != current]
    return "Другие playground'ы: " + " · ".join(links)
