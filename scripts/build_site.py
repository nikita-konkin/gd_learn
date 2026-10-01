"""Build the static GitHub Pages bundle.

Each Streamlit app is served through stlite, which runs CPython (Pyodide) in the
browser, so the published site is plain static files with no server component.

The Python sources are copied verbatim next to each ``index.html`` and referenced
by URL from the generated stlite manifest, so the deployed apps always match the
repository sources.

The gradient-descent app stays at the site root: it was published there first and
that URL is in circulation. Further apps of its course get a subdirectory each.
The ИСТ-51 course lives under ``ml-practice/``. Two pages carry no Python at all, so
they load at once: the ИСТ-51 front page and the catalog of every app by topic
at ``topics/``. Courses and topics themselves are registered in
``playground_common/links.py``; this file only builds what is registered there.
"""

from __future__ import annotations

import argparse
import html
import json
import posixpath
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playground_common.links import (  # noqa: E402  — the registry lives with the apps that ship it
    CATALOG,
    COURSES,
    IST51,
    TOPICS,
    course_of,
    page_of,
)
from playground_common.wording import plural  # noqa: E402

TEMPLATE = ROOT / "web" / "index.template.html"
# The ИСТ-51 pages are in Russian throughout, in the lecture slides' colours,
# and their loading screen links back to the course's front page.
IST51_TEMPLATE = ROOT / "web" / "ist51" / "index.template.html"
# Both static pages: the ИСТ-51 front page and the catalog by topic.
CATALOG_TEMPLATE = ROOT / "web" / "catalog.template.html"

# Pinned so a CDN release cannot silently change the deployed runtime.
STLITE_VERSION = "1.9.1"

# Helpers every app imports: the pyarrow stub patch, Russian numerals, the
# links between the apps. Shipped with each app, since each is served alone.
SHARED_PACKAGE = "playground_common"


@dataclass(frozen=True)
class App:
    """One stlite app on the published site."""

    slug: str  # path from the site root; "" means the site root
    entrypoint: str
    package: str
    title: str
    # numpy and pandas ship as prebuilt Pyodide wheels; the rest come from PyPI.
    requirements: tuple[str, ...] = ("numpy", "pandas", "plotly>=5.20,<8")
    data_globs: tuple[str, ...] = field(default=())
    template: Path = TEMPLATE
    # For the cards on the static pages: where in its course the app belongs,
    # and one sentence on what it shows.
    source: str = ""
    blurb: str = ""

    @property
    def output_subdir(self) -> str:
        return self.slug


IST51_SKLEARN = ("numpy", "pandas", "plotly>=5.20,<8", "scikit-learn>=1.5")

APPS = (
    App(
        slug="",
        entrypoint="gradient_descent_playground_v3.py",
        package="gd_playground",
        title="Gradient Descent Playground",
        blurb=(
            "Градиентный спуск на синтетических данных: распределения, модели, Batch GD, SGD и мини-пакеты, "
            "траектория на поверхности потерь."
        ),
    ),
    App(
        slug="mt",
        entrypoint="mt_metrics_playground.py",
        package="mt_playground",
        title="Метрики машинного перевода",
        data_globs=("data/*.csv",),
        source="ЛР № 3",
        blurb="Правьте перевод и смотрите, что BLEU, chrF и TER видят, а что пропускают.",
    ),
    App(
        slug="lm",
        entrypoint="lm_text_playground.py",
        package="lm_playground",
        title="Языковая модель и температура",
        data_globs=("data/*.csv",),
        source="ЛР № 4",
        blurb=(
            "Символьная n-граммная модель обучается прямо в браузере: распределение следующего токена, "
            "температура и граница между «складно» и «выучено наизусть»."
        ),
    ),
    App(
        slug="vec",
        entrypoint="text_features_playground.py",
        package="vec_playground",
        title="Векторизация текста",
        # scikit-learn and nltk both ship as prebuilt wheels in the Pyodide build.
        requirements=("numpy", "pandas", "plotly>=5.20,<8", "scikit-learn>=1.5", "nltk>=3.9"),
        data_globs=("data/*.csv",),
        source="ЛР № 1",
        blurb="Классификация 160 сегментов локализации: способ построения признаков — ручкой, точность — сразу.",
    ),
    App(
        slug="tm",
        entrypoint="tm_search_playground.py",
        package="tm_playground",
        title="Память переводов",
        requirements=("numpy", "pandas", "plotly>=5.20,<8", "scikit-learn>=1.5"),
        # The .npy files are the pretrained embeddings: they cannot be recomputed
        # in the browser, so they travel with the app as data.
        data_globs=("data/*.csv", "data/*.npy"),
        source="ЛР № 2",
        blurb="Нечёткий поиск по памяти переводов: четыре меры близости на одном запросе и где они расходятся.",
    ),
    App(
        slug="labels",
        entrypoint="data_labels_playground.py",
        package="labels_playground",
        title="Данные и разметка",
        # No nltk: the one remedy that stems words is computed offline, like the
        # 20 Newsgroups curve, and shipped as numbers.
        requirements=("numpy", "pandas", "plotly>=5.20,<8", "scikit-learn>=1.5"),
        data_globs=("data/*.csv",),
        source="лекция «Текст как данные»",
        blurb=(
            "Что даёт больше — другие признаки, другая модель или ещё разметка; можно ли принимать перевод "
            "по BLEU; насколько можно доверять самой разметке."
        ),
    ),
    App(
        slug="intro",
        entrypoint="intro_tasks_playground.py",
        package="intro_playground",
        title="Три задачи вводной лекции",
        # scikit-learn for the decision tree only; regression is one np.polyfit.
        requirements=("numpy", "pandas", "plotly>=5.20,<8", "scikit-learn>=1.5"),
        data_globs=("data/*.csv",),
        source="вводная лекция",
        blurb="Дерево решений по признакам, которые придумал бы лингвист, длина перевода и сочетания для глоссария.",
    ),
    App(
        slug="ml-practice/data",
        entrypoint="data_preparation_playground.py",
        package="data_playground",
        title="Конвейер подготовки данных",
        template=IST51_TEMPLATE,
        source="модуль 1, лекция 2",
        blurb=(
            "Порядок шагов очистки как параметр: статистика, посчитанная до удаления выбросов, "
            "вставляет в таблицу невозможные значения. И векторизация, измеренная в вашем браузере."
        ),
    ),
    App(
        slug="ml-practice/leak",
        entrypoint="data_leakage_playground.py",
        package="leak_playground",
        title="Утечка данных",
        # scikit-learn for SelectKBest, LogisticRegression and cross-validation.
        requirements=IST51_SKLEARN,
        template=IST51_TEMPLATE,
        source="модуль 2, лекция 6",
        blurb=(
            "Отбор признаков до разбиения выборок находит сигнал в чистом шуме. Два конвейера рядом: с утечкой и без."
        ),
    ),
    App(
        slug="ml-practice/metric",
        entrypoint="class_imbalance_playground.py",
        package="metric_playground",
        title="Метрика и дисбаланс",
        # No scikit-learn: the scores are prepared offline and the metrics are NumPy.
        data_globs=("data/*.csv",),
        template=IST51_TEMPLATE,
        source="модуль 2, лекции 4 и 6",
        blurb=(
            "Модель, не нашедшая ни одного отказа, верна в 98 случаях из 100. Порог, цена пропуска "
            "и три способа выправить перекос классов."
        ),
    ),
    App(
        slug="ml-practice/fit",
        entrypoint="model_complexity_playground.py",
        package="fit_playground",
        title="Сложность модели",
        # scikit-learn for every model on the page.
        requirements=IST51_SKLEARN,
        data_globs=("data/*.csv",),
        template=IST51_TEMPLATE,
        source="модуль 2, лекции 4 и 5",
        blurb=(
            "Многочлен, соседи, дерево, штраф Ridge и Lasso: гибкость модели — ручкой, "
            "цена ошибки — на данных, которых модель не видела."
        ),
    ),
    App(
        slug="ml-practice/nn",
        entrypoint="numpy_network_playground.py",
        package="nn_playground",
        title="Нейросеть на NumPy",
        # The two moons are shipped as a CSV, so NumPy is all the network needs.
        data_globs=("data/*.csv",),
        template=IST51_TEMPLATE,
        source="модуль 3, лекция 7",
        blurb=(
            "Сеть лекции 7 с одной строкой обратного прохода, которую можно сломать. Обучение ошибку "
            "не замечает — численная проверка градиента замечает."
        ),
    ),
    App(
        slug="ml-practice/dag",
        entrypoint="task_graph_playground.py",
        package="dag_playground",
        title="Граф задач и отказы",
        template=IST51_TEMPLATE,
        source="модуль 4, лекция 12",
        blurb=(
            "Планировщик лекции 12: повторы, пропуск задач после отказа — и что будет, если убрать "
            "одно ребро, охраняющее публикацию."
        ),
    ),
)

# Short course names for the cards in the catalog, keyed by the course's front page.
COURSE_TAGS = {"": "Основы машинного обучения", IST51.home: "ИСТ-51"}

FOOTER = "Код площадок распространяется по лицензии MIT."
IST51_FOOTER = (
    "Б.1.2.2 «Основы программирования систем искусственного интеллекта на языке Python», профиль ИСТ-51. "
    "Поволжский государственный технологический университет, кафедра радиотехники и связи. " + FOOTER
)


def collect_sources(app: App) -> list[str]:
    """Repo-relative POSIX paths of every file the app needs at runtime."""
    paths = [app.entrypoint]
    for package in (app.package, SHARED_PACKAGE):
        paths += sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / package).glob("*.py"))
    package_dir = ROOT / app.package
    for pattern in app.data_globs:
        paths += sorted(path.relative_to(ROOT).as_posix() for path in package_dir.glob(pattern))
    return paths


def site_manifest(app: App) -> dict[str, dict[str, str]]:
    """The stlite ``files`` mapping: virtual path -> URL on the site."""
    return {relative: {"url": f"./{relative}"} for relative in collect_sources(app)}


def _substitute(template: Path, replacements: dict[str, str]) -> str:
    text = template.read_text(encoding="utf-8")
    for placeholder, value in replacements.items():
        if placeholder not in text:
            raise SystemExit(f"placeholder {placeholder} missing from {template}")
        text = text.replace(placeholder, value)
    return text


def render_index(app: App) -> str:
    """``index.html`` for one app, with every build-time placeholder substituted."""
    return _substitute(
        app.template,
        {
            "__STLITE_VERSION__": STLITE_VERSION,
            "__TITLE__": app.title,
            "__ENTRYPOINT__": app.entrypoint,
            "__REQUIREMENTS__": json.dumps(list(app.requirements)),
            "__FILES__": json.dumps(site_manifest(app), indent=2),
        },
    )


def _relative(target: str, page: str) -> str:
    """Link from a static page at ``page`` to the app at ``target``, both paths from the site root."""
    return posixpath.relpath(target or ".", page) + "/"


def _card(app: App, page: str, tag: str) -> str:
    return (
        "<li>"
        f'<span class="tag">{html.escape(tag)}</span><br />'
        f'<a href="{_relative(app.slug, page)}">{html.escape(page_of(app.slug).title)}</a>'
        f"<p>{html.escape(app.blurb)}</p>"
        "</li>"
    )


def _sections(apps: list[App], page: str, tag) -> str:
    """One heading and one list of cards per topic, in the registry's order; empty topics are left out."""
    separator = "\n        "
    sections = []
    for key, title in TOPICS:
        cards = [_card(app, page, tag(app)) for app in apps if page_of(app.slug).topic == key]
        if cards:
            sections.append(
                f'<h2 id="{key}">{html.escape(title)}</h2>\n      <ul class="cards">{separator}'
                + separator.join(cards)
                + "\n      </ul>"
            )
    return "\n\n      ".join(sections)


def _course_tag(app: App) -> str:
    course = course_of(app.slug)
    name = COURSE_TAGS[course.home]
    return f"{name} · {app.source}" if app.source else name


def render_ist51_page() -> str:
    """The static front page of the ИСТ-51 course: its apps, grouped by topic, each marked with its lab module."""
    apps = [app for app in APPS if course_of(app.slug) is IST51]
    count = len(apps)
    intro = (
        '<p class="note">Площадки не заменяют лабораторные работы. Лабораторная показывает результат — '
        "площадка позволяет его покрутить: подвинуть параметр и увидеть, что изменилось. Значения по "
        "умолчанию на каждой площадке воспроизводят числа соответствующего занятия.</p>\n      "
        '<p class="note">Устанавливать ничего не нужно: Python исполняется в самом браузере. Первая '
        "загрузка страницы скачивает рантайм и занимает до минуты.</p>\n      "
        f'<p>Площадки обоих курсов сайта по темам — в <a href="../{CATALOG}/">каталоге</a>.</p>'
    )
    return _substitute(
        CATALOG_TEMPLATE,
        {
            "__TITLE__": f"Площадки курса «{IST51.title}»",
            "__SUBTITLE__": f"Б.1.2.2 · профиль ИСТ-51 · {count} {plural(count, 'площадка', 'площадки', 'площадок')}",
            "__INTRO__": intro,
            "__SECTIONS__": _sections(apps, IST51.home, lambda app: app.source),
            "__FOOTER__": html.escape(IST51_FOOTER),
        },
    )


def render_catalog() -> str:
    """The catalog: every app on the site, grouped by topic, each marked with its course."""
    courses = " и ".join(f"«{course.title}»" for course in COURSES)
    intro = (
        f'<p class="note">На сайте площадки двух курсов — {html.escape(courses)}. Здесь они собраны по темам: '
        "площадка одного курса часто отвечает на вопрос, который в другом остался за кадром.</p>\n      "
        '<p class="note">Устанавливать ничего не нужно: Python исполняется в самом браузере. Первая '
        "загрузка страницы скачивает рантайм и занимает до минуты.</p>\n      "
        f'<p>Курсы целиком: <a href="../">«{html.escape(COURSES[0].title)}»</a> начинается с градиентного '
        f'спуска, у <a href="../{IST51.home}/">«{html.escape(IST51.title)}»</a> есть своя страница.</p>'
    )
    count = len(APPS)
    return _substitute(
        CATALOG_TEMPLATE,
        {
            "__TITLE__": "Площадки по машинному обучению",
            "__SUBTITLE__": (
                f"{count} {plural(count, 'площадка', 'площадки', 'площадок')} двух курсов по {len(TOPICS)} темам"
            ),
            "__INTRO__": intro,
            "__SECTIONS__": _sections(list(APPS), CATALOG, _course_tag),
            "__FOOTER__": html.escape(FOOTER),
        },
    )


def build_app(app: App, site_dir: Path) -> Path:
    app_dir = site_dir / app.output_subdir if app.output_subdir else site_dir
    app_dir.mkdir(parents=True, exist_ok=True)

    for relative in collect_sources(app):
        destination = app_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)

    (app_dir / "index.html").write_text(render_index(app), encoding="utf-8")
    return app_dir


def build(output_dir: Path) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for app in APPS:
        build_app(app, output_dir)

    for path, text in ((IST51.home, render_ist51_page()), (CATALOG, render_catalog())):
        (output_dir / path).mkdir(parents=True, exist_ok=True)
        (output_dir / path / "index.html").write_text(text, encoding="utf-8")

    # Stop GitHub Pages' Jekyll step from dropping files and directories.
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")

    return output_dir


def main() -> None:
    # The summary names the apps, and most are named in Russian. A Windows
    # console defaults to cp1252, where that raises UnicodeEncodeError and the
    # script exits non-zero after having built the site correctly — a build that
    # looks broken locally and fine in CI.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist",
        help="directory to write the static site into (default: ./dist)",
    )
    args = parser.parse_args()

    output_dir = build(args.output.resolve())
    file_count = sum(1 for path in output_dir.rglob("*") if path.is_file())
    print(f"Built static site in {output_dir} ({file_count} files)")
    for app in APPS:
        print(f"  /{app.slug + '/' if app.slug else ''} -> {app.title}")
    print(f"  /{IST51.home}/ -> страница курса ИСТ-51")
    print(f"  /{CATALOG}/ -> каталог по темам")


if __name__ == "__main__":
    main()
