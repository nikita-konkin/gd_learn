import json
import re
from pathlib import Path

import pytest

from playground_common.links import CATALOG, IST51, TOPICS, page_of
from scripts.build_site import (
    APPS,
    CATALOG_TEMPLATE,
    IST51_TEMPLATE,
    SHARED_PACKAGE,
    STLITE_VERSION,
    TEMPLATE,
    build,
    collect_sources,
    render_catalog,
    render_index,
    render_ist51_page,
    site_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDERS = ("__STLITE_VERSION__", "__TITLE__", "__ENTRYPOINT__", "__REQUIREMENTS__", "__FILES__")

GD_APP = next(app for app in APPS if app.slug == "")
MT_APP = next(app for app in APPS if app.slug == "mt")
LM_APP = next(app for app in APPS if app.slug == "lm")
VEC_APP = next(app for app in APPS if app.slug == "vec")
TM_APP = next(app for app in APPS if app.slug == "tm")
LABELS_APP = next(app for app in APPS if app.slug == "labels")
INTRO_APP = next(app for app in APPS if app.slug == "intro")
IST51_APPS = [app for app in APPS if app.slug.startswith(f"{IST51.home}/")]
PLACEHOLDER = re.compile(r"__[A-Z_]+__")


@pytest.mark.parametrize("app", APPS, ids=lambda app: app.slug or "root")
def test_collect_sources_includes_entrypoint_and_package(app):
    sources = collect_sources(app)

    assert app.entrypoint in sources
    assert f"{app.package}/__init__.py" in sources
    assert f"{app.package}/app.py" in sources


@pytest.mark.parametrize("app", APPS, ids=lambda app: app.slug or "root")
def test_every_app_ships_the_shared_package(app):
    """Each app is served alone, so each needs its own copy of the shared helpers."""
    sources = collect_sources(app)

    shared = sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / SHARED_PACKAGE).glob("*.py"))
    assert shared, "the shared package has no modules"
    assert set(shared) <= set(sources)


def test_mt_app_ships_its_data_files():
    sources = collect_sources(MT_APP)

    assert "mt_playground/data/loc_corpus.csv" in sources
    assert "mt_playground/data/semantic_ru_mt.csv" in sources


def test_lm_app_ships_its_training_corpus():
    assert "lm_playground/data/corpus_ru.csv" in collect_sources(LM_APP)


def test_vec_app_ships_its_corpus_and_declares_scikit_learn():
    assert "vec_playground/data/corpus_ru.csv" in collect_sources(VEC_APP)
    assert any(r.startswith("scikit-learn") for r in VEC_APP.requirements)
    assert any(r.startswith("nltk") for r in VEC_APP.requirements)


def test_tm_app_ships_its_corpus_and_its_pretrained_vectors():
    """The .npy files cannot be recomputed in the browser, so they must travel."""
    sources = collect_sources(TM_APP)

    assert "tm_playground/data/tm_corpus.csv" in sources
    assert "tm_playground/data/queries.csv" in sources
    assert "tm_playground/data/emb_tm.npy" in sources
    assert "tm_playground/data/emb_queries.npy" in sources
    assert any(r.startswith("scikit-learn") for r in TM_APP.requirements)


def test_labels_app_ships_its_data_and_scikit_learn_but_not_nltk():
    """Stemming happens offline; in the browser it would be a wheel for one number."""
    sources = collect_sources(LABELS_APP)

    for name in ("corpus.csv", "mqm_examples.csv", "remedies.csv", "newsgroups_curve.csv"):
        assert f"labels_playground/data/{name}" in sources
    assert any(r.startswith("scikit-learn") for r in LABELS_APP.requirements)
    assert not any(r.startswith("nltk") for r in LABELS_APP.requirements)


def test_intro_app_ships_its_corpus_and_scikit_learn_but_not_nltk():
    """The decision tree needs scikit-learn; nothing here stems."""
    assert "intro_playground/data/corpus.csv" in collect_sources(INTRO_APP)
    assert any(r.startswith("scikit-learn") for r in INTRO_APP.requirements)
    assert not any(r.startswith("nltk") for r in INTRO_APP.requirements)


def test_tm_app_does_not_ship_nltk():
    """It does no stemming; every extra wheel is seconds of load time."""
    assert not any(r.startswith("nltk") for r in TM_APP.requirements)


def test_apps_that_do_not_need_scikit_learn_do_not_ship_it():
    """Every extra wheel is seconds of load time in the browser."""
    for app in (GD_APP, MT_APP, LM_APP):
        assert not any(r.startswith("scikit-learn") for r in app.requirements)


def test_gd_app_ships_no_data_files():
    assert all(not path.endswith(".csv") for path in collect_sources(GD_APP))


@pytest.mark.parametrize("app", APPS, ids=lambda app: app.slug or "root")
def test_manifest_maps_every_source_to_a_relative_url(app):
    manifest = site_manifest(app)

    assert set(manifest) == set(collect_sources(app))
    for relative, entry in manifest.items():
        assert entry == {"url": f"./{relative}"}


@pytest.mark.parametrize("app", APPS, ids=lambda app: app.slug or "root")
def test_index_html_is_fully_rendered(app):
    html = render_index(app)

    for placeholder in PLACEHOLDERS:
        assert placeholder not in html
    assert f'entrypoint: "{app.entrypoint}"' in html
    assert json.dumps(site_manifest(app), indent=2) in html
    assert json.dumps(list(app.requirements)) in html
    assert f"<title>{app.title}</title>" in html


def test_apps_have_distinct_slugs_and_one_root():
    slugs = [app.slug for app in APPS]

    assert len(slugs) == len(set(slugs))
    assert slugs.count("") == 1, "exactly one app lives at the site root"


def test_build_lays_out_root_and_subdirectory_apps(tmp_path):
    site = build(tmp_path / "dist")

    assert (site / ".nojekyll").exists()
    assert (site / "index.html").read_text(encoding="utf-8") == render_index(GD_APP)
    assert (site / "mt" / "index.html").read_text(encoding="utf-8") == render_index(MT_APP)
    assert (site / "lm" / "index.html").read_text(encoding="utf-8") == render_index(LM_APP)
    assert (site / "vec" / "index.html").read_text(encoding="utf-8") == render_index(VEC_APP)
    assert (site / "tm" / "index.html").read_text(encoding="utf-8") == render_index(TM_APP)


def test_apps_do_not_leak_each_others_files(tmp_path):
    """Each app gets its own package and the shared one, nothing else."""
    site = build(tmp_path / "dist")

    for app in APPS:
        folder = site / app.output_subdir
        assert {path.name for path in folder.glob("*_playground")} == {app.package}, app.slug or "root"
        assert (folder / SHARED_PACKAGE / "__init__.py").exists()
    assert not (site / "gd_playground" / "data").exists()


@pytest.mark.parametrize("app", APPS, ids=lambda app: app.slug or "root")
def test_build_copies_every_source_byte_for_byte(tmp_path, app):
    site = build(tmp_path / "dist")
    app_dir = site / app.slug if app.slug else site

    for relative in collect_sources(app):
        copied = app_dir / relative
        assert copied.exists(), f"{relative} was not copied into the site"
        assert copied.read_bytes() == (ROOT / relative).read_bytes()


def test_build_replaces_stale_output(tmp_path):
    output_dir = tmp_path / "dist"
    output_dir.mkdir()
    (output_dir / "stale.py").write_text("removed on rebuild", encoding="utf-8")

    build(output_dir)

    assert not (output_dir / "stale.py").exists()


def test_web_plotly_pin_matches_requirements():
    requirement = next(
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("plotly")
    )

    for app in APPS:
        assert requirement in app.requirements


# ---------------------------------------------------------------- ИСТ-51


def test_the_ist51_course_has_its_six_apps():
    assert [app.slug for app in IST51_APPS] == [
        "ml-practice/data",
        "ml-practice/leak",
        "ml-practice/metric",
        "ml-practice/fit",
        "ml-practice/nn",
        "ml-practice/dag",
    ]


def test_ist51_apps_declare_scikit_learn_only_where_they_use_it():
    """leak and fit fit models in the browser; metric and nn ship numbers prepared offline."""
    needs_sklearn = {"ml-practice/leak", "ml-practice/fit"}
    for app in IST51_APPS:
        declared = any(r.startswith("scikit-learn") for r in app.requirements)
        assert declared is (app.slug in needs_sklearn), app.slug


def test_ist51_apps_use_their_own_page_template_and_the_others_the_original():
    for app in APPS:
        assert app.template == (IST51_TEMPLATE if app in IST51_APPS else TEMPLATE), app.slug or "root"
    assert 'lang="ru"' in IST51_TEMPLATE.read_text(encoding="utf-8")


def test_no_app_ships_another_apps_package():
    packages = {app.package for app in APPS}
    for app in APPS:
        sources = collect_sources(app)
        for other in packages - {app.package}:
            assert not any(path.startswith(f"{other}/") for path in sources), (app.slug, other)


# ------------------------------------------------------- the static pages


def test_the_ist51_front_page_links_every_course_app_by_topic():
    html = render_ist51_page()

    for app in IST51_APPS:
        assert f'href="{app.slug.removeprefix(IST51.home + "/")}/"' in html
    assert f'href="../{CATALOG}/"' in html
    assert PLACEHOLDER.search(html) is None


def test_the_catalog_links_every_app_once_and_names_every_topic():
    html = render_catalog()

    for app in APPS:
        href = f"../{app.slug}/" if app.slug else "../"
        assert html.count(f'<a href="{href}">{page_of(app.slug).title}</a>') == 1, app.slug or "root"
    for key, title in TOPICS:
        assert f'<h2 id="{key}">{title}</h2>' in html
    assert PLACEHOLDER.search(html) is None


def test_the_catalog_marks_each_app_with_its_course():
    html = render_catalog()

    assert html.count('class="tag">Основы машинного обучения') == len(APPS) - len(IST51_APPS)
    assert html.count('class="tag">ИСТ-51 · ') == len(IST51_APPS)


@pytest.mark.parametrize("render", [render_ist51_page, render_catalog], ids=["ist51", "catalog"])
def test_static_pages_need_no_python_runtime(render):
    html = render()

    assert "stlite" not in html
    assert "<script" not in html
    assert 'lang="ru"' in html


def test_build_writes_both_static_pages(tmp_path):
    site = build(tmp_path / "dist")

    assert (site / IST51.home / "index.html").read_text(encoding="utf-8") == render_ist51_page()
    assert (site / CATALOG / "index.html").read_text(encoding="utf-8") == render_catalog()
    assert CATALOG_TEMPLATE.exists()


def test_the_pinned_stlite_version_is_exact():
    """A range would let a CDN release change the deployed runtime silently."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", STLITE_VERSION)


def test_the_builder_needs_nothing_beyond_the_standard_library():
    """CI's build job installs no packages, so the builder must not import numpy and the like."""
    import subprocess
    import sys

    blocked = "import sys; sys.modules.update(dict.fromkeys(['numpy', 'pandas', 'sklearn', 'streamlit'])); "
    result = subprocess.run(
        [sys.executable, "-c", blocked + "import scripts.build_site"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
