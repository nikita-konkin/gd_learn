import json
from pathlib import Path

import pytest

from scripts.build_site import APPS, build, collect_sources, render_index, site_manifest

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDERS = ("__STLITE_VERSION__", "__TITLE__", "__ENTRYPOINT__", "__REQUIREMENTS__", "__FILES__")

GD_APP = next(app for app in APPS if app.slug == "")
MT_APP = next(app for app in APPS if app.slug == "mt")
LM_APP = next(app for app in APPS if app.slug == "lm")
VEC_APP = next(app for app in APPS if app.slug == "vec")
TM_APP = next(app for app in APPS if app.slug == "tm")


@pytest.mark.parametrize("app", APPS, ids=lambda app: app.slug or "root")
def test_collect_sources_includes_entrypoint_and_package(app):
    sources = collect_sources(app)

    assert app.entrypoint in sources
    assert f"{app.package}/__init__.py" in sources
    assert f"{app.package}/app.py" in sources


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
    """Each app gets only its own package."""
    site = build(tmp_path / "dist")

    assert not (site / "mt" / "lm_playground").exists()
    assert not (site / "lm" / "mt_playground").exists()
    assert not (site / "vec" / "lm_playground").exists()
    assert not (site / "tm" / "vec_playground").exists()
    assert not (site / "vec" / "tm_playground").exists()
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
