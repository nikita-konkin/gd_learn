import json
from pathlib import Path

from scripts.build_site import (
    ENTRYPOINT,
    WEB_REQUIREMENTS,
    build,
    collect_sources,
    render_index,
    site_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDERS = ("__STLITE_VERSION__", "__ENTRYPOINT__", "__REQUIREMENTS__", "__FILES__")


def test_collect_sources_includes_entrypoint_and_package():
    sources = collect_sources()

    assert ENTRYPOINT in sources
    assert "gd_playground/app.py" in sources
    assert "gd_playground/__init__.py" in sources


def test_manifest_maps_every_source_to_a_relative_url():
    manifest = site_manifest()

    assert set(manifest) == set(collect_sources())
    for relative, entry in manifest.items():
        assert entry == {"url": f"./{relative}"}


def test_index_html_is_fully_rendered():
    html = render_index()

    for placeholder in PLACEHOLDERS:
        assert placeholder not in html
    assert f'entrypoint: "{ENTRYPOINT}"' in html
    assert json.dumps(site_manifest(), indent=2) in html
    assert json.dumps(WEB_REQUIREMENTS) in html


def test_build_copies_every_source_and_adds_nojekyll(tmp_path):
    output_dir = build(tmp_path / "dist")

    assert (output_dir / ".nojekyll").exists()
    assert (output_dir / "index.html").read_text(encoding="utf-8") == render_index()
    for relative in collect_sources():
        copied = output_dir / relative
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

    assert requirement in WEB_REQUIREMENTS
