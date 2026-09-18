"""Build the static GitHub Pages bundle.

Each Streamlit app is served through stlite, which runs CPython (Pyodide) in the
browser, so the published site is plain static files with no server component.

The Python sources are copied verbatim next to each ``index.html`` and referenced
by URL from the generated stlite manifest, so the deployed apps always match the
repository sources.

The gradient-descent app stays at the site root: it was published there first and
that URL is in circulation. Further apps get a subdirectory each.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "web" / "index.template.html"

# Pinned so a CDN release cannot silently change the deployed runtime.
STLITE_VERSION = "1.9.1"


@dataclass(frozen=True)
class App:
    """One stlite app on the published site."""

    slug: str  # "" means the site root
    entrypoint: str
    package: str
    title: str
    # numpy and pandas ship as prebuilt Pyodide wheels; the rest come from PyPI.
    requirements: tuple[str, ...] = ("numpy", "pandas", "plotly>=5.20,<8")
    data_globs: tuple[str, ...] = field(default=())

    @property
    def output_subdir(self) -> str:
        return self.slug


APPS = (
    App(
        slug="",
        entrypoint="gradient_descent_playground_v3.py",
        package="gd_playground",
        title="Gradient Descent Playground",
    ),
    App(
        slug="mt",
        entrypoint="mt_metrics_playground.py",
        package="mt_playground",
        title="Метрики машинного перевода",
        data_globs=("data/*.csv",),
    ),
    App(
        slug="lm",
        entrypoint="lm_text_playground.py",
        package="lm_playground",
        title="Языковая модель и температура",
        data_globs=("data/*.csv",),
    ),
    App(
        slug="vec",
        entrypoint="text_features_playground.py",
        package="vec_playground",
        title="Векторизация текста",
        # scikit-learn и nltk есть в сборке Pyodide готовыми колёсами.
        requirements=("numpy", "pandas", "plotly>=5.20,<8", "scikit-learn>=1.5", "nltk>=3.9"),
        data_globs=("data/*.csv",),
    ),
)


def collect_sources(app: App) -> list[str]:
    """Repo-relative POSIX paths of every file the app needs at runtime."""
    paths = [app.entrypoint]
    package_dir = ROOT / app.package
    paths += sorted(path.relative_to(ROOT).as_posix() for path in package_dir.glob("*.py"))
    for pattern in app.data_globs:
        paths += sorted(path.relative_to(ROOT).as_posix() for path in package_dir.glob(pattern))
    return paths


def site_manifest(app: App) -> dict[str, dict[str, str]]:
    """The stlite ``files`` mapping: virtual path -> URL on the site."""
    return {relative: {"url": f"./{relative}"} for relative in collect_sources(app)}


def render_index(app: App) -> str:
    """``index.html`` for one app, with every build-time placeholder substituted."""
    html = TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "__STLITE_VERSION__": STLITE_VERSION,
        "__TITLE__": app.title,
        "__ENTRYPOINT__": app.entrypoint,
        "__REQUIREMENTS__": json.dumps(list(app.requirements)),
        "__FILES__": json.dumps(site_manifest(app), indent=2),
    }
    for placeholder, value in replacements.items():
        if placeholder not in html:
            raise SystemExit(f"placeholder {placeholder} missing from {TEMPLATE}")
        html = html.replace(placeholder, value)
    return html


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

    # Stop GitHub Pages' Jekyll step from dropping files and directories.
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")

    return output_dir


def main() -> None:
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


if __name__ == "__main__":
    main()
