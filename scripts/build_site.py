"""Build the static GitHub Pages bundle.

The Streamlit app is served through stlite, which runs CPython (Pyodide) in the
browser, so the published site is plain static files with no server component.

The Python sources are copied verbatim next to ``index.html`` and referenced by
URL from the generated stlite manifest, so the deployed app always matches the
repository sources.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "web" / "index.template.html"

# Pinned so a CDN release cannot silently change the deployed runtime.
STLITE_VERSION = "1.9.1"

ENTRYPOINT = "gradient_descent_playground_v3.py"
PACKAGE = "gd_playground"

# numpy and pandas ship as prebuilt Pyodide wheels; plotly is resolved from PyPI.
# Kept in sync with requirements.txt by tests/test_site_build.py.
WEB_REQUIREMENTS = ["numpy", "pandas", "plotly>=5.20,<8"]


def collect_sources() -> list[str]:
    """Return repo-relative POSIX paths of every Python file the app needs."""
    paths = [ENTRYPOINT]
    paths += sorted(
        path.relative_to(ROOT).as_posix() for path in (ROOT / PACKAGE).glob("*.py")
    )
    return paths


def site_manifest() -> dict[str, dict[str, str]]:
    """Return the stlite ``files`` mapping: virtual path -> URL on the site."""
    return {relative: {"url": f"./{relative}"} for relative in collect_sources()}


def render_index() -> str:
    """Return ``index.html`` with every build-time placeholder substituted."""
    html = TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "__STLITE_VERSION__": STLITE_VERSION,
        "__ENTRYPOINT__": ENTRYPOINT,
        "__REQUIREMENTS__": json.dumps(WEB_REQUIREMENTS),
        "__FILES__": json.dumps(site_manifest(), indent=2),
    }
    for placeholder, value in replacements.items():
        if placeholder not in html:
            raise SystemExit(f"placeholder {placeholder} missing from {TEMPLATE}")
        html = html.replace(placeholder, value)
    return html


def build(output_dir: Path) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for relative in collect_sources():
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)

    (output_dir / "index.html").write_text(render_index(), encoding="utf-8")

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


if __name__ == "__main__":
    main()
