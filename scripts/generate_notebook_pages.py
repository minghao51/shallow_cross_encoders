#!/usr/bin/env python3
"""Generate mkdocs notebook landing page + export static HTML.

Source of truth: notebooks/*.py
Generated:  docs/notebooks/index.md  (landing page)
            docs/notebooks/exports/<slug>/index.html  (static export)
"""

import ast
import subprocess
import sys
from pathlib import Path


def extract_title(notebook_path: Path) -> str:
    tree = ast.parse(notebook_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "App":
                for kw in node.keywords:
                    if kw.arg == "app_title" and isinstance(kw.value, ast.Constant):
                        return kw.value.value
    return notebook_path.stem.replace("_", " ").title()


def get_notebooks():
    notebooks_dir = Path("notebooks")
    result = []
    for nb in sorted(notebooks_dir.glob("*.py")):
        if not nb.stem.startswith("_"):
            result.append(nb)
    return result


def generate_landing_page(notebooks):
    content = "# Notebooks\n\n"
    for nb in notebooks:
        slug = nb.stem
        title = extract_title(nb)
        export_path = f"exports/{slug}/"
        content += f"- [{title}]({export_path})\n"
    content += "\n---\n\n*Generated from `notebooks/*.py` — edit the source, not this page.*\n"

    landing_dir = Path("docs/notebooks")
    landing_dir.mkdir(parents=True, exist_ok=True)
    (landing_dir / "index.md").write_text(content)
    print(f"  {landing_dir / 'index.md'}")


def export_html(notebooks):
    export_base = Path("docs/notebooks") / "exports"
    for nb in notebooks:
        slug = nb.stem
        export_dir = export_base / slug
        export_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Exporting {slug}...")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "html",
                str(nb),
                "-o",
                str(export_dir / "index.html"),
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )
        if result.returncode != 0:
            print(f"  WARNING: export exited with code {result.returncode}")
        for line in result.stderr.splitlines():
            if "Error" in line or "Warning" in line or "traceback" in line.lower():
                print(f"    {line}")


def main():
    landing_only = "--landing-only" in sys.argv

    notebooks = get_notebooks()
    print("Generating landing page...")
    generate_landing_page(notebooks)

    if not landing_only:
        print("Exporting to static HTML...")
        export_html(notebooks)

    print("Done.")


if __name__ == "__main__":
    main()
