"""Generate MkDocs stub files and landing page for Quarto-rendered notebooks.

Source of truth: notebooks/*.qmd
Generated:      docs/notebooks/<slug>.md   (iframe embed with absolute path + run-locally)
                docs/notebooks/index.md     (notebook listing page)
"""

import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import structlog
import yaml

logger = structlog.get_logger(__name__)

NOTEBOOKS_DIR = Path("notebooks")
DOCS_NOTEBOOKS_DIR = Path("docs/notebooks")
MKDOCS_CONFIG = Path("mkdocs.yml")


def get_base_path() -> str:
    """Extract the URL base path from site_url in mkdocs.yml."""
    cfg = yaml.safe_load(MKDOCS_CONFIG.read_text())
    raw = cfg.get("site_url", "")
    parsed = urlparse(raw)
    base = parsed.path.rstrip("/")
    return base if base else ""


def parse_frontmatter(qmd_path: Path) -> dict:
    text = qmd_path.read_text()
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return {}
    meta = {}
    for line in match.group(1).strip().splitlines():
        m = re.match(r"^(\w+):\s*\"(.+)\"$", line)
        if m:
            meta[m.group(1)] = m.group(2)
        else:
            m = re.match(r"^(\w+):\s*(.+)$", line)
            if m:
                meta[m.group(1)] = m.group(2)
    return meta


def get_notebooks():
    result = []
    for nb in sorted(NOTEBOOKS_DIR.glob("*.qmd")):
        if not nb.stem.startswith("_"):
            result.append(nb)
    return result


def generate_landing_page(notebooks):
    content = "# Notebooks\n\n"
    content += (
        "Explore interactive tutorials and benchmarks rendered as static HTML.\n"
        "The source `.qmd` files live in `notebooks/` — edit the source, not these stubs.\n\n"
    )
    for nb in notebooks:
        meta = parse_frontmatter(nb)
        title = meta.get("title", nb.stem.replace("_", " ").title())
        description = meta.get("description", "")
        slug = nb.stem
        desc_part = f" — {description}" if description else ""
        content += f"- [{title}]({slug}.md){desc_part}\n"
    content += "\n---\n\n*Generated from `notebooks/*.qmd` — edit the source, not this page.*\n"

    DOCS_NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_NOTEBOOKS_DIR / "index.md").write_text(content)
    logger.info(f"  {DOCS_NOTEBOOKS_DIR / 'index.md'}")


def generate_stubs(notebooks, base_path):
    for nb in notebooks:
        meta = parse_frontmatter(nb)
        title = meta.get("title", nb.stem.replace("_", " ").title())
        description = meta.get("description", "")
        slug = nb.stem
        iframe_src = f"{base_path}/notebooks/html/{slug}.html"

        stub = f"# {title}\n\n"
        if description:
            stub += f"{description}\n\n"

        stub += (
            '<div style="margin: 0 -0.8rem">\n'
            f'  <iframe src="{iframe_src}"'
            '    style="width:100%; height:600px;'
            " border:1px solid var(--md-default-fg-color--lightest);"
            ' border-radius:4px;"'
            '    loading="lazy"></iframe>\n'
            "</div>\n\n"
            "## Run Locally\n\n"
            "```bash\n"
            "uv sync --extra docs\n"
            "uv run quarto render notebooks/\n"
            "```\n"
        )

        stub_path = DOCS_NOTEBOOKS_DIR / f"{slug}.md"
        stub_path.write_text(stub)
        logger.info(f"  {stub_path}")


def main():
    base_path = get_base_path()
    landing_only = "--landing-only" in sys.argv
    notebooks = get_notebooks()

    if not landing_only:
        logger.info("Generating notebook stubs...")
        generate_stubs(notebooks, base_path)

    logger.info("Generating landing page...")
    generate_landing_page(notebooks)

    logger.info("Done.")


if __name__ == "__main__":
    main()
