"""mkdocs hook: makes the same markdown work on GitHub and on the site.

* Links from a doc page to repository files outside docs/ (configs, presets, tools) point at the file on
  GitHub.
* A page whose only line is ``--8<-- "FILE"`` shows a file from the repository root (CHANGELOG.md,
  CONTRIBUTING.md); its links to ``docs/...`` are made relative to the docs folder.
"""

import posixpath
import re
from pathlib import Path

REPO = "https://github.com/BoredAmi/softsim/blob/main/"
ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r"(\]\()([^)\s#]+)([^)]*\))")
INCLUDE = re.compile(r'^--8<-- "([^"]+)"\s*$')


def on_page_markdown(markdown, page, **kwargs):
    base = posixpath.dirname(page.file.src_uri)
    include = INCLUDE.match(markdown.strip())
    if include:
        markdown = (ROOT / include.group(1)).read_text()
        base = ".."  # its links are relative to the repository root

    def fix(m):
        target = m.group(2)
        if "://" in target or target.startswith(("#", "mailto:")):
            return m.group(0)
        path = posixpath.normpath(posixpath.join("docs", base, target))
        if path.startswith("docs/"):
            rel = posixpath.relpath(path, posixpath.join("docs", posixpath.dirname(page.file.src_uri)))
            return f"{m.group(1)}{rel}{m.group(3)}"
        return f"{m.group(1)}{REPO}{path}{m.group(3)}"

    return LINK.sub(fix, markdown)
