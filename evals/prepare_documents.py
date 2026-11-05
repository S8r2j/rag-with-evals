"""Download the evaluation documents and clean them into plain Markdown.

The documents are openly licensed company handbook pages, pinned to a fixed
commit so the evaluation always runs against the same text. Cleaning only
removes website-specific markup (front matter, MDX components, HTML tables);
the wording is unchanged.

    python -m evals.prepare_documents
"""

import re
import sys
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

DOCUMENTS_DIR = Path(__file__).parent / "documents"


@dataclass(frozen=True)
class SourceDocument:
    name: str
    repo: str
    commit: str
    path: str
    licence: str


POSTHOG = ("PostHog/posthog.com", "4fc3246084b49b31922fd4fc9482505b12367efe", "MIT")
SOURCEGRAPH = ("sourcegraph/handbook", "5752928b213a3ddbc7cee4459524f404a5b96fb4", "Apache-2.0")

SOURCES = [
    SourceDocument(
        f"posthog-{page}.md",
        POSTHOG[0],
        POSTHOG[1],
        f"contents/handbook/people/{page}.md",
        POSTHOG[2],
    )
    for page in ["time-off", "benefits", "offboarding", "side-gigs", "training", "grievances"]
] + [
    SourceDocument(
        f"sourcegraph-{name}.md",
        SOURCEGRAPH[0],
        SOURCEGRAPH[1],
        f"content/departments/people-talent/{page}.md",
        SOURCEGRAPH[2],
    )
    for name, page in [
        ("wellness", "wellness"),
        ("learning-and-development", "learning-and-development"),
        ("equity-refresh", "equity-refresh-program"),
    ]
]


class _TableParser(HTMLParser):
    """Collects the text of each <td>/<th> cell, row by row."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None and self.rows:
            self.rows[-1].append(" ".join("".join(self._cell).split()))
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def html_table_to_markdown(html: str) -> str:
    parser = _TableParser()
    parser.feed(html)
    rows = [row for row in parser.rows if any(cell for cell in row)]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    # A row with a single filled cell is a caption ("Version 2"), not data.
    lines: list[str] = []
    table: list[list[str]] = []
    for row in rows:
        if sum(1 for cell in row if cell) == 1 and row[0]:
            if table:
                lines.append(_render(table))
                table = []
            lines.append(f"**{row[0]}**")
        else:
            table.append(row)
    if table:
        lines.append(_render(table))
    return "\n\n".join(lines)


def _render(rows: list[list[str]]) -> str:
    header, *body = rows
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(out)


def clean(markdown: str) -> str:
    text = re.sub(r"\A---\n.*?\n---\n", "", markdown, flags=re.S)  # front matter
    text = re.sub(
        r"<table>.*?</table>", lambda m: html_table_to_markdown(m.group(0)), text, flags=re.S
    )
    text = re.sub(r"<PrivateLink[^>]*>\s*(.*?)\s*</PrivateLink>", r"\1", text, flags=re.S)
    text = re.sub(
        r'<SmallTeam slug="([^"]+)"\s*/>',
        lambda m: f"{m.group(1).replace('-', ' ').title()} team",
        text,
    )
    text = re.sub(r"</?[A-Za-z][^>]*>", "", text)  # any remaining tags
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def raw_url(source: SourceDocument) -> str:
    return f"https://raw.githubusercontent.com/{source.repo}/{source.commit}/{source.path}"


def main() -> int:
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    for source in SOURCES:
        with urllib.request.urlopen(raw_url(source), timeout=30) as response:
            original = response.read().decode("utf-8")
        (DOCUMENTS_DIR / source.name).write_text(clean(original), encoding="utf-8")
        print(f"{source.name:45} {source.licence:11} {raw_url(source)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
