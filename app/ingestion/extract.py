"""Turn uploaded files into plain text, page by page where the format has pages.

Page numbers are kept so answers can cite where their evidence came from.
Formats without real pages (DOCX, plain text) yield a single page with
``number=None``: Word documents only have pages once rendered, so any number
we invented would not match what the user sees.
"""

import io
import re
from dataclasses import dataclass

import docx
from pypdf import PdfReader
from pypdf.errors import PdfReadError


class ExtractionError(Exception):
    """The file is valid enough to accept but its text can't be extracted.

    The message is shown to the user, so it should say what to do next.
    """


@dataclass(frozen=True)
class PageText:
    number: int | None
    text: str


@dataclass(frozen=True)
class ExtractedDocument:
    pages: list[PageText]
    page_count: int | None


_HYPHENATED_LINE_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t\f\v ]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Clean extracted text without changing its meaning.

    Rejoins words split across lines ("informa-\\ntion"), collapses runs of
    spaces, trims each line and keeps at most one blank line between
    paragraphs, since blank lines are what the chunker splits on.
    """
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHENATED_LINE_BREAK.sub(r"\1\2", text)
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return _EXCESS_BLANK_LINES.sub("\n\n", text).strip()


def extract_text(data: bytes, content_type: str, *, max_pages: int) -> ExtractedDocument:
    if content_type == "application/pdf":
        document = _extract_pdf(data, max_pages=max_pages)
    elif content_type.endswith("wordprocessingml.document"):
        document = _extract_docx(data)
    elif content_type.startswith("text/"):
        document = _extract_plain_text(data)
    else:
        raise ExtractionError(f"Text extraction is not supported for {content_type}")

    if not any(page.text for page in document.pages):
        raise ExtractionError(
            "No text could be extracted. If this is a scanned document, it needs OCR, "
            "which is not supported yet."
        )
    return document


def _extract_pdf(data: bytes, *, max_pages: int) -> ExtractedDocument:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ExtractionError(
                "The PDF is password-protected. Remove the password and upload it again."
            )
        page_count = len(reader.pages)
        if page_count > max_pages:
            raise ExtractionError(f"The PDF has {page_count} pages; the limit is {max_pages}.")
        pages = [
            PageText(number=index, text=normalize_text(page.extract_text() or ""))
            for index, page in enumerate(reader.pages, start=1)
        ]
    except ExtractionError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError) as exc:
        raise ExtractionError("The PDF appears to be damaged and could not be read.") from exc
    return ExtractedDocument(pages=pages, page_count=page_count)


def _extract_docx(data: bytes) -> ExtractedDocument:
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:  # python-docx raises a variety of zip/XML errors
        raise ExtractionError("The DOCX file appears to be damaged and could not be read.") from exc

    blocks = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            blocks.append(" | ".join(cell for cell in cells if cell))
    text = normalize_text("\n\n".join(block for block in blocks if block.strip()))
    return ExtractedDocument(pages=[PageText(number=None, text=text)], page_count=None)


def _extract_plain_text(data: bytes) -> ExtractedDocument:
    text = normalize_text(data.decode("utf-8-sig", errors="replace"))
    return ExtractedDocument(pages=[PageText(number=None, text=text)], page_count=None)
