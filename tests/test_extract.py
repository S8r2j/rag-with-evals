import pytest

from app.ingestion.extract import ExtractionError, extract_text, normalize_text
from tests.factories import make_blank_pdf, make_docx, make_encrypted_pdf, make_pdf

PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def test_pdf_text_is_extracted_per_page() -> None:
    data = make_pdf(["Revenue grew 12 percent.", "Costs fell in the third quarter."])
    document = extract_text(data, PDF, max_pages=10)
    assert document.page_count == 2
    assert [page.number for page in document.pages] == [1, 2]
    assert "Revenue grew 12 percent." in document.pages[0].text
    assert "third quarter" in document.pages[1].text


def test_blank_pages_keep_their_numbers() -> None:
    data = make_pdf(["First page", "", "Third page"])
    document = extract_text(data, PDF, max_pages=10)
    assert [page.number for page in document.pages] == [1, 2, 3]
    assert document.pages[1].text == ""


def test_pdf_without_any_text_explains_ocr() -> None:
    with pytest.raises(ExtractionError, match="OCR"):
        extract_text(make_blank_pdf(2), PDF, max_pages=10)


def test_password_protected_pdf_is_reported() -> None:
    with pytest.raises(ExtractionError, match="password-protected"):
        extract_text(make_encrypted_pdf("confidential"), PDF, max_pages=10)


def test_page_limit_is_enforced() -> None:
    with pytest.raises(ExtractionError, match="limit is 2"):
        extract_text(make_pdf(["a", "b", "c"]), PDF, max_pages=2)


def test_damaged_pdf_is_reported() -> None:
    with pytest.raises(ExtractionError, match="damaged"):
        extract_text(b"%PDF-1.7\nthis is not really a pdf", PDF, max_pages=10)


def test_docx_paragraphs_and_tables_are_extracted() -> None:
    data = make_docx(
        ["Quarterly report", "Sales were strong."],
        table=[["Region", "Revenue"], ["Asia", "120"]],
    )
    document = extract_text(data, DOCX, max_pages=10)
    assert document.page_count is None
    assert len(document.pages) == 1 and document.pages[0].number is None
    text = document.pages[0].text
    assert "Quarterly report\n\nSales were strong." in text
    assert "Region | Revenue" in text and "Asia | 120" in text


def test_damaged_docx_is_reported() -> None:
    with pytest.raises(ExtractionError, match="damaged"):
        extract_text(b"PK\x03\x04 broken zip", DOCX, max_pages=10)


def test_plain_text_handles_bom_and_unicode() -> None:
    document = extract_text("﻿नमस्ते world".encode(), "text/plain", max_pages=10)
    assert document.pages[0].text == "नमस्ते world"


def test_empty_text_file_is_reported() -> None:
    with pytest.raises(ExtractionError):
        extract_text(b"   \n\n  ", "text/plain", max_pages=10)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("informa-\ntion retrieval", "information retrieval"),
        ("too    many   spaces", "too many spaces"),
        ("a\r\nb\rc", "a\nb\nc"),
        ("para one\n\n\n\n\npara two", "para one\n\npara two"),
        ("  trailing spaces  \n  next line ", "trailing spaces\nnext line"),
        ("null\x00byte", "nullbyte"),
        ("well-known term", "well-known term"),
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected
