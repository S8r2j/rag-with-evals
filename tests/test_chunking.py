import pytest

from app.ingestion.chunking import Chunker, get_tokenizer
from app.ingestion.extract import PageText

tokenizer = get_tokenizer("o200k_base")


def count(text: str) -> int:
    return len(tokenizer.encode(text))


def sentence(i: int) -> str:
    return f"Sentence number {i} describes a distinct fact about the system."


def chunker(max_tokens: int = 60, overlap: int = 15) -> Chunker:
    return Chunker(tokenizer, max_tokens=max_tokens, overlap_tokens=overlap)


def test_short_text_becomes_a_single_chunk() -> None:
    chunks = chunker().chunk([PageText(1, "A short paragraph.")])
    assert len(chunks) == 1
    assert chunks[0].text == "A short paragraph."
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 1)
    assert chunks[0].index == 0


def test_no_chunk_exceeds_the_token_limit() -> None:
    text = " ".join(sentence(i) for i in range(200))
    chunks = chunker(max_tokens=60, overlap=15).chunk([PageText(1, text)])
    assert len(chunks) > 5
    assert all(chunk.token_count <= 60 for chunk in chunks)
    assert all(chunk.token_count == count(chunk.text) for chunk in chunks)


def test_all_content_is_kept_in_order() -> None:
    text = " ".join(sentence(i) for i in range(50))
    chunks = chunker(max_tokens=60, overlap=0).chunk([PageText(1, text)])
    rebuilt = " ".join(chunk.text for chunk in chunks)
    assert rebuilt == text
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_overlap() -> None:
    text = " ".join(sentence(i) for i in range(40))
    chunks = chunker(max_tokens=60, overlap=20).chunk([PageText(1, text)])
    for previous, current in zip(chunks, chunks[1:], strict=False):
        last_sentence = previous.text.split(". ")[-1]
        assert last_sentence in current.text


def test_paragraph_boundaries_are_preserved() -> None:
    pages = [PageText(1, "First paragraph.\n\nSecond paragraph.")]
    chunks = chunker().chunk(pages)
    assert chunks[0].text == "First paragraph.\n\nSecond paragraph."


def test_chunks_record_the_pages_they_span() -> None:
    pages = [
        PageText(1, " ".join(sentence(i) for i in range(3))),
        PageText(2, " ".join(sentence(i) for i in range(3, 6))),
        PageText(3, " ".join(sentence(i) for i in range(6, 9))),
    ]
    chunks = chunker(max_tokens=100, overlap=0).chunk(pages)
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 3
    assert any(chunk.page_start != chunk.page_end for chunk in chunks)
    for chunk in chunks:
        assert chunk.page_start <= chunk.page_end


def test_documents_without_pages_have_no_page_numbers() -> None:
    chunks = chunker().chunk([PageText(None, "Text from a Word document.")])
    assert (chunks[0].page_start, chunks[0].page_end) == (None, None)


def test_text_without_any_punctuation_is_split_by_tokens() -> None:
    text = " ".join(["word"] * 500)
    chunks = chunker(max_tokens=60, overlap=10).chunk([PageText(1, text)])
    assert len(chunks) > 1
    assert all(chunk.token_count <= 60 for chunk in chunks)


def test_empty_and_blank_pages_produce_no_chunks() -> None:
    assert chunker().chunk([PageText(1, ""), PageText(2, "\n\n  \n")]) == []


def test_a_single_oversized_paragraph_is_not_duplicated_by_overlap() -> None:
    paragraph = " ".join(sentence(i) for i in range(4))  # ~ 50 tokens
    pages = [PageText(1, paragraph + "\n\n" + paragraph.replace("Sentence", "Line"))]
    chunks = chunker(max_tokens=60, overlap=20).chunk(pages)
    assert len(chunks) == 2
    assert "Line" not in chunks[0].text
    assert "Sentence" not in chunks[1].text


@pytest.mark.parametrize(("max_tokens", "overlap"), [(50, 50), (50, 80), (50, -1)])
def test_invalid_overlap_is_rejected(max_tokens: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        Chunker(tokenizer, max_tokens=max_tokens, overlap_tokens=overlap)
