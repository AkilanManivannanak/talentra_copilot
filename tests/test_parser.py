from app.services.document_parser import chunk_text, normalise_whitespace


def test_normalise_whitespace() -> None:
    assert normalise_whitespace("A\n\nB\t C") == "A B C"


def test_chunk_text_returns_multiple_chunks() -> None:
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
