"""Text splitting with LangChain RecursiveCharacterTextSplitter + section metadata."""
from __future__ import annotations

from typing import Any


def _langchain_split(text: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Use LangChain splitter with metadata."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.create_documents([text])
    return [{"text": c.page_content, "metadata": c.metadata} for c in chunks]


def _fallback_split(text: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Simple sliding-window fallback."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append({"text": text[start:end], "metadata": {}})
        if end == len(text):
            break
        start += chunk_size - chunk_overlap
    return chunks


def split_document(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    section_label: str | None = None,
    extra_metadata: dict | None = None,
) -> list[dict]:
    """
    Split text into chunks with metadata.
    Each chunk dict: {text, metadata: {section, ...extra_metadata}}.
    """
    try:
        chunks = _langchain_split(text, chunk_size, chunk_overlap)
    except Exception:
        chunks = _fallback_split(text, chunk_size, chunk_overlap)

    # Attach section and extra metadata
    for chunk in chunks:
        chunk["metadata"]["section"] = section_label or "FULL"
        if extra_metadata:
            chunk["metadata"].update(extra_metadata)

    return chunks


def split_sections(
    sections: list,  # list of Section objects from preprocessing
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    extra_metadata: dict | None = None,
) -> list[dict]:
    """Split each section separately, preserving section label in metadata."""
    all_chunks = []
    for section in sections:
        chunks = split_document(
            section.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            section_label=section.label,
            extra_metadata=extra_metadata,
        )
        all_chunks.extend(chunks)
    return all_chunks
