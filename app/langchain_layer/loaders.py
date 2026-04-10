"""Document loaders wrapping LangChain with graceful fallback to raw text."""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Union


def _try_langchain_pdf(path: str) -> str | None:
    try:
        from langchain_community.document_loaders import PyPDFLoader  # type: ignore
        loader = PyPDFLoader(path)
        pages = loader.load()
        return "\n\n".join(p.page_content for p in pages)
    except Exception:
        return None


def _try_langchain_docx(path: str) -> str | None:
    try:
        from langchain_community.document_loaders import Docx2txtLoader  # type: ignore
        loader = Docx2txtLoader(path)
        docs = loader.load()
        return "\n\n".join(d.page_content for d in docs)
    except Exception:
        return None


def _try_pypdf(path: str) -> str | None:
    try:
        import PyPDF2  # type: ignore
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        return None


def _try_python_docx(path: str) -> str | None:
    try:
        import docx  # type: ignore
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return None


def load_document(source: Union[str, Path, bytes], filename: str = "") -> str:
    """
    Load text from a file path or raw bytes.
    Tries LangChain loaders first, then PyPDF2/python-docx, then raw decode.
    """
    if isinstance(source, (str, Path)):
        path = str(source)
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return _try_langchain_pdf(path) or _try_pypdf(path) or ""
        if ext in (".docx", ".doc"):
            return _try_langchain_docx(path) or _try_python_docx(path) or ""
        # Text-based files
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    # Bytes path (in-memory upload)
    ext = Path(filename).suffix.lower() if filename else ""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(source)
        tmp_path = tmp.name
    try:
        return load_document(tmp_path, filename)
    finally:
        os.unlink(tmp_path)
