"""LangChain-backed document loading, splitting, and prompt templates.

Retrieval lives in `app.services.hybrid`, not here: the previous `TalentraVectorStore`
in this package was a second, unreachable retrieval implementation whose Chroma/FAISS
branches could never fire (its `_get_embeddings()` returned None unconditionally). It
has been removed rather than left in place advertising backends it never used.
"""
from .loaders import load_document
from .prompts import (
    COPILOT_PROMPT,
    EVALUATION_PROMPT,
    INTERVIEW_QUESTION_PROMPT,
    REQUIREMENT_EXTRACTION_PROMPT,
    format_prompt,
    get_copilot_chain,
    get_evaluation_chain,
)
from .splitter import split_document, split_sections

__all__ = [
    "load_document",
    "split_document",
    "split_sections",
    "COPILOT_PROMPT",
    "EVALUATION_PROMPT",
    "INTERVIEW_QUESTION_PROMPT",
    "REQUIREMENT_EXTRACTION_PROMPT",
    "format_prompt",
    "get_copilot_chain",
    "get_evaluation_chain",
]
