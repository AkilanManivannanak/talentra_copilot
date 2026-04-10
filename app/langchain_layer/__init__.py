"""LangChain-powered document loading, splitting, embedding, and retrieval."""
from .loaders import load_document
from .splitter import split_document
from .vectorstore import TalentraVectorStore
from .prompts import (
    EVALUATION_PROMPT,
    COPILOT_PROMPT,
    REQUIREMENT_EXTRACTION_PROMPT,
    get_evaluation_chain,
    get_copilot_chain,
)

__all__ = [
    "load_document",
    "split_document",
    "TalentraVectorStore",
    "EVALUATION_PROMPT",
    "COPILOT_PROMPT",
    "REQUIREMENT_EXTRACTION_PROMPT",
    "get_evaluation_chain",
    "get_copilot_chain",
]
