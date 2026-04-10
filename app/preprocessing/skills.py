"""Skill entity extraction with spaCy NER + curated keyword fallback."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

# ---------------------------------------------------------------------------
# Curated skill taxonomy (extend as needed)
# ---------------------------------------------------------------------------
SKILL_TAXONOMY: dict[str, list[str]] = {
    # Languages
    "python": ["python", "py", "python3"],
    "javascript": ["javascript", "js", "es6", "typescript", "ts"],
    "java": ["java", "jvm"],
    "go": ["golang", "go lang"],
    "rust": ["rust-lang", "rust"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "sqlite", "t-sql", "pl/sql"],
    # ML / AI
    "machine learning": ["machine learning", "ml", "deep learning", "dl", "ai", "artificial intelligence"],
    "llm": ["llm", "large language model", "gpt", "bert", "transformer", "fine-tuning", "fine tuning", "finetuning"],
    "rag": ["rag", "retrieval augmented generation", "retrieval-augmented"],
    "nlp": ["nlp", "natural language processing", "text mining"],
    # Frameworks
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "tf", "keras"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "langchain": ["langchain", "lang chain"],
    "langgraph": ["langgraph", "lang graph"],
    # Data
    "pandas": ["pandas", "dataframe"],
    "spark": ["apache spark", "pyspark", "spark"],
    "dbt": ["dbt", "data build tool"],
    # APIs / infra
    "fastapi": ["fastapi", "fast api"],
    "docker": ["docker", "dockerfile", "container", "containerization"],
    "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "amazon web services", "s3", "ec2", "lambda", "sagemaker"],
    "gcp": ["gcp", "google cloud", "bigquery", "vertex ai"],
    "azure": ["azure", "microsoft azure"],
    # Retrieval
    "vector search": ["vector search", "vector store", "faiss", "chroma", "pinecone", "weaviate", "qdrant"],
    "elasticsearch": ["elasticsearch", "elastic search", "opensearch"],
    # Practices
    "ci/cd": ["ci/cd", "ci cd", "github actions", "jenkins", "gitlab ci"],
    "devops": ["devops", "sre", "site reliability"],
    "mlops": ["mlops", "ml ops", "model serving", "model deployment"],
    "api design": ["rest api", "restful", "graphql", "grpc", "openapi"],
    "testing": ["pytest", "unit test", "integration test", "tdd"],
}

# Invert for O(1) lookup: normalized_alias -> canonical_skill
_ALIAS_MAP: dict[str, str] = {}
for canonical, aliases in SKILL_TAXONOMY.items():
    for alias in aliases:
        _ALIAS_MAP[alias.lower()] = canonical

# Regex that splits on non-alphanumeric boundaries but keeps internal slashes/dots
_TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9\./\-+#]*")


@lru_cache(maxsize=1)
def _load_spacy():
    """Try to load spaCy model; return None if unavailable."""
    try:
        import spacy  # type: ignore
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def extract_skills(text: str, use_spacy: bool = True) -> list[str]:
    """
    Extract canonical skill names from text.
    Uses spaCy NER for PRODUCT/ORG/SKILL entities first, then regex keyword scan.
    Returns deduplicated list of canonical skill names.
    """
    found: set[str] = set()

    if use_spacy:
        nlp = _load_spacy()
        if nlp is not None:
            doc = nlp(text[:50_000])  # spaCy limit guard
            for ent in doc.ents:
                if ent.label_ in ("PRODUCT", "ORG", "LANGUAGE", "SKILL"):
                    norm = ent.text.lower().strip()
                    if norm in _ALIAS_MAP:
                        found.add(_ALIAS_MAP[norm])

    # Regex fallback / supplement — scan all n-grams up to 4 tokens
    tokens = _TOKEN_RE.findall(text.lower())
    for n in (1, 2, 3, 4):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            if phrase in _ALIAS_MAP:
                found.add(_ALIAS_MAP[phrase])

    return sorted(found)


def skill_overlap(skills_a: list[str], skills_b: list[str]) -> float:
    """Jaccard overlap between two skill lists (0-1)."""
    a, b = set(skills_a), set(skills_b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
