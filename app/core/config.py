from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Storage -----------------------------------------------------------
    data_dir: str = "./data"
    vectorstore_path: str = "./data/vectorstore"
    max_upload_mb: int = 8

    # --- Requirement extraction -------------------------------------------
    max_requirements: int = 8
    match_threshold: float = 0.28

    # --- Retrieval ---------------------------------------------------------
    # lexical  : IDF/phrase scoring only (zero extra deps, always available)
    # hybrid   : lexical + dense fused with reciprocal rank fusion
    # dense    : dense only (diagnostic; used for ablations)
    retrieval_mode: Literal["lexical", "hybrid", "dense"] = "hybrid"
    # "lsa" needs only scikit-learn and downloads nothing. Set to a sentence-transformers
    # model id (e.g. BAAI/bge-small-en-v1.5) after installing requirements-ml.txt.
    embedding_model: str = "lsa"
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rrf_k: int = 60
    dense_candidates: int = 50

    # --- Generation --------------------------------------------------------
    # rule-based synthesis is the default and needs no model, no key, no network.
    chat_model: str = "local-rule-based"
    openai_api_key: str = ""

    # --- Preprocessing -----------------------------------------------------
    redact_pii_on_ingest: bool = True
    # Build the spaCy/Presidio pipelines on a background thread at startup instead of
    # making the first upload pay ~8s of model construction.
    prewarm_models: bool = True
    use_spacy: bool = True

    # --- Agentic path ------------------------------------------------------
    agentic_enabled: bool = True
    agentic_interrupt_before_ats: bool = True

    # --- Ops ---------------------------------------------------------------
    app_env: str = "local"
    log_json: bool = False
    metrics_window_size: int = 1000
    evaluation_cache_size: int = 256
    search_cache_size: int = 512

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
