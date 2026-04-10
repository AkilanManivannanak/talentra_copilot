from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.config import get_settings, Settings
from app.core.store import MetadataStore
from app.langchain_layer.vectorstore import TalentraVectorStore
from app.agents.screener import ScreenerAgent
from app.agents.ranker import RankerAgent
from app.agents.bias_auditor import BiasAuditorAgent

router = APIRouter(tags=["evaluation"])


def get_store(s: Settings = Depends(get_settings)) -> MetadataStore:
    return MetadataStore(s.metadata_path)


def get_vectorstore(s: Settings = Depends(get_settings)) -> TalentraVectorStore:
    return TalentraVectorStore(s.vectorstore_dir)


class EvaluateRequest(BaseModel):
    candidate_ids: list[str]
    top_k_per_requirement: int = 5
    run_bias_audit: bool = True


@router.post("/roles/{role_id}/evaluate", summary="Evaluate candidates against a role")
def evaluate(
    role_id: str,
    body: EvaluateRequest,
    store: MetadataStore = Depends(get_store),
    vs: TalentraVectorStore = Depends(get_vectorstore),
):
    role = store.get_role(role_id)
    if not role:
        raise HTTPException(status_code=404, detail="Role not found")

    requirements = role.get("requirements", [])

    # Fetch candidate objects
    candidates = []
    for cid in body.candidate_ids:
        c = store.get_candidate(cid)
        if c:
            candidates.append(c)

    if not candidates:
        raise HTTPException(status_code=400, detail="No valid candidates found")

    # ① Screen
    screener = ScreenerAgent()
    screened = screener.screen(
        [{"id": c["id"], "name": c["name"], "text": c["text"], "skills": c.get("skills", [])}
         for c in candidates],
        requirements=requirements,
    )
    screen_map = {s["id"]: s for s in screened}

    # ② Rank (all candidates — screener result attached separately)
    ranker = RankerAgent(vectorstore=vs)
    ranked = ranker.rank(
        role_id=role_id,
        role_title=role["title"],
        requirements=requirements,
        candidate_ids=body.candidate_ids,
        top_k=body.top_k_per_requirement,
    )

    # Merge screen results into ranked output
    for entry in ranked:
        cid = entry["candidate_id"]
        screen = screen_map.get(cid, {})
        entry["screen_pass"] = screen.get("screen_pass", True)
        entry["fail_reasons"] = screen.get("fail_reasons", [])

    # ③ Bias audit
    bias_report = None
    if body.run_bias_audit:
        auditor = BiasAuditorAgent()
        candidate_texts = {c["id"]: c["text"] for c in candidates}
        bias_report = auditor.audit(ranked, candidate_texts=candidate_texts)

    return {
        "role_id": role_id,
        "role_title": role["title"],
        "candidates": ranked,
        "bias_audit": bias_report,
        "requirements": requirements,
    }
