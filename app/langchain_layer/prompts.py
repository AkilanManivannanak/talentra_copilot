"""LangChain prompt templates and chain builders for evaluation and copilot."""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Prompt templates (raw strings; wrapped in ChatPromptTemplate when LangChain
# is available, used as f-string templates otherwise)
# ---------------------------------------------------------------------------

REQUIREMENT_EXTRACTION_PROMPT = """You are a hiring expert. Extract a structured list of requirements from the job description below.
Return a JSON array of requirement strings. Each string should be a concise, testable skill or qualification.

Job Description:
{jd_text}

Return ONLY a JSON array, e.g.: ["5+ years Python", "Experience with FastAPI", "Docker knowledge"]
"""

EVALUATION_PROMPT = """You are an expert technical recruiter performing evidence-based candidate evaluation.

Role: {role_title}
Requirement: {requirement}

Candidate resume chunks (evidence):
{evidence}

Score this candidate's evidence for the requirement on a scale of 0-3:
  0 = No evidence
  1 = Weak or indirect evidence
  2 = Clear evidence
  3 = Strong, direct evidence with specifics

Respond with JSON: {{"score": <0-3>, "justification": "<one sentence>", "evidence_quote": "<best quote or empty>"}}
"""

COPILOT_PROMPT = """You are Talentra Copilot, an evidence-grounded hiring assistant.
You answer recruiter questions using ONLY the evaluation results and evidence provided below.
Do not invent information. If evidence is insufficient, say so.

Role: {role_title}
Evaluation results (candidates ranked by score):
{evaluation_summary}

Top evidence chunks:
{evidence}

Recruiter question: {question}

Answer concisely and cite evidence when possible.
"""

INTERVIEW_QUESTION_PROMPT = """You are an expert interviewer. Generate {num_questions} targeted behavioral and technical 
interview questions for the following candidate and role.

Role: {role_title}
Role requirements: {requirements}
Candidate skills: {candidate_skills}
Candidate's strongest evidence: {top_evidence}

Format as a JSON array of question strings. Mix behavioral (STAR format) and technical questions.
"""

BIAS_AUDIT_PROMPT = """You are a hiring bias auditor. Review the following candidate evaluations for potential bias.

Candidates and scores:
{candidate_scores}

Check for:
1. Name/gender-correlated score patterns
2. Educational prestige bias (top-school preference)
3. Recency bias (penalizing career gaps)
4. Over-indexing on brand names vs actual skills

Respond with JSON: {{"bias_flags": [<list of concerns>], "severity": "none|low|medium|high", "recommendation": "<action>"}}
"""


# ---------------------------------------------------------------------------
# Chain builders (graceful degradation: LangChain → None)
# ---------------------------------------------------------------------------

def _build_chain(prompt_template: str, llm: Any) -> Any:
    """Build a LangChain chain from a prompt template string and an LLM."""
    try:
        from langchain.prompts import PromptTemplate  # type: ignore
        from langchain.chains import LLMChain  # type: ignore
        # Extract input variables from {placeholder} patterns
        import re
        vars_ = re.findall(r"\{(\w+)\}", prompt_template)
        prompt = PromptTemplate(input_variables=list(set(vars_)), template=prompt_template)
        return LLMChain(llm=llm, prompt=prompt)
    except Exception:
        return None


def get_evaluation_chain(llm: Any | None = None) -> Any | None:
    """Return a LangChain evaluation chain or None if LangChain/LLM unavailable."""
    if llm is None:
        return None
    return _build_chain(EVALUATION_PROMPT, llm)


def get_copilot_chain(llm: Any | None = None) -> Any | None:
    """Return a LangChain copilot chain or None if LangChain/LLM unavailable."""
    if llm is None:
        return None
    return _build_chain(COPILOT_PROMPT, llm)


def get_interview_chain(llm: Any | None = None) -> Any | None:
    if llm is None:
        return None
    return _build_chain(INTERVIEW_QUESTION_PROMPT, llm)


def format_prompt(template: str, **kwargs) -> str:
    """Simple string-format fallback when LangChain is not available."""
    return template.format(**kwargs)
