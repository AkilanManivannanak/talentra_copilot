"""InterviewerAgent: generate tailored interview questions for a candidate."""
from __future__ import annotations

import random
from typing import Any

from .base import BaseAgent


# Fallback question templates when no LLM is available
_BEHAVIORAL_TEMPLATES = [
    "Tell me about a time you {skill_context}. What was the outcome?",
    "Describe a challenging project where you had to {skill_context}. How did you approach it?",
    "Walk me through a situation where {skill_context} was critical to success.",
    "Give an example of how you've used {skill} in a production or project setting.",
]

_TECHNICAL_TEMPLATES = [
    "How would you design a system that {requirement_context}?",
    "What are the trade-offs between {skill} and its main alternatives?",
    "How do you approach testing and validating {skill}-based components?",
    "Describe your experience with {skill}. What's the most complex use case you've handled?",
    "If you were onboarding a junior engineer on {skill}, what would you emphasize first?",
]


class InterviewerAgent(BaseAgent):
    """
    Generates a mix of behavioral (STAR-format) and technical interview questions
    tailored to the candidate's top evidence and role requirements.
    """

    def __init__(self, llm: Any | None = None):
        super().__init__(llm)

    def generate_questions(
        self,
        role_title: str,
        requirements: list[str],
        candidate_skills: list[str],
        top_evidence: list[dict],
        num_questions: int = 8,
    ) -> list[dict]:
        """
        Returns list of question dicts: {type, skill_focus, question}.
        Tries LLM first; falls back to template generation.
        """
        if self._llm:
            questions = self._llm_questions(
                role_title, requirements, candidate_skills, top_evidence, num_questions
            )
            if questions:
                return questions

        return self._template_questions(requirements, candidate_skills, num_questions)

    def _llm_questions(
        self,
        role_title: str,
        requirements: list[str],
        candidate_skills: list[str],
        top_evidence: list[dict],
        num_questions: int,
    ) -> list[dict]:
        from app.langchain_layer.prompts import INTERVIEW_QUESTION_PROMPT, format_prompt

        evidence_str = self._format_evidence(top_evidence, max_chars=600)
        prompt = format_prompt(
            INTERVIEW_QUESTION_PROMPT,
            role_title=role_title,
            requirements="; ".join(requirements[:8]),
            candidate_skills=", ".join(candidate_skills[:12]),
            top_evidence=evidence_str,
            num_questions=num_questions,
        )
        reply = self._call_llm(prompt, max_tokens=800)
        parsed = self._parse_json(reply)

        if isinstance(parsed, list):
            return [
                {"type": "mixed", "skill_focus": "", "question": str(q)}
                for q in parsed
            ]
        return []

    def _template_questions(
        self,
        requirements: list[str],
        candidate_skills: list[str],
        num_questions: int,
    ) -> list[dict]:
        """Generate questions from templates when LLM unavailable."""
        questions = []
        focus_items = (candidate_skills + requirements)[:num_questions]

        behavioral_n = max(1, num_questions // 2)
        technical_n = num_questions - behavioral_n

        for i, skill in enumerate(focus_items[:behavioral_n]):
            template = _BEHAVIORAL_TEMPLATES[i % len(_BEHAVIORAL_TEMPLATES)]
            q = template.format(
                skill=skill,
                skill_context=f"applied {skill} in a real-world scenario",
            )
            questions.append({"type": "behavioral", "skill_focus": skill, "question": q})

        for i, skill in enumerate(focus_items[:technical_n]):
            template = _TECHNICAL_TEMPLATES[i % len(_TECHNICAL_TEMPLATES)]
            q = template.format(
                skill=skill,
                requirement_context=f"heavily relies on {skill}",
            )
            questions.append({"type": "technical", "skill_focus": skill, "question": q})

        return questions[:num_questions]
