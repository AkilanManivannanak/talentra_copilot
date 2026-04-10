"""
Generate fine-tuning training data from:
  1. eval/demo_seed.json fixtures
  2. ATS recruiter feedback (accept/reject decisions)
  3. Copilot Q&A interaction logs

Output: JSONL in chat format for SFT (supervised fine-tuning).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are Talentra Copilot, an evidence-grounded hiring assistant. "
    "Answer recruiter questions using only the evidence provided. "
    "Cite specific resume text when making claims. Be concise and factual."
)


def _make_sft_example(
    user_message: str,
    assistant_message: str,
    system: str = SYSTEM_PROMPT,
) -> dict:
    """Create a single chat-format SFT example."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    }


def generate_from_seed(seed_path: str | Path) -> list[dict]:
    """
    Read eval/demo_seed.json and generate SFT examples from the
    ground-truth ranking and candidate texts.
    """
    seed_path = Path(seed_path)
    if not seed_path.exists():
        return []

    with open(seed_path) as f:
        seed = json.load(f)

    role = seed.get("role", {})
    candidates = seed.get("candidates", [])
    if not candidates:
        return []

    examples = []
    role_title = role.get("title", "Software Engineer")
    role_desc = role.get("description", "")

    # Ranking question examples
    for i, winner in enumerate(candidates):
        for loser in candidates[i + 1:]:
            w_name = _filename_to_name(winner["filename"])
            l_name = _filename_to_name(loser["filename"])
            user = (
                f"We're hiring for: {role_title}.\n"
                f"Job description: {role_desc[:300]}\n\n"
                f"Candidate A ({w_name}): {winner['text']}\n"
                f"Candidate B ({l_name}): {loser['text']}\n\n"
                f"Who is the stronger candidate and why?"
            )
            assistant = (
                f"{w_name} is the stronger candidate. "
                f"They demonstrate more direct experience relevant to this role. "
                f"Evidence: \"{winner['text'][:200]}\". "
                f"{l_name} shows less direct evidence: \"{loser['text'][:100]}\"."
            )
            examples.append(_make_sft_example(user, assistant))

    # Skill-specific evidence questions
    for candidate in candidates:
        name = _filename_to_name(candidate["filename"])
        skills_mentioned = re.findall(r"[A-Z][a-zA-Z]+(?:API|ML|AI|CI|CD|SQL|DB)?", candidate["text"])
        for skill in skills_mentioned[:3]:
            user = f"What evidence does {name} have for {skill}?"
            assistant = (
                f"Based on their resume, {name} demonstrates {skill} experience. "
                f"Relevant text: \"{candidate['text'][:250]}\"."
            )
            examples.append(_make_sft_example(user, assistant))

    return examples


def generate_from_ats_feedback(feedback_records: list[dict]) -> list[dict]:
    """
    Convert ATS recruiter decisions into preference-style SFT examples.
    feedback_records: [{candidate_id, candidate_name, decision, notes, role_title}]
    """
    examples = []
    shortlisted = [r for r in feedback_records if r.get("decision") == "shortlisted"]
    rejected = [r for r in feedback_records if r.get("decision") == "rejected"]

    for acc in shortlisted:
        for rej in rejected:
            if acc.get("role_title") != rej.get("role_title"):
                continue
            user = (
                f"Role: {acc.get('role_title', '')}\n"
                f"Should we move forward with {acc['candidate_name']} or {rej['candidate_name']}?"
            )
            assistant = (
                f"Move forward with {acc['candidate_name']}. "
                + (f"Recruiter notes: {acc.get('notes', '')}." if acc.get("notes") else "")
            )
            examples.append(_make_sft_example(user, assistant))

    return examples


def generate_training_data(
    seed_path: str | Path = "eval/demo_seed.json",
    feedback_records: list[dict] | None = None,
) -> list[dict]:
    """Combine all training data sources."""
    examples = generate_from_seed(seed_path)
    if feedback_records:
        examples.extend(generate_from_ats_feedback(feedback_records))
    return examples


def export_jsonl(examples: list[dict], output_path: str | Path) -> int:
    """Write examples to a JSONL file. Returns count written."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return len(examples)


def _filename_to_name(filename: str) -> str:
    name = re.sub(r"[_\-]resume.*$", "", filename, flags=re.IGNORECASE)
    name = re.sub(r"\.(txt|pdf|docx|md)$", "", name, flags=re.IGNORECASE)
    return name.replace("_", " ").title().strip()
