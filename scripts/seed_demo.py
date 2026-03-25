from __future__ import annotations

import json
from pathlib import Path

ROLE = {
    "title": "AI Engineer",
    "description": (
        "We are hiring an AI Engineer to build retrieval-backed applications using Python, FastAPI, vector search, "
        "evaluation pipelines, Docker, cloud deployment, and production APIs. Strong experience with machine learning, "
        "RAG, observability, and shipping end-to-end systems is preferred."
    ),
}

CANDIDATES = [
    {
        "filename": "akila_resume.txt",
        "text": (
            "Akila Manivannan AI Engineer. Built FastAPI services, retrieval systems, lexical search, evaluation dashboards, "
            "Dockerized ML apps, and deployed APIs to cloud environments. Experience with Python, machine learning, RAG, CI/CD, "
            "and production monitoring."
        ),
    },
    {
        "filename": "esha_resume.txt",
        "text": (
            "Esha Karlekar ML Intern. Experience with Python, machine learning coursework, experimentation, notebooks, and data analysis. "
            "Built a recommendation project and a Streamlit demo. Limited backend and cloud production work."
        ),
    },
    {
        "filename": "jaxon_resume.txt",
        "text": (
            "Jaxon Poentis Student developer. Basic Python, HTML, GitHub projects, and class assignments. Some API familiarity but no retrieval, "
            "deployment, or production ML system evidence."
        ),
    },
]


def main() -> None:
    Path('eval/demo_seed.json').write_text(json.dumps({"role": ROLE, "candidates": CANDIDATES}, indent=2), encoding='utf-8')
    print('Wrote eval/demo_seed.json')


if __name__ == '__main__':
    main()
