from app.services.requirement_extractor import RequirementExtractor


def test_requirement_extractor_fallback_picks_requirement_lines() -> None:
    extractor = RequirementExtractor(chat_model="gpt-4o-mini", openai_api_key="", max_requirements=5)
    description = """
    We are hiring an AI Engineer.
    Must have strong Python experience.
    Experience building RAG systems in production.
    Familiar with FastAPI and Docker.
    Nice to have: frontend exposure.
    """
    requirements = extractor.extract("AI Engineer", description)
    texts = [req.text for req in requirements]
    assert any("Python" in text for text in texts)
    assert any("RAG" in text for text in texts)
