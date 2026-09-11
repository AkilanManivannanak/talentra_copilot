from app.services.requirement_extractor import RequirementExtractor


def extractor(max_requirements: int = 8) -> RequirementExtractor:
    return RequirementExtractor(max_requirements=max_requirements)


PARAGRAPH_JD = (
    "We are hiring an AI Engineer to build retrieval-backed applications using Python, "
    "FastAPI, vector search, evaluation pipelines, Docker, cloud deployment, caching, and "
    "production APIs. Strong experience with machine learning, RAG, observability, and "
    "shipping end-to-end systems is preferred."
)

BULLET_JD = """AI Engineer, Talent Platform

Requirements:
- Must have 3+ years of professional Python experience.
- Required: experience building retrieval-augmented generation systems in production.
- Strong knowledge of FastAPI, Docker, and cloud deployment.
Nice to have: exposure to LangGraph.
"""


def test_paragraph_jd_is_not_echoed_back_as_one_requirement():
    """The original bug: a single-paragraph JD produced the whole 300-char paragraph as
    requirement #1, which then matched every candidate weakly and flattened the ranking."""
    requirements = extractor().extract("AI Engineer", PARAGRAPH_JD)
    assert requirements
    for requirement in requirements:
        assert len(requirement.text) <= 140, f"requirement too long: {requirement.text[:80]}..."
        assert requirement.text.strip() != PARAGRAPH_JD.strip()


def test_bullets_are_extracted_as_separate_requirements():
    requirements = extractor().extract("AI Engineer, Talent Platform", BULLET_JD)
    texts = " || ".join(r.text for r in requirements)
    assert "Python" in texts
    assert "retrieval-augmented" in texts or "RAG" in texts
    assert "FastAPI" in texts


def test_must_have_flag_is_set_once_at_extraction():
    requirements = extractor().extract("AI Engineer, Talent Platform", BULLET_JD)
    must_haves = [r for r in requirements if r.must_have]
    assert len(must_haves) >= 2
    assert all(r.weight > 1.0 for r in must_haves)


def test_nice_to_have_is_downweighted_and_not_a_must_have():
    requirements = extractor().extract("AI Engineer, Talent Platform", BULLET_JD)
    nice = [r for r in requirements if "LangGraph" in r.text]
    assert nice, "the nice-to-have line should still be extracted"
    assert not nice[0].must_have
    assert nice[0].weight < 1.0


def test_job_title_is_not_treated_as_a_requirement():
    requirements = extractor().extract("AI Engineer, Talent Platform", BULLET_JD)
    assert not any(r.text.strip().lower() == "ai engineer, talent platform" for r in requirements)


def test_every_requirement_gets_an_id():
    """Requirement.id used to be passed by the extractor but absent from the schema, so
    pydantic's extra="ignore" silently discarded it on every requirement in the system."""
    requirements = extractor().extract("AI Engineer", BULLET_JD)
    ids = [r.id for r in requirements]
    assert all(ids), "ids must be populated"
    assert len(set(ids)) == len(ids), "ids must be unique"


def test_keyword_expansion_does_not_pad_a_good_result():
    requirements = extractor().extract("AI Engineer, Talent Platform", BULLET_JD)
    stubs = [r for r in requirements if r.text.startswith("Experience with ")]
    assert not stubs, f"segmentation found real requirements; stubs should not appear: {stubs}"


def test_max_requirements_is_respected_and_keeps_must_haves():
    requirements = extractor(max_requirements=2).extract("AI Engineer, Talent Platform", BULLET_JD)
    assert len(requirements) == 2
    assert all(r.must_have for r in requirements), "a cut must never drop a hard filter first"
