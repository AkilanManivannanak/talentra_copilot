from app.models.schemas import Evidence
from app.services.faithfulness import FaithfulnessScorer, split_sentences


def evidence(snippet: str, name: str = "Alice") -> Evidence:
    return Evidence(document_id="d1", filename="a.pdf", entity_id="c1",
                    entity_name=name, snippet=snippet, score=0.9)


def test_supported_sentence_scores_high():
    scorer = FaithfulnessScorer()
    report = scorer.score(
        answer="Alice built FastAPI services and deployed them with Docker.",
        citations=[evidence("Built FastAPI services and deployed them with Docker on AWS.")],
    )
    assert report.faithfulness == 1.0
    assert report.sentences_supported == report.sentences_total == 1


def test_unsupported_claim_is_caught():
    """The point of the metric: a citation list next to an answer proves nothing."""
    scorer = FaithfulnessScorer()
    report = scorer.score(
        answer="Alice led a team of forty engineers in Antarctica.",
        citations=[evidence("Built FastAPI services and deployed them with Docker on AWS.")],
    )
    assert report.faithfulness == 0.0
    assert report.unsupported == ["Alice led a team of forty engineers in Antarctica."]


def test_mixed_answer_scores_between():
    scorer = FaithfulnessScorer()
    report = scorer.score(
        answer=("Alice built FastAPI services and deployed them with Docker. "
                "She also holds a private pilot licence."),
        citations=[evidence("Built FastAPI services and deployed them with Docker on AWS.")],
    )
    assert 0.0 < report.faithfulness < 1.0
    assert report.sentences_total == 2


def test_no_citations_is_zero_not_one():
    scorer = FaithfulnessScorer()
    report = scorer.score(answer="Alice is the strongest candidate.", citations=[])
    assert report.faithfulness == 0.0


def test_sentences_without_claims_are_excluded_not_failed():
    scorer = FaithfulnessScorer()
    report = scorer.score(answer="It is. And so.", citations=[evidence("Anything at all.")])
    assert report.sentences_total == 0
    assert report.faithfulness == 1.0


def test_split_sentences_handles_abbreviation_free_text():
    assert split_sentences("One thing. Two things! Three?") == ["One thing.", "Two things!", "Three?"]
