"""Metric implementations, pinned against hand-computed values.

An eval harness whose metrics are themselves unverified is not evidence of anything.
"""
import json
import math
from pathlib import Path

from eval.harness import kendall_tau, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k([3, 2, 1, 0], 4) == 1.0


def test_ndcg_is_below_one_for_an_inverted_ranking():
    assert ndcg_at_k([0, 1, 2, 3], 4) < 0.6


def test_ndcg_matches_a_hand_computed_value():
    # gains 2^rel - 1 = [3, 0, 1]; dcg = 3/log2(2) + 0/log2(3) + 1/log2(4) = 3 + 0 + 0.5
    # ideal = [3, 1, 0]; idcg = 3 + 1/log2(3) + 0 = 3 + 0.63093
    dcg = 3 + 0 + 1 / math.log2(4)
    idcg = 3 + 1 / math.log2(3)
    assert ndcg_at_k([2, 0, 1], 3) == round(dcg / idcg, 10) or abs(ndcg_at_k([2, 0, 1], 3) - dcg / idcg) < 1e-9


def test_precision_counts_only_relevance_two_and_above():
    assert precision_at_k([3, 2, 1], 3) == 2 / 3
    assert precision_at_k([1, 1, 1], 3) == 0.0


def test_recall_is_relative_to_all_relevant_items():
    assert recall_at_k([3, 0, 0, 2, 2], 2) == 1 / 3


def test_reciprocal_rank_finds_the_first_relevant_position():
    assert reciprocal_rank([0, 0, 2]) == 1 / 3
    assert reciprocal_rank([0, 0, 1]) == 0.0


def test_kendall_tau_is_one_for_identical_orderings():
    assert kendall_tau([3, 2, 1], [3, 2, 1]) == 1.0


def test_kendall_tau_is_minus_one_for_reversed_orderings():
    assert kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0


def test_label_set_is_well_formed():
    labels = json.loads((Path(__file__).resolve().parents[1] / "eval" / "labels.json").read_text())
    candidate_ids = {c["id"] for c in labels["candidates"]}
    assert labels["counts"]["judgements"] == len(labels["judgements"])
    assert labels["counts"]["candidates"] == len(candidate_ids) == 30
    for judgement in labels["judgements"]:
        assert judgement["candidate_id"] in candidate_ids
        assert judgement["relevance"] in (0, 1, 2, 3)
    # The limitation must stay attached to the data it applies to.
    assert "synthetic" in labels["limitation"].lower()


def test_labels_are_not_derivable_from_the_visible_prose_alone():
    """The judgements come from the structured profile; the retriever only sees prose.

    Two ways this could be circular, both checked: the prose could leak the grade, or
    every candidate at a given grade could share identical wording, which would turn the
    eval into a string-equality test rather than a retrieval test.
    """
    labels = json.loads((Path(__file__).resolve().parents[1] / "eval" / "labels.json").read_text())

    for candidate in labels["candidates"]:
        text = candidate["text"].lower()
        # Structural markers only: 'relevance' is legitimate resume vocabulary for a
        # candidate whose evidence is about ranking evaluation, and banning it would be
        # banning the domain rather than the leak.
        for leak in ("grade:", "profile:", "capability:", "judgement:", "label:", "rel="):
            assert leak not in text, f"{candidate['id']} leaks its label into the prose"
        assert not any(marker in text for marker in ("strong)", "clear)", "weak)", "none)")), \
            f"{candidate['id']} appears to annotate its own grade"

    phrasings: dict[tuple[str, int], set[str]] = {}
    for candidate in labels["candidates"]:
        bullets = [line for line in candidate["text"].splitlines() if line.startswith("- ")]
        for (capability, grade), bullet in zip(candidate["_profile"].items(), bullets, strict=False):
            phrasings.setdefault((capability, grade), set()).add(bullet)

    varied = [key for key, values in phrasings.items() if len(values) > 1]
    assert varied, "no capability/grade pair has more than one phrasing"
    # Most heavily-used pairs should have more than one wording.
    assert len(varied) >= 5, f"only {len(varied)} capability/grade pairs are varied"
