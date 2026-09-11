"""The agentic path, end to end through HTTP.

Every one of these would have failed before the remediation: the graph module was imported
by nothing, the router that referenced the agents was never registered in main.py, and the
router itself read Settings fields that do not exist.
"""


def test_agentic_route_is_registered(app_client):
    paths = app_client.get("/openapi.json").json()["paths"]
    assert "/roles/{role_id}/evaluate/agentic" in paths
    assert "/agentic/runs/{run_id}/resume" in paths


def test_graph_executes_every_node_in_order(seeded):
    client, role, ids = seeded["client"], seeded["role"], seeded["ids"]
    response = client.post(f"/roles/{role['id']}/evaluate/agentic", json={
        "candidate_ids": ids,
        "question": "Why is the top candidate ranked above the others?",
        "run_bias_audit": True,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["nodes_executed"][:3] == ["screen", "evaluate", "audit_bias"]
    assert body["route_taken"] == "copilot_qa"
    assert body["errors"] == []
    assert body["evaluation"] is not None
    assert body["bias_audit"] is not None
    assert body["answer"]["faithfulness"] is not None


def test_router_sends_a_skill_question_to_evidence_search(seeded):
    client, role, ids = seeded["client"], seeded["role"], seeded["ids"]
    body = client.post(f"/roles/{role['id']}/evaluate/agentic", json={
        "candidate_ids": ids,
        "question": "What Docker experience is in these resumes?",
    }).json()
    assert body["route_taken"] == "evidence_search"


def test_screening_excludes_failures_from_the_ranking(seeded):
    client, role, ids = seeded["client"], seeded["role"], seeded["ids"]
    body = client.post(f"/roles/{role['id']}/evaluate/agentic", json={"candidate_ids": ids}).json()
    passed = {item["candidate_id"] for item in body["screening"] if item["screen_pass"]}
    ranked = {item["candidate_id"] for item in body["evaluation"]["candidates"]}
    assert ranked <= passed, "candidates that failed a must-have must not appear in the ranking"


def test_ats_write_is_blocked_until_a_human_approves(seeded):
    """The human-in-the-loop guarantee. Approval is not a log line."""
    client, role, ids = seeded["client"], seeded["role"], seeded["ids"]
    target = ids[0]
    before = {c["id"]: c["stage"] for c in client.get("/candidates").json()["candidates"]}

    run = client.post(f"/roles/{role['id']}/evaluate/agentic", json={
        "candidate_ids": ids,
        "ats_action": {"candidate_id": target, "stage": "Interview", "note": "advanced by agent"},
    }).json()

    assert run["interrupted_before"] == "ats_update"
    assert run["pending_ats_action"]["candidate_id"] == target
    during = {c["id"]: c["stage"] for c in client.get("/candidates").json()["candidates"]}
    assert during[target] == before[target], "the ATS must not be written before approval"

    resumed = client.post(f"/agentic/runs/{run['run_id']}/resume",
                          json={"run_id": run["run_id"], "approve": True}).json()
    assert resumed["ats_committed"] is True
    after = {c["id"]: c["stage"] for c in client.get("/candidates").json()["candidates"]}
    assert after[target] == "Interview"


def test_declining_leaves_the_ats_untouched(seeded):
    client, role, ids = seeded["client"], seeded["role"], seeded["ids"]
    target = ids[0]
    run = client.post(f"/roles/{role['id']}/evaluate/agentic", json={
        "candidate_ids": ids,
        "ats_action": {"candidate_id": target, "stage": "Final"},
    }).json()

    resumed = client.post(f"/agentic/runs/{run['run_id']}/resume",
                          json={"run_id": run["run_id"], "approve": False}).json()
    assert resumed["ats_committed"] is False
    stages = {c["id"]: c["stage"] for c in client.get("/candidates").json()["candidates"]}
    assert stages[target] != "Final"


def test_resume_of_an_unknown_run_is_reported_not_committed(seeded):
    client = seeded["client"]
    body = client.post("/agentic/runs/deadbeef/resume",
                       json={"run_id": "deadbeef", "approve": True}).json()
    assert body["ats_committed"] is False
    assert "No run is parked" in body["message"]


def test_interview_kit_is_reachable_and_typed(seeded):
    client, role, ids = seeded["client"], seeded["role"], seeded["ids"]
    body = client.post(
        f"/roles/{role['id']}/candidates/{ids[0]}/interview-kit?num_questions=4"
    ).json()
    assert 1 <= len(body["questions"]) <= 4
    assert all(q["type"] in {"behavioral", "technical", "mixed"} for q in body["questions"])
