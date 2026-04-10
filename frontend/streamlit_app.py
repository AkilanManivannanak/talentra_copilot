from __future__ import annotations

import requests
import streamlit as st

st.set_page_config(page_title="Talentra Copilot", layout="wide")
backend_url = st.sidebar.text_input("Backend URL", value="http://localhost:8000")

st.title("Talentra Copilot")
st.caption("Evidence-grounded candidate evaluation, recruiter copilot, and ATS intelligence")

role_tab, candidate_tab, evaluate_tab, copilot_tab, ats_tab, ops_tab = st.tabs([
    "Create Role",
    "Upload Candidates",
    "Evaluate",
    "Copilot Q&A",
    "ATS Workflow",
    "Ops",
])


def get_roles() -> list[dict]:
    try:
        resp = requests.get(f"{backend_url}/roles", timeout=20)
        resp.raise_for_status()
        return resp.json().get("roles", [])
    except Exception:
        return []


def get_candidates() -> list[dict]:
    try:
        resp = requests.get(f"{backend_url}/candidates", timeout=20)
        resp.raise_for_status()
        return resp.json().get("candidates", [])
    except Exception:
        return []


with role_tab:
    st.subheader("Create a role")
    title = st.text_input("Role title")
    description = st.text_area("Job description", height=250)
    jd_file = st.file_uploader("Or upload a JD document", type=["pdf", "docx", "txt", "md"])

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Create role from text"):
            if not title or not description:
                st.warning("Provide a title and job description.")
            else:
                try:
                    resp = requests.post(
                        f"{backend_url}/roles/text",
                        json={"title": title, "description": description},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.success(f"Created role: {data['title']}")
                    with st.expander("Extracted requirements"):
                        for req in data["requirements"]:
                            st.write(f"- {req['text']}")
                except requests.RequestException as exc:
                    st.error(str(exc))
    with col_b:
        if st.button("Create role from file"):
            if not jd_file:
                st.warning("Upload a JD file first.")
            else:
                try:
                    files = {"file": (jd_file.name, jd_file.getvalue(), jd_file.type)}
                    data = {"title": title}
                    resp = requests.post(f"{backend_url}/roles/upload", data=data, files=files, timeout=120)
                    resp.raise_for_status()
                    payload = resp.json()
                    st.success(f"Created role: {payload['title']}")
                    with st.expander("Extracted requirements"):
                        for req in payload["requirements"]:
                            st.write(f"- {req['text']}")
                except requests.RequestException as exc:
                    st.error(str(exc))

with candidate_tab:
    st.subheader("Upload candidate resumes")
    resumes = st.file_uploader("Resumes", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if st.button("Ingest candidates"):
        if not resumes:
            st.warning("Upload at least one resume.")
        else:
            files = [("resumes", (file.name, file.getvalue(), file.type)) for file in resumes]
            try:
                resp = requests.post(f"{backend_url}/candidates/upload", files=files, timeout=180)
                resp.raise_for_status()
                payload = resp.json()
                for candidate in payload.get("candidates", []):
                    st.success(f"{candidate['name']} ingested from {candidate['filename']} ({candidate['chunk_count']} chunks)")
                for failed in payload.get("failed", []):
                    st.warning(f"Skipped {failed['filename']}: {failed['error']}")
            except requests.RequestException as exc:
                detail = None
                if exc.response is not None:
                    try:
                        payload = exc.response.json()
                        detail = payload.get("detail")
                    except Exception:
                        detail = None
                if isinstance(detail, dict):
                    st.error(detail.get("message", str(exc)))
                    for failed in detail.get("failed", []):
                        st.warning(f"Skipped {failed['filename']}: {failed['error']}")
                else:
                    st.error(str(exc))

with evaluate_tab:
    st.subheader("Evaluate candidates against a role")
    roles = get_roles()
    candidates = get_candidates()
    role_options = {f"{role['title']} ({role['id']})": role["id"] for role in roles}
    candidate_options = {f"{candidate['name']} ({candidate['id']})": candidate["id"] for candidate in candidates}

    selected_role_label = st.selectbox("Role", options=list(role_options.keys())) if role_options else None
    selected_candidates = st.multiselect("Candidates", options=list(candidate_options.keys())) if candidate_options else []

    if st.button("Run evaluation"):
        if not selected_role_label:
            st.warning("Create a role first.")
        else:
            role_id = role_options[selected_role_label]
            candidate_ids = [candidate_options[label] for label in selected_candidates]
            try:
                resp = requests.post(
                    f"{backend_url}/roles/{role_id}/evaluate",
                    json={"candidate_ids": candidate_ids, "top_k_per_requirement": 2},
                    timeout=180,
                )
                resp.raise_for_status()
                payload = resp.json()
                for idx, candidate in enumerate(payload["candidates"], start=1):
                    with st.expander(f"#{idx} {candidate['candidate_name']} — score {candidate['overall_score']:.2f}"):
                        st.write(candidate["summary"])
                        st.write(
                            f"Matched requirements: {candidate['matched_requirements']} | Missing requirements: {candidate['missing_requirements']}"
                        )
                        for assessment in candidate["assessments"]:
                            status = "✅" if assessment["covered"] else "⚠️"
                            st.markdown(f"{status} **{assessment['requirement']['text']}** — {assessment['score']:.2f}")
                            for evidence in assessment["evidence"]:
                                st.caption(f"{evidence['entity_name']} · {evidence['filename']} · {evidence['score']:.2f}")
                                st.write(evidence["snippet"])
            except requests.RequestException as exc:
                st.error(str(exc))

with copilot_tab:
    st.subheader("Ask grounded follow-up questions")
    roles = get_roles()
    candidates = get_candidates()
    role_options = {f"{role['title']} ({role['id']})": role["id"] for role in roles}
    candidate_options = {f"{candidate['name']} ({candidate['id']})": candidate["id"] for candidate in candidates}
    selected_role_label = st.selectbox("Role for Q&A", options=list(role_options.keys()), key="qa-role") if role_options else None
    qa_candidates = st.multiselect("Limit to candidates", options=list(candidate_options.keys()), key="qa-candidates") if candidate_options else []
    question = st.text_area(
        "Question",
        height=120,
        placeholder="Compare the selected candidates against the role requirements and explain the ranking.",
    )

    if st.button("Ask Copilot"):
        if not selected_role_label or not question:
            st.warning("Select a role and ask a question.")
        else:
            role_id = role_options[selected_role_label]
            candidate_ids = [candidate_options[label] for label in qa_candidates]
            try:
                resp = requests.post(
                    f"{backend_url}/copilot/query",
                    json={"question": question, "role_id": role_id, "candidate_ids": candidate_ids, "top_k": 8},
                    timeout=180,
                )
                resp.raise_for_status()
                payload = resp.json()
                st.write(payload["answer"])
                with st.expander("Citations"):
                    for citation in payload["citations"]:
                        st.markdown(f"**{citation['entity_name']}** · {citation['filename']} · {citation['score']:.2f}")
                        st.write(citation["snippet"])
                with st.expander("Reasoning trace"):
                    for item in payload["reasoning_trace"]:
                        st.write(f"- {item}")
            except requests.RequestException as exc:
                st.error(str(exc))

with ats_tab:
    st.subheader("ATS workflow and recruiter ops")
    roles = get_roles()
    candidates = get_candidates()
    role_options = {f"{role['title']} ({role['id']})": role["id"] for role in roles}
    candidate_options = {f"{candidate['name']} ({candidate['id']})": candidate["id"] for candidate in candidates}
    selected_role_label = st.selectbox("Role for ATS dashboard", options=list(role_options.keys()), key="ats-role") if role_options else None

    if selected_role_label:
        role_id = role_options[selected_role_label]
        try:
            resp = requests.get(f"{backend_url}/ats/roles/{role_id}/dashboard", timeout=60)
            resp.raise_for_status()
            dashboard = resp.json()
            cols = st.columns(len(dashboard["stage_counts"]))
            for col, item in zip(cols, dashboard["stage_counts"]):
                col.metric(item["stage"], item["count"])
            rows = [
                {
                    "Candidate": item["candidate_name"],
                    "Stage": item["stage"],
                    "Shortlisted": "Yes" if item["shortlisted"] else "No",
                    "Score": item["latest_score"],
                    "Matched": item["matched_requirements"],
                    "Missing": item["missing_requirements"],
                    "Notes": item["notes_count"],
                }
                for item in dashboard["candidates"]
            ]
            if rows:
                st.dataframe(rows, use_container_width=True)
            with st.expander("Shortlist view"):
                for item in dashboard["shortlist"]:
                    st.markdown(f"**{item['candidate_name']}** · stage: {item['stage']} · score: {item['latest_score']}")
                    if item.get("summary"):
                        st.write(item["summary"])
        except requests.RequestException as exc:
            st.error(str(exc))

    candidate_label = st.selectbox("Candidate", options=list(candidate_options.keys()), key="ats-candidate") if candidate_options else None
    stage_options = ["Applied", "Screening", "Shortlisted", "Interview", "Final", "Rejected"]
    col1, col2 = st.columns(2)
    with col1:
        new_stage = st.selectbox("Update stage", options=stage_options, key="ats-stage")
        if st.button("Save stage") and candidate_label:
            candidate_id = candidate_options[candidate_label]
            try:
                resp = requests.post(f"{backend_url}/ats/candidates/{candidate_id}/stage", json={"stage": new_stage}, timeout=30)
                resp.raise_for_status()
                st.success("Candidate stage updated.")
            except requests.RequestException as exc:
                st.error(str(exc))
    with col2:
        shortlist_value = st.checkbox("Shortlist candidate", value=False, key="ats-shortlist")
        if st.button("Save shortlist flag") and candidate_label:
            candidate_id = candidate_options[candidate_label]
            try:
                resp = requests.post(
                    f"{backend_url}/ats/candidates/{candidate_id}/shortlist",
                    json={"shortlisted": shortlist_value},
                    timeout=30,
                )
                resp.raise_for_status()
                st.success("Shortlist flag updated.")
            except requests.RequestException as exc:
                st.error(str(exc))

    note_text = st.text_area("Recruiter note", height=120, key="ats-note")
    if st.button("Add recruiter note") and candidate_label:
        candidate_id = candidate_options[candidate_label]
        try:
            resp = requests.post(f"{backend_url}/ats/candidates/{candidate_id}/notes", json={"text": note_text}, timeout=30)
            resp.raise_for_status()
            st.success("Recruiter note saved.")
        except requests.RequestException as exc:
            st.error(str(exc))

    if candidate_label:
        candidate_id = candidate_options[candidate_label]
        try:
            resp = requests.get(f"{backend_url}/ats/candidates/{candidate_id}/notes", timeout=30)
            resp.raise_for_status()
            notes = resp.json().get("notes", [])
            with st.expander("Candidate notes", expanded=False):
                if not notes:
                    st.write("No notes yet.")
                for note in notes:
                    st.caption(note["created_at"])
                    st.write(note["text"])
        except requests.RequestException:
            pass

with ops_tab:
    st.subheader("System metrics and build metadata")
    left, right = st.columns(2)
    try:
        metrics_resp = requests.get(f"{backend_url}/ops/metrics", timeout=20)
        metrics_resp.raise_for_status()
        metrics = metrics_resp.json()
        build_resp = requests.get(f"{backend_url}/ops/build", timeout=20)
        build_resp.raise_for_status()
        build = build_resp.json()

        with left:
            st.metric("Window size", metrics["window_size"])
            st.metric("p50 latency (ms)", metrics["p50_ms"])
            st.metric("p95 latency (ms)", metrics["p95_ms"])
            st.metric("Error rate", metrics["error_rate"])
        with right:
            st.json(build)
        if metrics.get("routes"):
            st.dataframe(metrics["routes"], use_container_width=True)
    except requests.RequestException as exc:
        st.error(str(exc))
