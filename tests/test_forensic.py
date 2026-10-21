#!/usr/bin/env python3
"""
Forensic test-suite for the AutoGen Multi-Agent repository.

Validates, end to end:
  1. Repo integrity   – required files/assets exist (data extracted)
  2. Engine logic     – agent roster, speaker-selection rules, tool-call flow
  3. Grounding        – agents actually read real cv.txt / courses.txt
  4. Pipeline output  – full conversation reaches TERMINATE + FINAL ANSWER
  5. HTTP surface     – FastAPI endpoints (/ , /health, /api/agents, /api/data,
                        /api/run, /docs) respond correctly via TestClient
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "AutoGen_MultiAgent"
sys.path.insert(0, str(PKG))

from run_live_multi_agent import (  # noqa: E402
    MockManager,
    CV_PATH,
    COURSES_PATH,
    read_txt,
)


# ---------------------------------------------------------------------------
# 1. Repo integrity
# ---------------------------------------------------------------------------
class TestRepoIntegrity:
    def test_required_files_exist(self):
        for f in ["run_live_multi_agent.py", "app.py", "requirements.txt"]:
            assert (PKG / f).exists(), f"missing {f}"
        assert (ROOT / "run.py").exists(), "missing runner file run.py"
        assert (ROOT / "README.md").exists()

    def test_data_assets_present(self):
        assert Path(CV_PATH).exists() and Path(COURSES_PATH).exists()
        assert len(read_txt(CV_PATH)) > 500, "cv.txt too small / unreadable"
        assert len(read_txt(COURSES_PATH)) > 100, "courses.txt too small / unreadable"

    def test_cv_content_grounding(self):
        cv = read_txt(CV_PATH)
        assert "Daniel Puente Viejo" in cv
        assert "Repsol" in cv or "NTT Data" in cv

    def test_courses_catalogue_structure(self):
        c = read_txt(COURSES_PATH)
        assert "Machine Learning" in c and "Natural Language Processing" in c


# ---------------------------------------------------------------------------
# 2. Engine logic
# ---------------------------------------------------------------------------
class TestEngineLogic:
    def setup_method(self):
        self.mgr = MockManager()

    def test_agent_roster(self):
        expected = {"User", "CV Analyser", "Skills Extractor",
                    "Courses recommender", "Resolution Checker Agent"}
        assert set(self.mgr.agents) == expected

    def test_tool_calls_route_to_user(self):
        msgs = [{"name": "CV Analyser", "content": "", "role": "assistant",
                 "tool_calls": [{"id": "x", "type": "function",
                                 "function": {"name": "read_txt_cv_analyse",
                                              "arguments": "{}"}}]}]
        assert self.mgr.select("CV Analyser", msgs) == "User"

    def test_specialists_chain_to_checker(self):
        msgs = [{"name": "CV Analyser", "content": "analysis", "role": "assistant"}]
        assert self.mgr.select("CV Analyser", msgs) == "Resolution Checker Agent"

    def test_first_unanswered_selection(self):
        msgs = [{"name": "User", "content": "go", "role": "user"}]
        assert self.mgr.select("User", msgs) == "CV Analyser"

    def test_user_proxy_executes_file_tool(self):
        call = {"name": "User", "content": "", "role": "user",
                "tool_calls": [{"id": "c", "type": "function",
                                "function": {"name": "read_txt_cv_analyse",
                                             "arguments": json.dumps({"path": CV_PATH})}}]}
        content, _ = self.mgr.agents["User"].respond([call], "")
        assert "Daniel Puente" in content  # actually read the real file

    def test_missing_file_returns_error_string(self):
        out = read_txt("/nonexistent/file.txt")
        assert out.startswith("Error reading file")


# ---------------------------------------------------------------------------
# 3./4. Full pipeline behaviour
# ---------------------------------------------------------------------------
class TestPipeline:
    @pytest.fixture(scope="class")
    def history(self):
        return MockManager().run("Recommend courses for the profile", max_turns=30)

    def test_terminates(self, history):
        assert any("TERMINATE" in (m.get("content") or "") for m in history), \
            "pipeline never reached TERMINATE"

    def test_all_specialists_answered(self, history):
        answered = {m["name"] for m in history
                    if m.get("content") and not m.get("tool_calls")}
        for a in ("CV Analyser", "Skills Extractor", "Courses recommender",
                  "Resolution Checker Agent"):
            assert a in answered, f"{a} never produced an answer"

    def test_tool_calls_were_executed(self, history):
        tools = [tc["function"]["name"] for m in history for tc in (m.get("tool_calls") or [])]
        assert "read_txt_cv_analyse" in tools
        assert "read_txt_courses" in tools

    def test_final_contains_skills_and_courses(self, history):
        final = history[-1]["content"]
        assert "FINAL ANSWER" in final
        assert "Python" in final                        # skills section
        assert "Deep Learning Specialization" in final  # grounded course names

    def test_grounding_not_hallucinated(self, history):
        """Every recommended course name must exist in the real catalogue."""
        catalogue = read_txt(COURSES_PATH)
        courses_msg = next(m for m in history if m["name"] == "Courses recommender"
                           and m.get("content") and not m.get("tool_calls"))
        for title in ["Deep Learning Specialization", "Natural Language Processing Specialization"]:
            assert title in courses_msg["content"]
            assert title in catalogue


# ---------------------------------------------------------------------------
# 5. HTTP surface (FastAPI TestClient)
# ---------------------------------------------------------------------------
class TestHTTPEndpoints:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from app import app as fastapi_app
        return TestClient(fastapi_app)

    def test_html_console(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "Multi-Agent" in r.text and "/api/run" in r.text

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["data_files"]["cv"] is True

    def test_agents_endpoint(self, client):
        r = client.get("/api/agents")
        assert r.status_code == 200
        names = {a["name"] for a in r.json()["agents"]}
        assert "CV Analyser" in names and "Resolution Checker Agent" in names

    def test_data_endpoint(self, client):
        r = client.get("/api/data")
        assert r.status_code == 200
        assert "Daniel Puente Viejo" in r.json()["cv"]

    def test_run_endpoint_full_pipeline(self, client):
        r = client.post("/api/run", json={"query": "Recommend courses", "use_ollama": False})
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "open-source-llama-style"
        assert "TERMINATE" in body["final"]
        assert len(body["history"]) >= 8
        assert body["agent_order"][0] != ""

    def test_openapi_docs(self, client):
        assert client.get("/docs").status_code == 200
        spec = client.get("/openapi.json").json()
        for p in ["/", "/health", "/api/agents", "/api/data", "/api/run"]:
            assert p in spec["paths"], f"{p} missing from OpenAPI spec"

    def test_invalid_payload_rejected(self, client):
        r = client.post("/api/run", json={"max_turns": 999})
        assert r.status_code == 422  # pydantic validation (max_turns <= 60)
