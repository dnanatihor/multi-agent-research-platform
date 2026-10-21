"""Orchestration, review, parallelism, and blackboard tests."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from research_platform.api import create_app
from research_platform.config import Settings
from research_platform.service import expect_conflict


def settings(**overrides: object) -> Settings:
    base = dict(app_env="dev", llm_mode="scripted", rate_limit_per_minute=100)
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


QUESTION = {"question": "How do transformers use attention during training?"}


def test_review_integrates_every_agent() -> None:
    with TestClient(create_app(settings())) as client:
        started = client.post("/api/v1/reports", json=QUESTION)
        assert started.status_code == 202
        body = started.json()
        assert "## Research" in body["draft"]
        assert "## Analysis" in body["draft"]
        assert "## Narrative" in body["draft"]
        assert all(amount > 0 for amount in body["perAgent"].values())
        assert abs(sum(body["perAgent"].values()) - body["costUsd"]) < 1e-8
        finished = client.post(
            f"/api/v1/reports/{body['reportId']}/review",
            json={"decision": "approve"},
        )
        answer = finished.json()["answer"]
        assert finished.json()["status"] == "completed"
        assert "## Research" in answer and "## Analysis" in answer and "## Narrative" in answer
        assert finished.json()["costUsd"] < 2
        trace = client.get(f"/api/v1/reports/{body['reportId']}/trace").json()
        assert any(event.get("step") == "parallel" for event in trace["trajectory"])
        cost = client.get(f"/api/v1/reports/{body['reportId']}/cost").json()
        assert set(cost["perAgent"]) == {"researcher", "analyst", "writer"}
        assert all(amount > 0 for amount in cost["perAgent"].values())
        assert abs(sum(cost["perAgent"].values()) - cost["totalUsd"]) < 1e-8


def test_human_notes_are_applied() -> None:
    with TestClient(create_app(settings())) as client:
        started = client.post("/api/v1/reports", json=QUESTION)
        report_id = started.json()["reportId"]
        revised = client.post(
            f"/api/v1/reports/{report_id}/review",
            json={"decision": "revise", "notes": "remove speculation"},
        )
        answer = revised.json()
        assert "remove speculation" in answer["answer"]
        assert "## Research" in answer["answer"]
        assert all(amount > 0 for amount in answer["perAgent"].values())
        assert abs(sum(answer["perAgent"].values()) - answer["costUsd"]) < 1e-8


def test_failed_agent_still_contributes() -> None:
    with TestClient(create_app(settings(fail_agent="analyst"))) as client:
        started = client.post("/api/v1/reports", json=QUESTION)
        finished = client.post(
            f"/api/v1/reports/{started.json()['reportId']}/review",
            json={"decision": "approve"},
        )
        answer = finished.json()["answer"]
        assert "Degraded contribution" in answer
        assert "## Research" in answer and "## Narrative" in answer


def test_parallelism_beats_sequential() -> None:
    service = create_app(settings(specialist_delay_s=0.2)).state.service
    with TestClient(create_app(settings(specialist_delay_s=0.2))) as client:
        started = client.post("/api/v1/reports", json=QUESTION)
        assert started.json()["parallelSeconds"] < 0.35
    parallel = asyncio.run(service.time_pair(0.2, parallel=True))
    sequential = asyncio.run(service.time_pair(0.2, parallel=False))
    assert parallel < sequential


def test_blackboard_conflict() -> None:
    asyncio.run(expect_conflict())
