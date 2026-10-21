"""HTTP edge for the multi-agent research platform."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from research_platform.config import Settings
from research_platform.platform import AppError, CorrelationMiddleware, Metrics, RateLimiter, error_response
from research_platform.service import PlatformService, Report

INDEX = Path(__file__).resolve().parent / "static" / "index.html"


class ReportIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class ReviewIn(BaseModel):
    decision: str
    notes: str = ""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    if settings.app_env == "prod":
        _ = settings.resolved_api_key
    service = PlatformService(settings)
    metrics = Metrics(settings.service_name)
    app = FastAPI(title="Multi-Agent Research Platform", version="0.1.0")
    app.state.service = service
    app.state.metrics = metrics

    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> Response:
        return error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(_request: Request, _exc: RequestValidationError) -> Response:
        return error_response(AppError(422, "Invalid request", "The request body is invalid.", "invalid-request"))

    @app.post("/api/v1/reports")
    async def start(body: ReportIn) -> JSONResponse:
        report = await service.start(body.question)
        metrics.add_cost(report.cost_usd)
        code = 202 if report.status == "awaiting_review" else 200
        return JSONResponse(_body(report), status_code=code)

    @app.post("/api/v1/reports/{report_id}/review")
    async def review(report_id: str, body: ReviewIn) -> dict[str, object]:
        return _body(await service.review(report_id, body.decision, body.notes))

    @app.get("/api/v1/reports/{report_id}")
    def get_report(report_id: str) -> dict[str, object]:
        return _body(service.get(report_id))

    @app.get("/api/v1/reports/{report_id}/trace")
    def trace(report_id: str) -> dict[str, object]:
        report = service.get(report_id)
        return {"trajectory": report.trajectory, "parallelSeconds": report.parallel_seconds}

    @app.get("/api/v1/reports/{report_id}/cost")
    def cost(report_id: str) -> dict[str, object]:
        report = service.get(report_id)
        return {"totalUsd": report.cost_usd, "perAgent": report.costs}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": settings.service_name}

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        return Response(content=metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(INDEX)

    app.add_middleware(
        CorrelationMiddleware,
        service=settings.service_name,
        environment=settings.app_env,
        metrics=metrics,
        limiter=RateLimiter(settings.rate_limit_per_minute),
    )
    return app


def _body(report: Report) -> dict[str, object]:
    return {
        "reportId": report.report_id,
        "status": report.status,
        "draft": report.draft,
        "answer": report.answer,
        "trajectory": report.trajectory,
        "costUsd": report.cost_usd,
        "parallelSeconds": report.parallel_seconds,
        "interrupt": report.interrupt,
        "correlationId": report.correlation_id,
        "perAgent": report.costs,
    }
