"""Thin serving adapters for the transport-neutral diffusion program runtime.

The runtime owns model state and batch admission. ``ProgramService`` translates
JSON-shaped requests into runtime calls and returns state/telemetry metadata;
an optional output encoder can attach an image URL, bytes, or tensor handle.
``create_fastapi_app`` is intentionally lazy so scheduler and worker processes
can use the program ABI without installing a web framework.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .program import ProgramError, ProgramRuntime, ProgramStepResult
from .scheduler import ContinuousBatchScheduler


class ProgramServiceError(RuntimeError):
    """Raised when a transport request is malformed before reaching the runtime."""


def _output_metadata(output: Any) -> dict[str, Any]:
    if output is None:
        return {"kind": "none"}
    shape = getattr(output, "shape", None)
    dtype = getattr(output, "dtype", None)
    if shape is not None:
        return {
            "kind": "tensor-like",
            "type": type(output).__name__,
            "shape": [int(value) for value in shape],
            "dtype": str(dtype) if dtype is not None else None,
        }
    images = getattr(output, "images", None)
    if images is not None:
        return {
            "kind": "batch-output",
            "type": type(output).__name__,
            "rows": len(images),
            "row_types": [type(image).__name__ for image in images],
        }
    return {"kind": "object", "type": type(output).__name__}


class ProgramService:
    """JSON-shaped service facade over a ``ProgramRuntime`` instance."""

    def __init__(
        self,
        runtime: ProgramRuntime,
        *,
        output_encoder: Callable[[Any], Any] | None = None,
        scheduler: ContinuousBatchScheduler | None = None,
    ) -> None:
        self.runtime = runtime
        self.output_encoder = output_encoder
        self.scheduler = scheduler or ContinuousBatchScheduler(runtime)

    def _result(self, result: ProgramStepResult) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "state_before": result.state_before.to_dict(),
            "state_after": result.state_after.to_dict(),
            "telemetry": dict(result.telemetry),
            "output": _output_metadata(result.output),
            "io_frame": result.io_frame.to_dict() if result.io_frame is not None else None,
        }
        if self.output_encoder is not None:
            payload["render"] = self.output_encoder(result.output)
        return payload

    @staticmethod
    def _options(request: Mapping[str, Any]) -> dict[str, Any]:
        options = request.get("options", {})
        if not isinstance(options, Mapping):
            raise ProgramServiceError("options must be an object")
        return dict(options)

    def health(self) -> dict[str, str]:
        return {"status": "ok", "service": "diffusion-program"}

    def open_session(self, request: Mapping[str, Any]) -> dict[str, Any]:
        link_id = request.get("link_id")
        if not isinstance(link_id, str) or not link_id:
            raise ProgramServiceError("link_id is required")
        resolution = request.get("resolution")
        if resolution is not None:
            if not isinstance(resolution, Sequence) or isinstance(resolution, (str, bytes)):
                raise ProgramServiceError("resolution must be [height, width]")
            resolution = tuple(int(value) for value in resolution)
        session_id = self.runtime.open_session(
            link_id,
            session_id=request.get("session_id"),
            seed=request.get("seed"),
            resolution=resolution,
            total_steps=request.get("total_steps"),
        )
        return self.session(session_id)

    def session(self, session_id: str) -> dict[str, Any]:
        session = self.runtime.get_session(session_id)
        binding = None
        if session.state.status in {"context_compiled", "running"}:
            binding = session.binding().to_dict()
        return {
            "session_id": session_id,
            "state": session.state.to_dict(),
            "binding": binding,
            "io_frame": session.last_io_frame.to_dict()
            if session.last_io_frame is not None
            else None,
        }

    def compile_context(self, session_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        prompt = request.get("prompt")
        if not isinstance(prompt, str):
            raise ProgramServiceError("prompt is required")
        embeds = self.runtime.compile_context(session_id, prompt, **self._options(request))
        return {
            "session": self.session(session_id),
            "conditioning_key": embeds.key,
        }

    def step(self, session_id: str, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
        result = self.runtime.step(session_id, **self._options(request or {}))
        return self._result(result)

    def step_batch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        session_ids = request.get("session_ids")
        if not isinstance(session_ids, Sequence) or isinstance(session_ids, (str, bytes)):
            raise ProgramServiceError("session_ids must be a non-empty array")
        ids = tuple(str(session_id) for session_id in session_ids)
        if not ids:
            raise ProgramServiceError("session_ids must be a non-empty array")
        results = self.runtime.step_batch(ids, **self._options(request))
        return {"results": [self._result(result) for result in results]}

    def enqueue(self, request: Mapping[str, Any]) -> dict[str, Any]:
        session_ids = request.get("session_ids")
        if not isinstance(session_ids, Sequence) or isinstance(session_ids, (str, bytes)):
            raise ProgramServiceError("session_ids must be an array")
        default_deadline = request.get("deadline_s")
        default_priority = request.get("priority", 0)
        for raw_session in session_ids:
            if isinstance(raw_session, Mapping):
                session_id = raw_session.get("session_id")
                if not isinstance(session_id, str) or not session_id:
                    raise ProgramServiceError("queued session objects require session_id")
                deadline_s = raw_session.get("deadline_s", default_deadline)
                priority = raw_session.get("priority", default_priority)
            else:
                session_id = str(raw_session)
                deadline_s = default_deadline
                priority = default_priority
            self.scheduler.enqueue(
                session_id,
                deadline_s=None if deadline_s is None else float(deadline_s),
                priority=int(priority),
            )
        return {
            "queued": list(self.scheduler.pending()),
            "stats": self.scheduler.stats(),
            "events": list(self.scheduler.drain_events()),
        }

    def refill(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Admit replacement rows through the same guarded queue contract."""

        return self.enqueue(request)

    def cancel(self, request: Mapping[str, Any]) -> dict[str, Any]:
        session_id = request.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ProgramServiceError("session_id is required")
        cancelled = self.scheduler.cancel(session_id, reason=str(request.get("reason", "client")))
        return {
            "session_id": session_id,
            "cancelled": cancelled,
            "queued": list(self.scheduler.pending()),
            "stats": self.scheduler.stats(),
            "events": list(self.scheduler.drain_events()),
        }

    def flush(self, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
        options = request or {}
        max_batches = options.get("max_batches")
        if max_batches is not None:
            max_batches = int(max_batches)
        worker_id = str(options.get("worker_id", "worker-0"))
        dispatch_now = options.get("now")
        if dispatch_now is not None:
            dispatch_now = float(dispatch_now)
        step_options = self._options(options)
        step_options.pop("max_batches", None)
        step_options.pop("worker_id", None)
        step_options.pop("now", None)
        results = self.scheduler.flush(
            max_batches=max_batches,
            now=dispatch_now,
            worker_id=worker_id,
            **step_options,
        )
        return {
            "results": [self._result(result) for result in results],
            "stats": self.scheduler.stats(),
            "events": list(self.scheduler.drain_events()),
        }

    def checkpoint(self, session_id: str) -> dict[str, Any]:
        return self.runtime.checkpoint(session_id).metadata()

    def pause(self, session_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        cut_step = request.get("cut_step")
        if isinstance(cut_step, bool) or not isinstance(cut_step, int):
            raise ProgramServiceError("cut_step must be an integer")
        options = self._options(request)
        options.pop("cut_step", None)
        checkpoint = self.runtime.pause(
            session_id,
            cut_step=cut_step,
            **options,
        )
        return {"session": self.session(session_id), "checkpoint": checkpoint.metadata()}

    def resume(self, session_id: str, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
        options = self._options(request or {})
        output_type = str(options.pop("output_type", "pil"))
        result = self.runtime.resume(session_id, output_type=output_type)
        return self._result(result)

    def replay_batch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        session_id = request.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ProgramServiceError("session_id is required")
        branch_ids = request.get("branch_ids")
        if not isinstance(branch_ids, Sequence) or isinstance(branch_ids, (str, bytes)):
            raise ProgramServiceError("branch_ids must be a non-empty array")
        session = self.runtime.get_session(session_id)
        checkpoint = self.runtime.checkpoint(session_id)
        options = self._options(request)
        mode = str(options.pop("mode", "exact"))
        output_type = str(options.pop("output_type", "pil"))
        if options:
            raise ProgramServiceError(
                "unsupported replay options: " + ", ".join(sorted(map(str, options)))
            )
        result = self.runtime.replay_batch(
            session.link.link_fingerprint,
            checkpoint,
            branch_ids=tuple(str(value) for value in branch_ids),
            mode=mode,
            output_type=output_type,
        )
        rows = result.rows_by_branch()
        return {
            "checkpoint": checkpoint.metadata(),
            "branch_ids": list(result.branch_ids),
            "rows": {
                branch_id: _output_metadata(value) for branch_id, value in rows.items()
            },
            "telemetry": dict(result.telemetry),
        }

    def close_session(self, session_id: str) -> dict[str, str]:
        self.runtime.close_session(session_id)
        return {"status": "closed", "session_id": session_id}


def create_fastapi_app(service: ProgramService) -> Any:
    """Create an optional HTTP app for the program service."""

    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - depends on optional server extra
        raise ProgramServiceError(
            "FastAPI is required for create_fastapi_app; install mrun[server]"
        ) from exc

    app = FastAPI(title="mrun diffusion program", version="1")

    def call(function: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return function(*args, **kwargs)
        except (ProgramError, ProgramServiceError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return service.health()

    @app.post("/v1/sessions")
    def open_session(request: dict[str, Any]) -> dict[str, Any]:
        return call(service.open_session, request)

    @app.get("/v1/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        return call(service.session, session_id)

    @app.post("/v1/sessions/{session_id}/context")
    def compile_context(session_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return call(service.compile_context, session_id, request)

    @app.post("/v1/sessions/{session_id}/step")
    def step(session_id: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        return call(service.step, session_id, request)

    @app.post("/v1/step-batch")
    def step_batch(request: dict[str, Any]) -> dict[str, Any]:
        return call(service.step_batch, request)

    @app.post("/v1/scheduler/enqueue")
    def enqueue(request: dict[str, Any]) -> dict[str, Any]:
        return call(service.enqueue, request)

    @app.post("/v1/scheduler/refill")
    def refill(request: dict[str, Any]) -> dict[str, Any]:
        return call(service.refill, request)

    @app.post("/v1/scheduler/cancel")
    def cancel(request: dict[str, Any]) -> dict[str, Any]:
        return call(service.cancel, request)

    @app.post("/v1/scheduler/flush")
    def flush(request: dict[str, Any] | None = None) -> dict[str, Any]:
        return call(service.flush, request)

    @app.post("/v1/sessions/{session_id}/checkpoint")
    def checkpoint(session_id: str) -> dict[str, Any]:
        return call(service.checkpoint, session_id)

    @app.post("/v1/sessions/{session_id}/pause")
    def pause(session_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return call(service.pause, session_id, request)

    @app.post("/v1/sessions/{session_id}/resume")
    def resume(session_id: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        return call(service.resume, session_id, request)

    @app.post("/v1/replay-batch")
    def replay_batch(request: dict[str, Any]) -> dict[str, Any]:
        return call(service.replay_batch, request)

    @app.delete("/v1/sessions/{session_id}")
    def close_session(session_id: str) -> dict[str, str]:
        return call(service.close_session, session_id)

    return app


__all__ = ["ProgramService", "ProgramServiceError", "create_fastapi_app"]
