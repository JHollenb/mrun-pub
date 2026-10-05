"""Stdlib HTTP client. Configure MRUN_URL and MRUN_TOKEN explicitly."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_URLS = (
    "http://127.0.0.1:9025",
)


class ApiError(RuntimeError):
    def __init__(self, status: int, msg: str) -> None:
        super().__init__(f"HTTP {status}: {msg}")
        self.status = status


class Api:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        token: str | None = None,
        agent_token: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        env_url = os.environ.get("MRUN_URL")
        self._candidates = [base_url] if base_url else (
            [env_url] if env_url else list(DEFAULT_URLS)
        )
        self.base_url: str | None = None
        self.token = token if token is not None else os.environ.get("MRUN_TOKEN")
        # Never infer this from the general client token. Only the host-agent process
        # receives the independently configured credential.
        self.agent_token = agent_token
        self.timeout_s = timeout_s

    # -- transport ------------------------------------------------------------
    def _resolve(self) -> str:
        if self.base_url:
            return self.base_url
        last: Exception | None = None
        for cand in self._candidates:
            try:
                req = urllib.request.Request(cand.rstrip("/") + "/healthz")
                with urllib.request.urlopen(req, timeout=3.0):
                    self.base_url = cand.rstrip("/")
                    return self.base_url
            except Exception as exc:  # noqa: BLE001
                last = exc
        raise ConnectionError(f"no mrun scheduler reachable at {self._candidates}: {last}")

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        raw_body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> tuple[int, bytes, dict[str, str]]:
        url = self._resolve() + path
        data = None
        request_headers = {str(key): str(value) for key, value in (headers or {}).items()}
        if self.token:
            request_headers["X-Mrun-Token"] = self.token
        if self.agent_token:
            request_headers["X-MRun-Agent-Token"] = self.agent_token
        if json_body is not None:
            data = json.dumps(json_body).encode()
            request_headers["Content-Type"] = "application/json"
        elif raw_body is not None:
            data = raw_body
            request_headers["Content-Type"] = "application/octet-stream"
        req = urllib.request.Request(url, data=data, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout_s or self.timeout_s) as resp:
                return resp.status, resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), dict(exc.headers or {})

    def json(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        status, body, _ = self.request(
            method, path, json_body=json_body, headers=headers, timeout_s=timeout_s
        )
        if status == 204:
            return None
        if status >= 400:
            raise ApiError(status, body.decode(errors="replace")[:500])
        return json.loads(body) if body else None

    def retry_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        headers: dict[str, str] | None = None,
        max_wait_s: float = 300.0,
    ) -> Any:
        """Retry with backoff on connection errors (scheduler down / off-LAN)."""
        delay = 1.0
        start = time.time()
        while True:
            try:
                return self.json(method, path, json_body=json_body, headers=headers)
            except (ConnectionError, OSError):
                self.base_url = None  # re-resolve (LAN <-> tailscale moves)
                if time.time() - start > max_wait_s:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
