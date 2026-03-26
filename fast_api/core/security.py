from __future__ import annotations

import os
import re
import time
from collections import defaultdict, deque
from collections.abc import Callable
from threading import Lock
from typing import Deque

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_text(value: str, *, field_name: str, max_length: int | None = None) -> str:
    cleaned = CONTROL_CHARS_RE.sub("", value).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        raise ValueError(f"{field_name} cannot be blank")
    if max_length is not None and len(cleaned) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters")
    return cleaned


def sanitize_multiline_text(value: str, *, field_name: str, max_length: int | None = None) -> str:
    cleaned = CONTROL_CHARS_RE.sub("", value).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    if not cleaned:
        raise ValueError(f"{field_name} cannot be blank")
    if max_length is not None and len(cleaned) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters")
    return cleaned


def sanitize_doc_source(value: str) -> str:
    cleaned = sanitize_text(value, field_name="doc_source", max_length=255)
    if "/" in cleaned or "\\" in cleaned or ".." in cleaned:
        raise ValueError("doc_source must be a document name, not a path")
    if not re.fullmatch(r"[\w.\- ()]+", cleaned):
        raise ValueError("doc_source contains unsupported characters")
    return cleaned


def sanitize_user_id(value: str) -> str:
    cleaned = sanitize_text(value, field_name="user_id", max_length=80)
    if not re.fullmatch(r"[A-Za-z0-9/_-]+", cleaned):
        raise ValueError("user_id contains unsupported characters")
    return cleaned


def parse_csv_env(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_cors_config() -> dict[str, object]:
    origins = parse_csv_env("CORS_ALLOW_ORIGINS", "http://localhost:3000,http://localhost:5173")
    methods = parse_csv_env("CORS_ALLOW_METHODS", "GET,POST,PUT,PATCH,DELETE,OPTIONS")
    headers = parse_csv_env("CORS_ALLOW_HEADERS", "Authorization,Content-Type")
    expose_headers = parse_csv_env("CORS_EXPOSE_HEADERS", "Retry-After,X-RateLimit-Limit,X-RateLimit-Remaining")
    allow_credentials = os.environ.get("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"

    if "*" in origins and allow_credentials:
        allow_credentials = False

    return {
        "allow_origins": origins,
        "allow_methods": methods,
        "allow_headers": headers,
        "expose_headers": expose_headers,
        "allow_credentials": allow_credentials,
    }


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_window: int, window_seconds: int) -> None:
        super().__init__(app)
        self._requests_per_window = requests_per_window
        self._window_seconds = window_seconds
        self._buckets: defaultdict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def _client_key(self, request: Request) -> str:
        forwarded_for = request.headers.get("x-forwarded-for", "")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        if request.client and request.client.host:
            return request.client.host
        return "unknown"

    async def dispatch(self, request: Request, call_next: Callable):
        if (
            self._requests_per_window <= 0
            or request.url.path == "/health"
            or request.method == "OPTIONS"
        ):
            return await call_next(request)

        now = time.time()
        client_key = self._client_key(request)
        reset_after = self._window_seconds

        with self._lock:
            bucket = self._buckets[client_key]
            while bucket and now - bucket[0] >= self._window_seconds:
                bucket.popleft()

            if len(bucket) >= self._requests_per_window:
                reset_after = max(1, int(self._window_seconds - (now - bucket[0])))
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Rate limit exceeded. Try again later."},
                    headers={
                        "Retry-After": str(reset_after),
                        "X-RateLimit-Limit": str(self._requests_per_window),
                        "X-RateLimit-Remaining": "0",
                    },
                )

            bucket.append(now)
            remaining = max(0, self._requests_per_window - len(bucket))
            if bucket:
                reset_after = max(1, int(self._window_seconds - (now - bucket[0])))

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._requests_per_window)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["Retry-After"] = str(reset_after)
        return response


async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [
        {
            "field": ".".join(str(part) for part in err["loc"] if part != "body"),
            "message": err["msg"],
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Invalid request payload.", "errors": errors},
    )


def require_sanitized_user_id(user_id: str) -> str:
    try:
        return sanitize_user_id(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
