"""
main.py
-------
Uten Assistant — FastAPI Application

This file does three things only:
    1. Creates the FastAPI app
    2. Creates the Uten pipeline instance (shared across all requests)
    3. Registers all routers

No business logic lives here.
No route handlers live here.
No dependency functions live here.

Run with:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from core.models import Base
from fast_api.core.deps import DATABASE_URL, engine
from fast_api.core.security import (
    RateLimitMiddleware,
    build_cors_config,
    validation_exception_handler,
)
from rag.pipeline.ingest import Uten

from fast_api.routers.auth_routes  import router as auth_router
from fast_api.routers.user_routes  import router as user_router
from fast_api.routers.admin_routes import router as admin_router
from fast_api.routers.document_route import router as document_router
from fast_api.routers.ask_routes   import router as ask_router


load_dotenv(override=True)


def _get_anthropic_api_key() -> str | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if api_key is None:
        return None

    normalized = api_key.strip()
    return normalized or None


def _mask_key(api_key: str | None) -> str:
    if not api_key:
        return "missing"
    if len(api_key) <= 8:
        return "loaded"
    return f"{api_key[:6]}...{api_key[-4:]}"



@asynccontextmanager
async def lifespan(app: FastAPI):
    anthropic_api_key = _get_anthropic_api_key()
    print(f"[Uten API] Anthropic API key: {_mask_key(anthropic_api_key)}")
    app.state.uten = Uten(
        database_url      = DATABASE_URL,
        anthropic_api_key = anthropic_api_key,
    )
    Base.metadata.create_all(bind=engine)
    print("[Uten API] Database tables ready.")
    print("[Uten API] Server is up.")
    yield
    print("[Uten API] Shutting down.")



app = FastAPI(
    title       = "Uten Assistant API",
    description = "RAG-powered knowledge assistant with RBAC",
    version     = "1.0.0",
    lifespan    = lifespan,
)

cors_config = build_cors_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_config["allow_origins"],
    allow_credentials=cors_config["allow_credentials"],
    allow_methods=cors_config["allow_methods"],
    allow_headers=cors_config["allow_headers"],
    expose_headers=cors_config["expose_headers"],
)
app.add_middleware(
    RateLimitMiddleware,
    requests_per_window=int(os.environ.get("RATE_LIMIT_REQUESTS", "60")),
    window_seconds=int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60")),
)
app.add_exception_handler(RequestValidationError, validation_exception_handler)



app.include_router(auth_router, prefix="/api")
app.include_router(user_router, prefix="/api")
app.include_router(admin_router, prefix="/api")
app.include_router(document_router, prefix="/api")
app.include_router(ask_router, prefix="/api")



@app.get("/health", tags=["Health"])
def health():
    """Returns 200 if server is running. No auth required."""
    return {"status": "ok", "uten": app.state.uten.health()}
