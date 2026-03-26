"""
deps.py
-------
Uten Assistant — Shared FastAPI Dependencies

All route files import their dependencies from here.
This is the single place that wires together:
    - Database session management
    - JWT authentication
    - Role enforcement
    - UtenUser construction

Why a separate deps.py?
    Without it, every router file would need to import from auth.py,
    models.py, and the database config — creating a tangled web of
    imports.  deps.py is a clean single import point:

        from deps import get_db, get_current_user, require_role, get_uten_user

    In Dart terms this is like a barrel file (index.dart) that re-exports
    everything a feature folder needs, so feature files only have one import.
"""

from __future__ import annotations

import os
from typing import Iterator
from urllib.parse import quote_plus

from fastapi import Depends
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from core.models import User
from .auth import get_current_user, require_role, build_uten_user
from core.access_control import UtenUser



load_dotenv()


def build_database_url() -> str:
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return database_url

    host = os.environ.get("DBHOSTNAME", "localhost").strip("'\"")
    port = os.environ.get("DBPORT", "5432").strip("'\"")
    user = os.environ.get("DBUSERNAME", "postgres").strip("'\"")
    password = os.environ.get("DBPASSWORD", "postgres").strip("'\"")
    database = os.environ.get("DBDATABASE", "rag_db").strip("'\"")

    return (
        f"postgresql://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{database}"
    )


DATABASE_URL = build_database_url()

engine       = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Iterator[Session]:
    """
    Yield a database session for one request, then close it.

    Every route that needs DB access declares:
        db: Session = Depends(get_db)

    FastAPI calls this before the route, passes the session in,
    and resumes the finally block after the route finishes.
    The connection is always returned to the pool — even on exceptions.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



get_current_user = get_current_user

require_role = require_role



def get_uten_user(
    current_user: User = Depends(get_current_user),
) -> UtenUser:
    """
    FastAPI dependency that returns a UtenUser for the current request.

    Chains on top of get_current_user() — JWT is verified and DB is
    queried first, then the result is converted to UtenUser.

    Usage in a route:
        @router.post("/ask")
        def ask(uten_user: UtenUser = Depends(get_uten_user)):
            response = uten.ask(question, user=uten_user)

    This removes the build_uten_user() boilerplate from every route
    that needs to call into the Uten pipeline.
    """
    return build_uten_user(current_user)


def get_uten_user_with_role(*roles: str):
    """
    Dependency factory combining role check + UtenUser construction.

    Usage:
        @router.post("/index")
        def index(uten_user: UtenUser = Depends(get_uten_user_with_role("super_admin", "manager", "member"))):
            uten.index(files, user=uten_user)

    Equivalent to get_uten_user() but also enforces the role check
    before building the UtenUser — cleaner than stacking two Depends().
    """
    def dependency(
        current_user: User = Depends(require_role(*roles)),
    ) -> UtenUser:
        return build_uten_user(current_user)

    return dependency
