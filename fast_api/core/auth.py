"""
auth.py
-------
Uten Assistant — FastAPI Authentication & Dependency Injection

Handles:
    - Password hashing (bcrypt via passlib)
    - JWT creation and verification (python-jose)
    - FastAPI dependency functions that protect routers
    - Building a UtenUser from a verified database User

How it fits in the request lifecycle:
    1. Client POSTs email+password to /auth/login
    2. auth.py verifies password against bcrypt hash in DB
    3. auth.py creates a JWT containing user_id (not role — role comes from DB)
    4. Client stores JWT and sends it as Authorization: Bearer <token>
    5. On every protected request, get_current_user() dependency:
           a. Extracts token from header
           b. Verifies JWT signature
           c. Reads user_id from token payload
           d. Queries users table to get current role, team, org
           e. Returns a User ORM object
    6. require_role() dependency receives that User and checks their role
    7. build_uten_user() converts User ORM → UtenUser for the Uten pipeline

Why read role from DB on every request (not from JWT)?
    If an admin changes a user's role (e.g. demotes a manager),
    the change takes effect immediately on the next request.
    If we stored the role in the JWT, the old role would be valid
    until the token expires (could be hours or days).
    Security-critical permissions must always be verified against
    the database — never trusted from client-supplied tokens.

Dependencies:
    pip install fastapi python-jose[cryptography] passlib[bcrypt]
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from core.models import User
from core.access_control import UtenUser, Role



SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "change-me-in-production")

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("TOKEN_EXPIRE_MINUTES", "60"))



pwd_context = CryptContext(
    schemes=["pbkdf2_sha256", "bcrypt"],
    deprecated="auto",
)


def hash_password(plain_password: str) -> str:
    """
    Hash a plain-text password using the configured primary scheme.

    Call this when creating a user or changing their password.
    Store only the returned hash — never store the plain password.

    Example:
        hashed = hash_password("supersecret123")
        user = User(email="vic@xedla.com", hashed_password=hashed, ...)
    """
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Check whether a plain password matches a stored bcrypt hash.

    bcrypt.verify() re-hashes the plain password with the stored salt
    and compares — it never reverses the hash.

    Returns True if they match, False otherwise.
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return False



def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a signed JWT token for a user.

    Args:
        user_id:       The user's database ID — stored in the "sub" claim.
                       "sub" (subject) is the standard JWT field for user identity.
        expires_delta: How long until the token expires.
                       Defaults to ACCESS_TOKEN_EXPIRE_MINUTES.

    Returns:
        A signed JWT string — return this to the client on login.

    What goes in the token:
        {
          "sub": "user-uuid-here",   ← user_id only, NOT the role
          "exp": 1234567890          ← expiry timestamp
        }

    What does NOT go in the token:
        role, team_id, permissions — these are read from the DB on every
        request so changes take effect immediately.
    """
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {
        "sub": user_id,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> str:
    """
    Verify a JWT signature and extract the user_id.

    Args:
        token: the raw JWT string from the Authorization header.

    Returns:
        user_id string extracted from the "sub" claim.

    Raises:
        HTTPException 401 if the token is invalid, expired, or tampered.
    """
    credentials_exception = HTTPException(
        status_code = status.HTTP_401_UNAUTHORIZED,
        detail      = "Invalid or expired token",
        headers     = {"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        return user_id
    except JWTError:
        raise credentials_exception



oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def _get_db_dependency():
    """
    Late-bind get_db() to avoid an auth.py <-> deps.py import cycle.
    """
    from .deps import get_db

    yield from get_db()



def get_current_user(
    token: str = Depends(oauth2_scheme),
    db:    Session = Depends(_get_db_dependency),
) -> User:
    """
    FastAPI dependency — decodes JWT and returns the User from the database.

    Usage in a route:
        @app.get("/me")
        def get_me(current_user: User = Depends(get_current_user)):
            return current_user

    What happens:
        1. FastAPI extracts the Bearer token from the Authorization header.
        2. decode_token() verifies the signature and expiry.
        3. We query the database for the user with that ID.
        4. We check the user is still active (account not suspended).
        5. Return the full User ORM object.

    Why query the DB (not just trust the JWT)?
        Role changes and account suspensions take effect immediately.
        A suspended user's token will still pass JWT verification
        (the signature is valid) but this DB check catches them.
    """
    user_id = decode_token(token)

    user = db.query(User).filter(User.id == user_id).first()

    if user is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "User not found",
        )

    if not user.is_active:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = "Account is disabled",
        )

    if user.account_status == "pending":
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = "Account is pending approval by an administrator. "
                          "You will be notified once approved.",
        )

    if user.account_status == "rejected":
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = "Your account registration was not approved. "
                          "Contact your organisation administrator.",
        )

    if user.account_status == "suspended":
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = "Account has been suspended. "
                          "Contact your organisation administrator.",
        )

    return user


def require_role(*allowed_roles: str):
    """
    FastAPI dependency factory — enforces role-based access on a route.

    This is a factory function — it returns a dependency function
    configured for the specific roles you allow.

    Usage:
        @app.post("/admin/promote")
        def promote_user(user: User = Depends(require_role("super_admin"))):
            ...

        @app.post("/index")
        def index_docs(user: User = Depends(require_role("super_admin", "manager", "member"))):
            ...

    Why a factory and not a fixed dependency?
        Different routers have different role requirements.
        The factory pattern lets you declare the allowed roles
        right in the route definition — clear and readable.
        In Dart this is like a guard class you configure per-route.
    """
    def dependency(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code = status.HTTP_403_FORBIDDEN,
                detail      = (
                    f"Role '{current_user.role}' is not permitted for this action. "
                    f"Required: {list(allowed_roles)}"
                ),
            )
        return current_user

    return dependency


def build_uten_user(db_user: User) -> UtenUser:
    """
    Convert a SQLAlchemy User ORM object into a UtenUser for the pipeline.

    This is the bridge between your FastAPI auth layer and your Uten RAG layer.
    The User ORM object is heavy — it has lazy-loaded relationships, a
    database session attached, etc.  UtenUser is a simple dataclass with
    just the fields the Uten pipeline needs.

    Why separate them?
        Separation of concerns.  The Uten pipeline should not know about
        SQLAlchemy sessions, ORM relationships, or password hashes.
        UtenUser carries only what's needed for access decisions.

    Args:
        db_user: the User ORM object returned by get_current_user()

    Returns:
        UtenUser ready to pass into uten.index() or uten.ask()
    """
    return UtenUser(
        user_id = db_user.id,
        role    = Role(db_user.role),    # str → Role enum
        team_id = db_user.team_id,       # may be None for super_admin
        org_id  = db_user.org_id,
    )
