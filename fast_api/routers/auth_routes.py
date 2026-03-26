"""
routers/auth_routes.py — Authentication routers.
All schemas imported from schemas.py.
"""
from __future__ import annotations
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from core.models import Organisation, Team, User
from ..core.auth import hash_password, verify_password, create_access_token
from ..core.deps import get_db, get_current_user
from ..core.security import sanitize_text
from core.schmas import RegisterRequest, RegisterResponse, TokenResponse, UserProfile

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _short_code(value: str, *, max_parts: int = 3) -> str:
    cleaned = "".join(ch if ch.isalnum() else " " for ch in value).strip()
    parts = [part for part in cleaned.split() if part]
    if not parts:
        return "GEN"

    if len(parts) == 1:
        token = parts[0][:max_parts]
    else:
        token = "".join(part[0] for part in parts[:max_parts])

    return token.upper()


def _build_user_id(org_id: str, team_name: str | None, role: str, db: Session) -> str:
    org_code = _short_code(org_id)
    team_code = _short_code(team_name) if team_name else _short_code(role)

    prefix = f"XED/{org_code}/{team_code}/"
    count = db.query(User).filter(User.id.like(f"{prefix}%")).count()
    return f"{prefix}{count + 1:03d}"


@router.post("/register", response_model=RegisterResponse, status_code=201)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """
    Create a new user account.
    Account starts as 'pending' — a super_admin must approve before the user
    can access any Uten features.
    """
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(400, "An account with this email already exists.")

    org_id = req.organization.value
    org = db.query(Organisation).filter(Organisation.id == org_id).first()
    if not org:
        org = Organisation(id=org_id, name=org_id)
        db.add(org)
        db.flush()

    team_id = None
    team_name = None
    if req.role in {"manager", "member"}:
        team = db.query(Team).filter(
            Team.name == req.team,
            Team.org_id == org_id,
        ).first()
        if not team:
            team = Team(name=req.team, org_id=org_id)
            db.add(team)
            db.flush()
        team_id = team.id
        team_name = team.name

    user_id = _build_user_id(org_id=org_id, team_name=team_name, role=req.role, db=db)

    user = User(
        id              = user_id,
        email           = req.email,
        hashed_password = hash_password(req.password),
        role            = req.role,
        team_id         = team_id,
        org_id          = org_id,
        account_status  = "pending",
        is_active       = True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return RegisterResponse(
        message        = "Account created. Awaiting administrator approval.",
        user_id        = user.id,
        account_status = user.account_status,
    )


@router.post("/login", response_model=TokenResponse)
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db:   Session = Depends(get_db),
):
    """
    Log in with email and password. Returns a JWT + current account_status.
    Pending/rejected users receive a token but are blocked on all other routers.
    """
    username = sanitize_text(form.username, field_name="email", max_length=300)
    password = form.password.strip()
    user = db.query(User).filter(User.email == username).first()

    if not user:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Incorrect email or password",
            headers     = {"WWW-Authenticate": "Bearer"},
        )

    ok = verify_password(password, user.hashed_password)

    if not ok:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Incorrect email or password",
            headers     = {"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(403, "Account is disabled.")

    user.last_login = datetime.utcnow()
    db.commit()

    return TokenResponse(
        access_token   = create_access_token(user_id=user.id),
        account_status = user.account_status,
    )


@router.get("/me", response_model=UserProfile)
def get_me(current_user: User = Depends(get_current_user)):
    """Return the current authenticated user's profile."""
    return UserProfile.model_validate(current_user)
