"""
routers/admin_routes.py — Admin routers.
All schemas imported from schemas.py.
"""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.models import Team, User
from ..core.deps import get_db, require_role
from ..core.security import require_sanitized_user_id
from core.schmas import (
    ApprovalRequest, ApprovalResponse,
    RoleChangeRequest, RoleChangeResponse,
    StatusChangeRequest, StatusChangeResponse,
    AdminUserSummary, AdminUsersResponse,
)

router  = APIRouter(prefix="/admin", tags=["Admin"])
admin_only = Depends(require_role("super_admin"))


@router.get("/users/pending", response_model=AdminUsersResponse)
def list_pending_users(
    db: Session = Depends(get_db),
    current_user: User = admin_only,
):
    """List all accounts waiting for approval in this org."""
    users = (
        db.query(User)
        .filter(User.org_id == current_user.org_id,
                User.account_status == "pending")
        .all()
    )
    return AdminUsersResponse(
        users = [AdminUserSummary.model_validate(u) for u in users],
        total = len(users),
    )


@router.post("/users/{user_id}/approval", response_model=ApprovalResponse)
def approve_or_reject(
    user_id: str,
    req:     ApprovalRequest,
    db:      Session = Depends(get_db),
    current_user: User = admin_only,
):
    """
    Approve or reject a pending account.
    Only pending accounts can be actioned — raises 400 for any other status.
    """
    user_id = require_sanitized_user_id(user_id)
    user = db.query(User).filter(
        User.id == user_id, User.org_id == current_user.org_id
    ).first()

    if not user:
        raise HTTPException(404, "User not found in your organisation.")
    if user.account_status != "pending":
        raise HTTPException(
            400,
            f"User is '{user.account_status}', not 'pending'. Nothing to action."
        )

    user.account_status = "approved" if req.action == "approve" else "rejected"
    db.commit()

    return ApprovalResponse(
        message    = f"Account {req.action}d successfully.",
        user_id    = user.id,
        user_email = user.email,
        status     = user.account_status,
    )


@router.patch("/users/{user_id}/role", response_model=RoleChangeResponse)
def change_role(
    user_id: str,
    req:     RoleChangeRequest,
    db:      Session = Depends(get_db),
    current_user: User = admin_only,
):
    """
    Change a user's role. Takes effect on their very next request
    because roles are always read from the DB, not from JWT claims.
    """
    user_id = require_sanitized_user_id(user_id)
    user = db.query(User).filter(
        User.id == user_id, User.org_id == current_user.org_id
    ).first()

    if not user:
        raise HTTPException(404, "User not found.")

    old_role = user.role
    user.role = req.new_role
    if req.new_role in {"manager", "member"}:
        team = db.query(Team).filter(
            Team.name == req.team,
            Team.org_id == current_user.org_id,
        ).first()
        if not team:
            team = Team(name=req.team, org_id=current_user.org_id)
            db.add(team)
            db.flush()
        user.team_id = team.id
    else:
        user.team_id = None
    db.commit()

    return RoleChangeResponse(
        message  = f"Role changed from '{old_role}' to '{req.new_role}'.",
        user_id  = user_id,
        old_role = old_role,
        new_role = req.new_role,
        team_id  = user.team_id,
    )


@router.patch("/users/{user_id}/status", response_model=StatusChangeResponse)
def change_account_status(
    user_id: str,
    req:     StatusChangeRequest,
    db:      Session = Depends(get_db),
    current_user: User = admin_only,
):
    """Suspend or reactivate an approved user."""
    user_id = require_sanitized_user_id(user_id)
    user = db.query(User).filter(
        User.id == user_id, User.org_id == current_user.org_id
    ).first()

    if not user:
        raise HTTPException(404, "User not found.")
    if user.id == current_user.id:
        raise HTTPException(400, "You cannot suspend your own account.")

    if req.action == "suspend":
        if user.account_status == "suspended":
            raise HTTPException(400, "Account is already suspended.")
        user.account_status = "suspended"
    else:
        if user.account_status != "suspended":
            raise HTTPException(400, "Account is not currently suspended.")
        user.account_status = "approved"

    db.commit()
    return StatusChangeResponse(
        message = f"Account {req.action}d.",
        user_id = user_id,
        status  = user.account_status,
    )


@router.get("/users", response_model=AdminUsersResponse)
def list_all_users(
    db: Session = Depends(get_db),
    current_user: User = admin_only,
):
    """List all users in the organisation."""
    users = (
        db.query(User)
        .filter(User.org_id == current_user.org_id)
        .order_by(User.created_at.desc())
        .all()
    )
    return AdminUsersResponse(
        users = [AdminUserSummary.model_validate(u) for u in users],
        total = len(users),
    )
