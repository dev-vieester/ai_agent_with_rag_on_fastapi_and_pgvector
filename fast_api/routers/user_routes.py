"""
routers/user_routes.py — User profile routers.
All schemas imported from schemas.py.
"""
from __future__ import annotations
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.models import Document, User
from ..core.deps import get_db, get_current_user
from core.schmas import UserProfile, UserDocumentsResponse, DocumentMeta

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/me", response_model=UserProfile)
def get_my_profile(current_user: User = Depends(get_current_user)):
    """Return the current user's profile."""
    return UserProfile.model_validate(current_user)


@router.get("/me/documents", response_model=UserDocumentsResponse)
def get_my_documents(
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """List all documents the current user has uploaded."""
    rows = (
        db.query(
            Document.filename,
            Document.visibility,
        )
        .filter(Document.owner_id == current_user.id)
        .distinct()
        .all()
    )
    docs = [DocumentMeta(doc_source=r.filename,
                         doc_title=r.filename,
                         visibility=r.visibility) for r in rows]
    return UserDocumentsResponse(documents=docs, total=len(docs))
