from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from core.access_control import UtenUser
from core.models import Document
from core.schmas import DeleteResponse, IndexResponse, TextIndexRequest
from ..core.deps import get_db, get_uten_user_with_role
from ..core.security import sanitize_doc_source, sanitize_multiline_text, sanitize_text

router = APIRouter(prefix="/documents", tags=["Documents"])
ALLOWED_UPLOAD_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".csv", ".json", ".jsonl", ".html", ".htm"}
MAX_UPLOAD_SIZE_BYTES = int(os.environ.get("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024)))


def get_uten(request: Request):
    return request.app.state.uten


@router.post("/index", response_model=IndexResponse)
async def index_documents(
    request: Request,
    files: list[UploadFile] | None = File(None),
    text: str | None = Form(None),
    title: str | None = Form(None),
    uten_user: UtenUser = Depends(get_uten_user_with_role("super_admin", "manager", "member")),
):
    """Upload files or paste text, then ingest, chunk, embed, and store."""
    uten = get_uten(request)

    try:
        if files:
            with tempfile.TemporaryDirectory() as tmpdir:
                paths: list[str] = []
                for upload in files:
                    if not upload.filename:
                        raise HTTPException(400, "Each uploaded file must have a filename.")
                    safe_name = sanitize_doc_source(Path(upload.filename).name)
                    extension = Path(safe_name).suffix.lower()
                    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
                        raise HTTPException(400, f"Unsupported file type '{extension}'.")
                    content = await upload.read()
                    if not content:
                        raise HTTPException(400, f"Uploaded file '{safe_name}' is empty.")
                    if len(content) > MAX_UPLOAD_SIZE_BYTES:
                        raise HTTPException(
                            413,
                            f"Uploaded file '{safe_name}' exceeds the {MAX_UPLOAD_SIZE_BYTES} byte limit.",
                        )
                    dest = Path(tmpdir) / safe_name
                    dest.write_bytes(content)
                    paths.append(str(dest))
                report = uten.index(paths, user=uten_user)
        elif text and text.strip():
            sanitized_text = sanitize_multiline_text(text, field_name="text")
            sanitized_title = sanitize_text(
                title if title and title.strip() else "Pasted Text",
                field_name="title",
                max_length=200,
            )
            report = uten.index_text(
                text=sanitized_text,
                title=sanitized_title,
                user=uten_user,
            )
        else:
            raise HTTPException(400, "Provide at least one file or a text payload.")
    except PermissionError as exc:
        raise HTTPException(403, str(exc))

    return IndexResponse(
        indexed=report.succeeded,
        failed=report.failed,
        total_chunks=report.total_chunks,
        errors=report.errors,
    )


@router.post("/index/text", response_model=IndexResponse)
async def index_text_document(
    request: Request,
    payload: TextIndexRequest,
    uten_user: UtenUser = Depends(get_uten_user_with_role("super_admin", "manager", "member")),
):
    """Index pasted text through the ingest pipeline using a JSON body."""
    try:
        report = get_uten(request).index_text(
            text=payload.text,
            title=payload.title,
            user=uten_user,
        )
    except PermissionError as exc:
        raise HTTPException(403, str(exc))

    return IndexResponse(
        indexed=report.succeeded,
        failed=report.failed,
        total_chunks=report.total_chunks,
        errors=report.errors,
    )


@router.delete("/{doc_source:path}", response_model=DeleteResponse)
def delete_document(
    request: Request,
    doc_source: str,
    uten_user: UtenUser = Depends(get_uten_user_with_role("super_admin", "manager", "member")),
    db: Session = Depends(get_db),
):
    """Delete one document and its related chunks and embeddings."""
    doc_source = sanitize_doc_source(Path(doc_source).name)
    document = db.query(Document).filter(
        Document.filename == doc_source,
        Document.org_id == uten_user.org_id,
    ).first()

    if not document:
        raise HTTPException(404, "Document not found.")

    if not (
        document.owner_id == uten_user.user_id
        or uten_user.role.value == "super_admin"
    ):
        raise HTTPException(403, "You do not have permission to delete this document.")

    deleted = get_uten(request).delete(
        doc_source,
        org_id=uten_user.org_id,
        owner_id=document.owner_id,
        visibility=document.visibility,
        document_id=document.id,
    )
    return DeleteResponse(
        message="Document deleted.",
        doc_source=doc_source,
        deleted_chunks=deleted,
    )
