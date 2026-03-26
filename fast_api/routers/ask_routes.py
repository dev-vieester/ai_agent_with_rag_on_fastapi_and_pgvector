"""
routers/ask_routes.py - Question-answering routers.
All schemas imported from schemas.py.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from core.access_control import UtenUser

from ..core.deps import get_uten_user_with_role
from core.schmas import AskRequest, AskResponse, MessageTurn

router = APIRouter(prefix="/ask", tags=["Ask"])


def get_uten(request: Request):
    return request.app.state.uten


@router.post("", response_model=AskResponse)
def ask(
    request: Request,
    req: AskRequest,
    uten_user: UtenUser = Depends(get_uten_user_with_role("super_admin", "manager", "member")),
):
    """Ask Uten a question. Answers are grounded in documents the user can see."""
    history = [{"role": t.role, "content": t.content} for t in req.history]

    response = get_uten(request).ask(
        question=req.question,
        user=uten_user,
        history=history or None,
        doc_source=req.doc_source,
        top_k=req.top_k,
    )

    typed_history = [MessageTurn(role=m["role"], content=m["content"]) for m in response.history]

    return AskResponse(
        answer=response.answer,
        sources=response.sources,
        history=typed_history,
        model=response.model,
    )


@router.post("/stream")
def ask_stream(
    request: Request,
    req: AskRequest,
    uten_user: UtenUser = Depends(get_uten_user_with_role("super_admin", "manager", "member")),
):
    """Streaming version - tokens arrive as Server-Sent Events."""
    history = [{"role": t.role, "content": t.content} for t in req.history]

    def token_generator():
        for token in get_uten(request).ask_stream(
            question=req.question,
            user=uten_user,
            history=history or None,
            doc_source=req.doc_source,
            top_k=req.top_k,
        ):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
