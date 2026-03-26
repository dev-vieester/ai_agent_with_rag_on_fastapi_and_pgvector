from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from core.models import Base, Chunk, Document, Embedding
from ..core.embedder import EmbeddedChunk


@dataclass
class SearchResult:
    id: int
    text: str
    doc_title: str
    doc_source: str
    chunk_index: int
    visibility: str
    similarity: float
    metadata: dict


class VectorStore:
    def __init__(self, database_url: str, embedding_dim: int = 384) -> None:
        self._embedding_dim = embedding_dim
        self._engine = create_engine(database_url, pool_pre_ping=True)
        self._ensure_extension()
        self._ensure_table()
        self._ensure_index()

    def _ensure_extension(self) -> None:
        with Session(self._engine) as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            session.commit()

    def _ensure_table(self) -> None:
        Base.metadata.create_all(self._engine, checkfirst=True)

    def _ensure_index(self) -> None:
        with Session(self._engine) as session:
            session.execute(text("""
                CREATE INDEX IF NOT EXISTS embeddings_embedding_idx
                ON embeddings
                USING hnsw (embedding vector_cosine_ops)
            """))
            session.commit()

    def add_chunks(
        self,
        chunks: list[EmbeddedChunk],
        visibility: str = "org",
        owner_id: Optional[str] = None,
        org_id: str = "default",
    ) -> int:
        if not chunks:
            return 0

        grouped: dict[str, list[EmbeddedChunk]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.doc_source, []).append(chunk)

        stored = 0
        with Session(self._engine) as session:
            for doc_source, doc_chunks in grouped.items():
                filename = Path(doc_source).name or doc_source
                suffix = Path(filename).suffix.lstrip(".").lower() or "txt"
                document = Document(
                    filename=filename,
                    file_type=suffix,
                    visibility=visibility,
                    owner_id=owner_id,
                    org_id=org_id,
                )
                session.add(document)
                session.flush()

                for embedded_chunk in doc_chunks:
                    chunk = Chunk(
                        document_id=document.id,
                        chunk_index=embedded_chunk.chunk_index,
                        content=embedded_chunk.text,
                    )
                    session.add(chunk)
                    session.flush()

                    session.add(Embedding(
                        chunk_id=chunk.id,
                        embedding=embedded_chunk.vector,
                    ))
                    stored += 1

            session.commit()

        return stored

    def delete_by_source(
        self,
        doc_source: str,
        org_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        visibility: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> int:
        filename = Path(doc_source).name or doc_source
        with Session(self._engine) as session:
            query = session.query(Document)
            if document_id is not None:
                query = query.filter(Document.id == document_id)
            else:
                query = query.filter(Document.filename == filename)
            if org_id is not None:
                query = query.filter(Document.org_id == org_id)
            if owner_id is not None:
                query = query.filter(Document.owner_id == owner_id)
            if visibility is not None:
                query = query.filter(Document.visibility == visibility)
            documents = query.all()
            deleted = 0
            for document in documents:
                deleted += len(document.chunks)
                session.delete(document)
            session.commit()
            return deleted

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        doc_source: Optional[str] = None,
        allowed_visibilities: Optional[list[str]] = None,
        org_id: str = "default",
    ) -> list[SearchResult]:
        conditions = ["d.org_id = :org_id"]
        params: dict[str, object] = {
            "qvec": "[" + ",".join(str(f) for f in query_vector) + "]",
            "k": top_k,
            "org_id": org_id,
        }

        if allowed_visibilities is not None and len(allowed_visibilities) > 0:
            vis_params = {f"v{i}": value for i, value in enumerate(allowed_visibilities)}
            vis_placeholders = ", ".join(f":v{i}" for i in range(len(allowed_visibilities)))
            conditions.append(f"d.visibility IN ({vis_placeholders})")
            params.update(vis_params)

        if doc_source:
            conditions.append("d.filename = :doc_source")
            params["doc_source"] = Path(doc_source).name or doc_source

        where_clause = "WHERE " + " AND ".join(conditions)

        sql = f"""
            SELECT
                c.id,
                c.content AS text,
                d.filename AS doc_title,
                d.filename AS doc_source,
                c.chunk_index,
                d.visibility,
                1 - (e.embedding <=> CAST(:qvec AS vector)) AS similarity
            FROM embeddings e
            JOIN chunks c ON c.id = e.chunk_id
            JOIN documents d ON d.id = c.document_id
            {where_clause}
            ORDER BY similarity DESC
            LIMIT :k
        """

        with Session(self._engine) as session:
            rows = session.execute(text(sql), params).fetchall()

        return [
            SearchResult(
                id=row.id,
                text=row.text,
                doc_title=row.doc_title,
                doc_source=row.doc_source,
                chunk_index=row.chunk_index,
                visibility=row.visibility,
                similarity=round(float(row.similarity), 4),
                metadata={},
            )
            for row in rows
        ]

    def count(self) -> int:
        with Session(self._engine) as session:
            result = session.execute(text("SELECT COUNT(*) FROM chunks"))
            return result.scalar()

    def __repr__(self) -> str:
        return f"VectorStore(engine={self._engine.url}, dims={self._embedding_dim})"
