from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Iterator, Optional

from core.access_control import UtenUser, allowed_visibilities, assert_can_upload, make_visibility

from ..core.document_ingestion import IngestionResult, ingest_many, ingest_text
from ..core.embedder import Embedder, EmbeddedChunk
from ..core.generator import Generator, UtenResponse
from ..storage.vector_store import SearchResult, VectorStore


@dataclass
class IndexReport:
    total: int
    succeeded: int
    failed: int
    total_chunks: int
    errors: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Indexed {self.succeeded}/{self.total} files -> {self.total_chunks} chunks stored",
        ]
        if self.errors:
            lines.append("Failures:")
            for path, err in self.errors.items():
                lines.append(f"  {path}: {err}")
        return "\n".join(lines)


class Uten:
    def __init__(
        self,
        database_url: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        embed_model: str = "all-MiniLM-L6-v2",
        generation_model: str = "claude-sonnet-4-6",
        embedding_dim: int = 384,
        top_k: int = 5,
        min_similarity: float = 0.3,
        max_workers: int = 4,
    ) -> None:
        db_url = database_url or os.environ.get("UTEN_DATABASE_URL")
        api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not db_url:
            raise ValueError(
                "database_url is required. Pass it directly or set the "
                "UTEN_DATABASE_URL environment variable."
            )

        self._top_k = top_k
        self._min_similarity = min_similarity
        self._max_workers = max_workers

        print("[Uten] Loading embedder...")
        self._embedder = Embedder(model_name=embed_model)

        print("[Uten] Connecting to vector store...")
        self._store = VectorStore(
            database_url=db_url,
            embedding_dim=embedding_dim,
        )

        print("[Uten] Initialising generator...")
        self._generator = Generator(
            api_key=api_key,
            model=generation_model,
        )

        print("[Uten] Ready.\n")

    def _resolve_access_scope(
        self,
        user: Optional[UtenUser],
    ) -> tuple[str, Optional[str], str]:
        if user is not None:
            assert_can_upload(user)
            return make_visibility(user), user.user_id, user.org_id
        return "org", None, "default"

    def _store_ingestion_results(
        self,
        *,
        ingest_results: list[IngestionResult],
        total_inputs: int,
        input_label: str,
        visibility: str,
        owner_id: Optional[str],
        org_id: str,
        replace: bool,
    ) -> IndexReport:
        successes = [result for result in ingest_results if result.success]
        failures = [result for result in ingest_results if not result.success]
        errors = {result.file_path: result.error for result in failures}

        if not successes:
            print(f"[Uten] No {input_label} could be ingested.")
            return IndexReport(
                total=total_inputs,
                succeeded=0,
                failed=len(failures),
                total_chunks=0,
                errors=errors,
            )

        if replace:
            for result in successes:
                deleted = self._store.delete_by_source(
                    result.document.source,
                    org_id=org_id,
                    owner_id=owner_id,
                    visibility=visibility,
                )
                if deleted:
                    print(f"[Uten] Replaced {deleted} existing chunks for {result.file_path}")

        print(f"[Uten] Embedding {sum(len(result.document.chunks) for result in successes)} chunks...")
        documents = [result.document for result in successes]
        embedded_per_doc = self._embedder.embed_many(documents)
        all_chunks: list[EmbeddedChunk] = [
            embedded_chunk
            for doc_chunks in embedded_per_doc
            for embedded_chunk in doc_chunks
        ]

        print(f"[Uten] Storing {len(all_chunks)} chunks (visibility={visibility})...")
        stored = self._store.add_chunks(
            all_chunks,
            visibility=visibility,
            owner_id=owner_id,
            org_id=org_id,
        )

        report = IndexReport(
            total=total_inputs,
            succeeded=len(successes),
            failed=len(failures),
            total_chunks=stored,
            errors=errors,
        )
        print(f"[Uten] {report}")
        return report

    def index(
        self,
        file_paths: list[str],
        user: Optional[UtenUser] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        replace: bool = True,
    ) -> IndexReport:
        visibility, owner_id, org_id = self._resolve_access_scope(user)

        print(
            f"[Uten] Ingesting {len(file_paths)} file(s) "
            f"(visibility={visibility}, owner={owner_id})..."
        )

        ingest_results = ingest_many(
            file_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_workers=self._max_workers,
        )
        return self._store_ingestion_results(
            ingest_results=ingest_results,
            total_inputs=len(file_paths),
            input_label="files",
            visibility=visibility,
            owner_id=owner_id,
            org_id=org_id,
            replace=replace,
        )

    def index_text(
        self,
        text: str,
        title: str = "Pasted Text",
        user: Optional[UtenUser] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        replace: bool = True,
    ) -> IndexReport:
        visibility, owner_id, org_id = self._resolve_access_scope(user)

        print(
            f"[Uten] Ingesting text input '{title}' "
            f"(visibility={visibility}, owner={owner_id})..."
        )

        document = ingest_text(
            text=text,
            title=title,
            source=title,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        result = IngestionResult(file_path=title, document=document)
        return self._store_ingestion_results(
            ingest_results=[result],
            total_inputs=1,
            input_label="text inputs",
            visibility=visibility,
            owner_id=owner_id,
            org_id=org_id,
            replace=replace,
        )

    def ask(
        self,
        question: str,
        user: Optional[UtenUser] = None,
        history: Optional[list[dict]] = None,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        doc_source: Optional[str] = None,
    ) -> UtenResponse:
        k = top_k or self._top_k
        sim = min_similarity or self._min_similarity
        vis_filter = allowed_visibilities(user) if user else None
        org_id = user.org_id if user else "default"

        q_vector: list[float] = self._embedder.embed_query(question)
        context_chunks: list[SearchResult] = self._store.search(
            query_vector=q_vector,
            top_k=k,
            doc_source=doc_source,
            allowed_visibilities=vis_filter,
            org_id=org_id,
        )

        return self._generator.ask(
            question=question,
            context_chunks=context_chunks,
            history=history,
            min_similarity=sim,
        )

    def ask_stream(
        self,
        question: str,
        user: Optional[UtenUser] = None,
        history: Optional[list[dict]] = None,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        doc_source: Optional[str] = None,
    ) -> Iterator[str]:
        k = top_k or self._top_k
        sim = min_similarity or self._min_similarity
        vis_filter = allowed_visibilities(user) if user else None
        org_id = user.org_id if user else "default"

        q_vector = self._embedder.embed_query(question)
        context_chunks = self._store.search(
            q_vector,
            top_k=k,
            doc_source=doc_source,
            allowed_visibilities=vis_filter,
            org_id=org_id,
        )

        yield from self._generator.ask_stream(
            question=question,
            context_chunks=context_chunks,
            history=history,
            min_similarity=sim,
        )

    def delete(
        self,
        doc_source: str,
        org_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        visibility: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> int:
        deleted = self._store.delete_by_source(
            doc_source,
            org_id=org_id,
            owner_id=owner_id,
            visibility=visibility,
            document_id=document_id,
        )
        print(f"[Uten] Deleted {deleted} chunks for '{doc_source}'")
        return deleted

    def stats(self) -> dict:
        total_chunks = self._store.count()
        return {
            "total_chunks": total_chunks,
            "embed_model": repr(self._embedder),
            "generator_model": repr(self._generator),
        }

    def health(self) -> dict:
        try:
            total_chunks = self._store.count()
            db_status = "ok"
        except Exception as exc:
            total_chunks = -1
            db_status = f"error: {exc}"

        return {
            "status": "ok" if db_status == "ok" else "degraded",
            "db_status": db_status,
            "total_chunks": total_chunks,
            "embed_model": getattr(self._embedder, "_model_name", repr(self._embedder)),
            "embedding_dims": getattr(self._embedder, "dimensions", None),
            "generator_model": getattr(self._generator, "_model", repr(self._generator)),
            "top_k": self._top_k,
            "min_similarity": self._min_similarity,
            "max_workers": self._max_workers,
        }

    def __repr__(self) -> str:
        return (
            f"Uten(embedder={self._embedder}, "
            f"store={self._store}, "
            f"generator={self._generator})"
        )


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uten",
        description="Uten Assistant - RAG pipeline CLI",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("UTEN_DATABASE_URL"),
        help="PostgreSQL connection string (or set UTEN_DATABASE_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key (or set ANTHROPIC_API_KEY)",
    )

    subparsers = parser.add_subparsers(dest="command")

    idx = subparsers.add_parser("index", help="Index one or more files")
    idx.add_argument("files", nargs="+", help="File paths to index")
    idx.add_argument("--no-replace", action="store_true", help="Append instead of replacing existing chunks")

    ask = subparsers.add_parser("ask", help="Ask a single question")
    ask.add_argument("question", help="The question to ask")
    ask.add_argument("--stream", action="store_true", help="Stream the response token by token")
    ask.add_argument("--source", default=None, help="Limit to a specific document source")

    delete_cmd = subparsers.add_parser("delete", help="Remove a document from the index")
    delete_cmd.add_argument("source", help="Document source path to delete")

    subparsers.add_parser("stats", help="Show knowledge base statistics")
    subparsers.add_parser("chat", help="Start an interactive chat session")

    return parser


def _run_cli(args: argparse.Namespace) -> None:
    if not args.db:
        print("Error: --db or UTEN_DATABASE_URL is required.", file=sys.stderr)
        sys.exit(1)

    uten = Uten(
        database_url=args.db,
        anthropic_api_key=getattr(args, "api_key", None),
    )

    if args.command == "index":
        report = uten.index(args.files, replace=not args.no_replace)
        print(report)
    elif args.command == "ask":
        if args.stream:
            for token in uten.ask_stream(args.question, doc_source=args.source):
                print(token, end="", flush=True)
            print()
        else:
            response = uten.ask(args.question, doc_source=args.source)
            print(f"\nUten: {response.answer}")
            if response.sources:
                print(f"\nSources: {', '.join(response.sources)}")
    elif args.command == "delete":
        uten.delete(args.source)
    elif args.command == "stats":
        stats = uten.stats()
        print(f"Total chunks : {stats['total_chunks']}")
        print(f"Embedder     : {stats['embed_model']}")
        print(f"Generator    : {stats['generator_model']}")
    elif args.command == "chat":
        print("Uten is ready. Type 'quit' to exit.\n")
        history = None
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if question.lower() in {"quit", "exit", "q"}:
                print("Goodbye.")
                break
            if not question:
                continue

            print("Uten: ", end="", flush=True)
            for token in uten.ask_stream(question, history=history):
                print(token, end="", flush=True)
            print("\n")

            response = uten.ask(question, history=history)
            history = response.history
    else:
        print("No command specified. Use --help for usage.")
        sys.exit(1)


if __name__ == "__main__":
    parser = _build_cli()
    args = parser.parse_args()
    _run_cli(args)
