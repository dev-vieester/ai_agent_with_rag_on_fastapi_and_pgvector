"""
embedder.py
-----------
Uten Assistant RAG — Embedding Pipeline (Step 3)

Converts text chunks produced by document_ingestion.py into float vectors
using a local, free sentence-transformers model.  No API key required.
No cost per call.  Runs entirely on your machine.

Model used by default:
    all-MiniLM-L6-v2
        - 384-dimensional vectors
        - Max 256 tokens per chunk
        - ~80 MB download (cached after first run)
        - Fast on CPU, excellent semantic quality

Swapping models:
    Change DEFAULT_MODEL to any sentence-transformers model name.
    Larger models produce better vectors but are slower:
        "all-mpnet-base-v2"       → 768 dims, higher quality, slower
        "multi-qa-MiniLM-L6-cos-v1" → tuned specifically for Q&A retrieval

Usage:
    from embedder import Embedder, EmbeddedChunk
    from document_ingestion import ingest_file

    embedder = Embedder()

    doc = ingest_file("policy.pdf")

    embedded = embedder.embed_document(doc)

    for ec in embedded:
        print(ec.text[:60])
        print(ec.vector[:5])        # first 5 of 384 floats
        print(ec.metadata)

Dependencies:
    pip install sentence-transformers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import time

from sentence_transformers import SentenceTransformer

from .document_ingestion import Document



DEFAULT_MODEL = "all-MiniLM-L6-v2"



@dataclass
class EmbeddedChunk:
    """
    A single chunk of text paired with its embedding vector.

    This is the unit that gets stored in pgvector.  Every row in your
    vector table will correspond to one EmbeddedChunk.

    Attributes:
        text       : the raw chunk text (stored alongside the vector so
                     you can return it to Claude as context)
        vector     : list of floats — the embedding (384 numbers for
                     all-MiniLM-L6-v2)
        doc_title  : title of the parent document (for display / filtering)
        doc_source : file path or "text_input" (for citations)
        chunk_index: position of this chunk within its document
                     (useful for re-ordering retrieved chunks)
        metadata   : any extra fields from the parent Document.metadata
    """

    text:        str
    vector:      list[float]
    doc_title:   str         = "Untitled"
    doc_source:  str         = "unknown"
    chunk_index: int         = 0
    metadata:    dict        = field(default_factory=dict)



class Embedder:
    """
    Wraps SentenceTransformer and exposes a clean interface for Uten.

    Why a class and not a module-level function?
        Loading the model is expensive — it downloads ~80 MB on first run
        and loads the neural network weights into RAM.  You only want to
        do this ONCE per application session, not once per document.

        By making it a class, the model is loaded in __init__ and reused
        across every embed_document() call.  In Dart terms this is like
        a singleton service you inject via Riverpod — created once,
        shared everywhere.

    Usage pattern:
        embedder = Embedder()

        chunks = embedder.embed_document(doc)
        vector = embedder.embed_query("what are the card limits?")
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """
        Load the embedding model.

        On first call:  downloads the model from HuggingFace (~80 MB)
                        and caches it in ~/.cache/huggingface/
        On subsequent calls: loads from local cache instantly (~1-2 sec)

        Args:
            model_name: any sentence-transformers model name.
                        Defaults to all-MiniLM-L6-v2.
        """
        print(f"[Embedder] Loading model '{model_name}'...")
        start = time.time()

        self._model      = SentenceTransformer(model_name)
        self._model_name = model_name

        self.dimensions  = self._model.get_sentence_embedding_dimension()

        elapsed = time.time() - start
        print(f"[Embedder] Ready. Model: {model_name} | Dims: {self.dimensions} | Loaded in {elapsed:.2f}s")


    def embed_document(
        self,
        doc: Document,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[EmbeddedChunk]:
        """
        Embed all chunks from a Document object.

        Args:
            doc:           A Document produced by document_ingestion.py.
            batch_size:    How many chunks to embed in one forward pass.
                           Larger = faster but more RAM.  32 is safe for
                           most machines.
            show_progress: Print a tqdm progress bar (useful for large docs).

        Returns:
            List of EmbeddedChunk objects, one per doc.chunks entry,
            in the same order.

        Why batch encoding?
            Calling model.encode() once per chunk is very slow — the model
            has setup overhead per call.  Passing all chunks as a list lets
            the model process them in parallel on GPU/CPU in one shot.
            This is called 'batch inference' — same concept as batching
            API requests, just happening inside the neural network.
        """
        if not doc.chunks:
            return []

        vectors = self._model.encode(
            doc.chunks,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embedded = []
        for i, (chunk_text, vector) in enumerate(zip(doc.chunks, vectors)):
            embedded.append(EmbeddedChunk(
                text        = chunk_text,
                vector      = vector.tolist(),   # numpy array → plain Python list
                doc_title   = doc.title,
                doc_source  = doc.source,
                chunk_index = i,
                metadata    = {
                    **doc.metadata,              # spread parent doc metadata
                    "chunk_index": i,
                    "total_chunks": len(doc.chunks),
                    "model": self._model_name,
                },
            ))

        return embedded


    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string for similarity search.

        This is called at retrieval time — when the user asks a question,
        we embed the question and search pgvector for the closest chunks.

        Args:
            query: the user's question or search string.

        Returns:
            A list of floats (length = self.dimensions).

        Note:
            We use the same model and normalization as embed_document()
            so vectors are in the same space and distances are meaningful.
            Using different models for documents vs queries would give
            completely wrong search results.
        """
        vector = self._model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vector.tolist()


    def embed_many(
        self,
        docs: list[Document],
        batch_size: int = 32,
    ) -> list[list[EmbeddedChunk]]:
        """
        Embed all chunks across a list of documents.

        Typically called after ingest_many() — you pass the successful
        documents in and get back a nested list of EmbeddedChunks.

        Args:
            docs:       List of Document objects.
            batch_size: Batch size passed to the underlying encode call.

        Returns:
            List of lists — one inner list per document, preserving order.

        Example:
            results  = ingest_many(file_paths)
            good_docs = [r.document for r in results if r.success]
            embedded  = embedder.embed_many(good_docs)

            all_chunks = [ec for doc_chunks in embedded for ec in doc_chunks]
        """
        return [
            self.embed_document(doc, batch_size=batch_size)
            for doc in docs
        ]


    def __repr__(self) -> str:
        return f"Embedder(model='{self._model_name}', dims={self.dimensions})"
