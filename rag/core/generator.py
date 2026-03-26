"""
generator.py
------------
Uten Assistant RAG — Generation Pipeline (Step 5)

Takes retrieved context chunks from vector_store.py and a user question,
constructs a grounded prompt, and calls the Anthropic Claude API to
generate an answer.

This is the final step of the RAG pipeline:

    User question
         ↓
    embed_query()          ← embedder.py
         ↓
    store.search()         ← vector_store.py
         ↓
    build_context_prompt() ← this file
         ↓
    Claude API call        ← this file
         ↓
    UtenResponse (answer + sources + history)

Key design decisions:
    - System prompt instructs Claude to answer ONLY from provided context.
      This prevents hallucination — if the answer isn't in your documents,
      Claude says so rather than making something up.
    - Conversation history is maintained across turns so follow-up
      questions work naturally ("tell me more", "what about the fees?").
    - Sources are extracted from the retrieved chunks so Claude can cite
      which document each answer came from.
    - Streaming is supported for a responsive UI — tokens appear as they
      are generated rather than waiting for the full response.

Usage:
    from generator import Generator
    from vector_store import VectorStore, SearchResult
    from embedder import Embedder

    generator = Generator(api_key="your-anthropic-key")

    response = generator.ask(
        question="What are the virtual card limits?",
        context_chunks=search_results,
    )
    print(response.answer)
    print(response.sources)

    response1 = generator.ask("What is Lock and Save?", context_chunks)
    response2 = generator.ask(
        "What interest rate does it offer?",
        context_chunks,
        history=response1.history,   # carry history forward
    )

Dependencies:
    pip install anthropic
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Iterator
import os

import anthropic

from ..storage.vector_store import SearchResult



DEFAULT_MODEL = "claude-sonnet-4-6"

MAX_TOKENS = 1024

MIN_SIMILARITY = 0.3

SYSTEM_PROMPT = """You are Uten, an intelligent assistant with access to a \
knowledge base of documents.

Your job is to answer the user's questions accurately and concisely using \
ONLY the context provided below each question.

Rules you must follow:
1. Base your answer strictly on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say clearly:
   "I don't have enough information in my knowledge base to answer that."
3. When your answer draws from a specific document, mention the document title
   naturally in your response (e.g. "According to the XedlaPay Features doc...").
4. Keep answers concise and direct. Use bullet points for lists.
5. Never make up facts, figures, or policies not present in the context.
"""



@dataclass
class UtenResponse:
    """
    The complete output of one ask() call.

    Attributes:
        answer   : Claude's generated answer text
        sources  : deduplicated list of document titles that were used
                   as context — for displaying citations to the user
        history  : the updated conversation history including this turn —
                   pass this back in to the next ask() call for multi-turn
        chunks_used : the SearchResult objects that were included in the
                      prompt — useful for debugging retrieval quality
        model    : which Claude model generated the answer
    """
    answer:      str
    sources:     list[str]
    history:     list[dict]
    chunks_used: list[SearchResult]
    model:       str


@dataclass
class ConversationHistory:
    """
    A lightweight wrapper around a list of message dicts.

    The Anthropic API expects messages in this format:
        [
            {"role": "user",      "content": "What is Lock and Save?"},
            {"role": "assistant", "content": "Lock and Save is a savings..."},
            {"role": "user",      "content": "What interest rate?"},
        ]

    We wrap this list in a class to get helper methods for adding turns
    and trimming old history when it gets too long.

    Why trim history?
        Every message in history is sent to Claude on every turn, consuming
        tokens from the context window.  For a long conversation this can
        become expensive and hit token limits.  Keeping the last N turns
        is the standard approach.
    """

    messages: list[dict] = field(default_factory=list)
    max_turns: int = 10   # keep last 10 turns (5 user + 5 assistant)

    def add_turn(self, role: str, content: str) -> None:
        """
        Append one message to the history.

        Args:
            role:    "user" or "assistant"
            content: the message text
        """
        self.messages.append({"role": role, "content": content})

        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def to_list(self) -> list[dict]:
        """Return the raw list the Anthropic API expects."""
        return self.messages.copy()



class Generator:
    """
    Wraps the Anthropic API and handles RAG prompt construction.

    Like Embedder and VectorStore, this is designed to be created once
    at app startup.  The Anthropic client handles connection pooling
    internally.

    The ask() method is the main entry point — it takes a question and
    retrieved chunks, builds the prompt, calls Claude, and returns a
    clean UtenResponse.
    """

    def __init__(
        self,
        api_key:   Optional[str] = None,
        model:     str = DEFAULT_MODEL,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        """
        Initialise the Anthropic client.

        Args:
            api_key:    Anthropic API key.  If not provided, reads from
                        the ANTHROPIC_API_KEY environment variable.
                        Never hardcode this in source — use an env var or
                        a secrets manager.
            model:      Claude model to use for generation.
            max_tokens: Maximum tokens in Claude's response.

        Why environment variables for secrets?
            Same reason you use --dart-define or flutter_dotenv in Flutter:
            the key should never appear in your source code or git history.
            On your machine: export ANTHROPIC_API_KEY=sk-ant-...
            On a server: set it in your deployment environment.
        """
        normalized_api_key = api_key.strip() if isinstance(api_key, str) else api_key
        self._client    = anthropic.Anthropic(api_key=normalized_api_key)
        self._model     = model
        self._max_tokens = max_tokens

        print(f"[Generator] Ready. Model: {model}")


    def _normalize_prior_history(
        self,
        question: str,
        history: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Ensure the current question is not already present in prior history.

        Some clients send the active user message both as AskRequest.question
        and as the last history entry. That would cause Claude to receive the
        same user turn twice. We keep history as previous turns only.
        """
        prior_messages = list(history or [])
        normalized_question = question.strip()

        while (
            prior_messages
            and prior_messages[-1].get("role") == "user"
            and str(prior_messages[-1].get("content", "")).strip() == normalized_question
        ):
            prior_messages.pop()

        return prior_messages

    def _build_context_block(
        self,
        chunks: list[SearchResult],
        min_similarity: float = MIN_SIMILARITY,
    ) -> tuple[str, list[SearchResult], list[str]]:
        """
        Convert retrieved SearchResult objects into a formatted context
        string to inject into Claude's prompt.

        Args:
            chunks:         Results from VectorStore.search()
            min_similarity: Drop chunks below this score

        Returns:
            Tuple of:
                context_text  — the formatted string to inject into prompt
                used_chunks   — the chunks that passed the filter
                sources       — deduplicated document titles

        Why format chunks with headers?
            Without headers, multiple chunks from different documents blend
            together and Claude can't tell which document said what.
            The header makes citations natural and accurate.

        Why filter by similarity?
            Vector search always returns top_k results even if they're
            not actually relevant.  A chunk with similarity 0.2 is likely
            a false positive — including it adds noise and can mislead Claude.
        """
        used_chunks = [c for c in chunks if c.similarity >= min_similarity]

        if not used_chunks:
            return "No relevant context found.", [], []

        sections = []
        for chunk in used_chunks:
            header = (
                f"--- Source: {chunk.doc_title} | "
                f"chunk {chunk.chunk_index} | "
                f"relevance {chunk.similarity:.2f} ---"
            )
            sections.append(f"{header}\n{chunk.text}")

        context_text = "\n\n".join(sections)

        sources = list(dict.fromkeys(c.doc_title for c in used_chunks))

        return context_text, used_chunks, sources


    def ask(
        self,
        question:       str,
        context_chunks: list[SearchResult],
        history:        Optional[list[dict]] = None,
        min_similarity: float = MIN_SIMILARITY,
    ) -> UtenResponse:
        """
        Generate an answer to a question grounded in the retrieved context.

        Args:
            question:       The user's question string.
            context_chunks: SearchResults from VectorStore.search().
            history:        Previous conversation turns as a list of
                            {"role": ..., "content": ...} dicts.
                            Pass None (or omit) for the first question.
                            Pass response.history from the previous call
                            for follow-up questions.
            min_similarity: Minimum relevance score for context inclusion.

        Returns:
            UtenResponse with answer, sources, and updated history.

        How the prompt is structured:
            The system prompt is sent separately via the system= parameter —
            not as a message in the history.  This is how the Anthropic API
            works: system sets behaviour, messages are the conversation.

            Each user message embeds the context block inside it:
                "Context:\n{chunks}\n\nQuestion: {question}"

            This means every question carries its own fresh context — the
            retrieved chunks are always current for that specific question.
            We don't put context in history because old context from turn 1
            is irrelevant to the question in turn 5.
        """
        context_text, used_chunks, sources = self._build_context_block(
            context_chunks, min_similarity
        )

        user_message = (
            f"Context from knowledge base:\n"
            f"{context_text}\n\n"
            f"Question: {question}"
        )

        prior_messages = self._normalize_prior_history(question, history)
        messages = [
            *prior_messages,                          # previous turns
            {"role": "user", "content": user_message} # current question
        ]

        response = self._client.messages.create(
            model      = self._model,
            max_tokens = self._max_tokens,
            system     = SYSTEM_PROMPT,
            messages   = messages,
        )

        answer = response.content[0].text

        updated_history = [
            *prior_messages,
            {"role": "user",      "content": question},  # plain question
            {"role": "assistant", "content": answer},
        ]

        return UtenResponse(
            answer      = answer,
            sources     = sources,
            history     = updated_history,
            chunks_used = used_chunks,
            model       = self._model,
        )


    def ask_stream(
        self,
        question:       str,
        context_chunks: list[SearchResult],
        history:        Optional[list[dict]] = None,
        min_similarity: float = MIN_SIMILARITY,
    ) -> Iterator[str]:
        """
        Streaming version of ask() — yields tokens as they arrive.

        Use this for a chat UI where you want the response to appear
        word by word instead of waiting for the full response.

        Args:
            Same as ask().

        Yields:
            str — one text delta at a time as Claude generates it.

        Example:
            for token in generator.ask_stream(question, chunks):
                print(token, end="", flush=True)

        Why streaming matters for UX:
            A non-streaming call on a long answer might take 3-5 seconds
            before anything appears.  Streaming shows the first word in
            ~300ms — the user sees progress immediately, which feels
            much more responsive.  Same reason Flutter uses StreamBuilder
            instead of FutureBuilder for live data.
        """
        context_text, used_chunks, sources = self._build_context_block(
            context_chunks, min_similarity
        )

        user_message = (
            f"Context from knowledge base:\n"
            f"{context_text}\n\n"
            f"Question: {question}"
        )

        prior_messages = self._normalize_prior_history(question, history)
        messages = [
            *prior_messages,
            {"role": "user", "content": user_message},
        ]

        with self._client.messages.stream(
            model      = self._model,
            max_tokens = self._max_tokens,
            system     = SYSTEM_PROMPT,
            messages   = messages,
        ) as stream:
            for text_delta in stream.text_stream:
                yield text_delta

    def __repr__(self) -> str:
        return f"Generator(model='{self._model}', max_tokens={self._max_tokens})"
