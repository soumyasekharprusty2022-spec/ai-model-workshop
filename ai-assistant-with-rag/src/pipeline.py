import re  # Match greetings and identity questions before retrieval.

from .generator import OUT_OF_SCOPE_MESSAGE, condense_question, generate  # Use generation helpers.
from .retriever import retrieve  # Use semantic search and reranking.

# Deterministic catch for greetings/identity questions. Handled *before* retrieval so
# an unrelated but superficially-matching document chunk (e.g. a first-person foreword
# signed by an official) can never get mistaken by the LLM for an answer to "who are you".
_IDENTITY_PATTERNS = [  # Questions that should not be answered from the health document.
    r"\bwho are you\b",
    r"\bwhat are you\b",
    r"\byour name\b",
    r"\bwhat can you do\b",
    r"^\s*(hi|hello|hey)\b",
]


def _is_greeting_or_identity(question: str) -> bool:
    return any(re.search(p, question, re.IGNORECASE) for p in _IDENTITY_PATTERNS)  # Match any pattern.


def run(question: str, history: list[dict] | None = None) -> dict:
    """
    history: list of {"question": ..., "answer": ...} dicts, oldest first. Pass the
    last few turns only -- callers are responsible for capping length (see app.py).
    """
    history = history or []  # Normalize missing history to an empty list.

    # Checked against the raw question, not history -- a mid-conversation "hi" should
    # still short-circuit the same way a first-turn "hi" does.
    if _is_greeting_or_identity(question):  # Prevent unrelated questions reaching the document search.
        return {"answer": OUT_OF_SCOPE_MESSAGE, "sources": []}  # Return a deterministic response.

    standalone_question = condense_question(question, history)  # Resolve follow-up references.

    chunks = retrieve(standalone_question)  # Find and rerank relevant document chunks.
    answer = generate(standalone_question, chunks)  # Ask Ollama to answer from those chunks.

    if answer == OUT_OF_SCOPE_MESSAGE:  # Do not claim sources for a rejected question.
        sources = []  # Keep the source list consistent with the fallback answer.
    else:
        sources = sorted({doc.metadata.get("source", "unknown") for doc in chunks})  # Deduplicate paths.

    return {"answer": answer, "sources": sources}  # Expose the complete result to the caller.
