from langchain_chroma import Chroma  # Connect to the persisted Chroma vector store.
from langchain_huggingface import HuggingFaceEmbeddings  # Create query/document embeddings.
from sentence_transformers import CrossEncoder  # Rerank retrieved text with a query-aware model.

from .config import (
    CHROMA_DIR,
    DEVICE,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    TOP_K_RETRIEVE,
    TOP_N_RERANK,
)

# Lazy-loaded singletons so the (slow) models only load once per process
_embeddings = None  # Cache the embedding model after its first use.
_vectordb = None  # Cache the Chroma connection after its first use.
_reranker = None  # Cache the cross-encoder after its first use.


def _get_vectordb():
    global _embeddings, _vectordb  # Update the module-level lazy caches.
    if _vectordb is None:  # Avoid repeatedly loading expensive models.
        _embeddings = HuggingFaceEmbeddings(  # Create the same embedding function used at ingest time.
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": DEVICE},
        )
        _vectordb = Chroma(  # Open the persisted collection for similarity search.
            persist_directory=str(CHROMA_DIR),
            embedding_function=_embeddings,
        )
    return _vectordb  # Return the shared vector database instance.


def _get_reranker():
    global _reranker  # Update the module-level reranker cache.
    if _reranker is None:  # Load the reranker only when retrieval first needs it.
        _reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE)  # Initialize the scoring model.
    return _reranker  # Return the shared reranker instance.


def retrieve(query: str):
    """Two-stage retrieval: wide vector search, then cross-encoder rerank."""
    vectordb = _get_vectordb()  # Obtain the persisted vector index.
    candidates = vectordb.similarity_search(query, k=TOP_K_RETRIEVE)  # Run broad semantic search.
    if not candidates:  # Handle an empty or unavailable knowledge base gracefully.
        return []  # Give generation no context.

    reranker = _get_reranker()  # Obtain the more precise cross-encoder.
    pairs = [(query, doc.page_content) for doc in candidates]  # Prepare query/text scoring pairs.
    scores = reranker.predict(pairs)  # Calculate relevance scores for every candidate.

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)  # Rank high to low.
    return [doc for doc, _ in ranked[:TOP_N_RERANK]]  # Return only the best context chunks.
