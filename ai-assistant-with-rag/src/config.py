import os  # Read optional runtime configuration from environment variables.
from pathlib import Path  # Build operating-system-independent project paths.

import torch  # Detect whether a CUDA-capable GPU is available.
from dotenv import load_dotenv  # Load local .env settings when present.

load_dotenv()  # Make values from .env available through os.environ.

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent  # Locate the repository root.
DATA_DIR = BASE_DIR / "data"  # Store source documents below the repository root.
CHROMA_DIR = BASE_DIR / "chroma_db"  # Store the persisted Chroma index here.

# --- Ingest settings ---
# Change to "**/*.pdf" (or add both) if your source docs aren't .docx.
# .docx is preferred over .pdf: PyPDFLoader can mangle font encoding on some PDFs
# (dropped currency symbols, corrupted words). If you must use PDFs, spot-check
# extracted text for corruption before trusting it.
DATA_GLOB = "**/*.docx"  # Ingest Word documents recursively from data/.
CHUNK_SIZE = 1000  # Limit each indexed text chunk to a manageable size.
CHUNK_OVERLAP = 200  # Repeat boundary text so facts are less likely to be split.

# --- Embeddings (local) ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Convert text to vectors.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Select GPU when available.

# --- Retrieval ---
TOP_K_RETRIEVE = 20  # Fetch a broad set of inexpensive vector-search candidates.
TOP_N_RERANK = 5  # Keep the most relevant candidates after precise reranking.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Score query/document pairs.

# --- Conversation memory ---
# How many prior Q&A turns to keep for follow-up question condensing. Higher = better
# multi-turn context but more tokens sent to the local model per turn.
MAX_HISTORY_TURNS = 5  # Bound conversation context sent to the local model.

# --- Generation: local Ollama (no external API calls) ---
# Requires the Ollama daemon running locally (`ollama serve`) and the model pulled
# (`ollama pull phi3:mini`, or whichever OLLAMA_MODEL you set).
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")  # Ollama endpoint.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")  # Local chat model name.
