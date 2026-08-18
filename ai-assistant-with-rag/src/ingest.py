"""
Run as a module from the repo root:
    python -m src.ingest

Do NOT run as `python src/ingest.py` -- direct execution has no package context,
so the relative import below fails with:
    ImportError: attempted relative import with no known parent package
"""
import glob  # Find source files recursively using the configured pattern.
import os  # Inspect file extensions in a platform-independent way.

from langchain_chroma import Chroma  # Persist document vectors in Chroma.
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader  # Parse supported files.
from langchain_huggingface import HuggingFaceEmbeddings  # Generate vectors for text chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Divide documents into chunks.

from .config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    DATA_GLOB,
    DEVICE,
    EMBEDDING_MODEL,
)

LOADER_MAP = {  # Map each supported extension to its LangChain loader class.
    ".docx": Docx2txtLoader,  # Extract text from Word documents.
    ".pdf": PyPDFLoader,  # Extract text from PDF documents.
}


def load_documents():
    paths = glob.glob(str(DATA_DIR / DATA_GLOB), recursive=True)  # Find files under data/.
    docs = []  # Collect loaded LangChain Document objects.
    for path in paths:  # Process every matching source file.
        ext = os.path.splitext(path)[1].lower()  # Normalize the file extension.
        loader_cls = LOADER_MAP.get(ext)  # Select the parser for that extension.
        if loader_cls is None:  # Protect against unsupported file types.
            print(f"Skipping unsupported file type: {path}")  # Explain why a file was ignored.
            continue  # Move to the next path.
        docs.extend(loader_cls(path).load())  # Parse the file and append its pages/documents.
    print(f"Loaded {len(docs)} documents")  # Report the ingestion input size.
    return docs  # Pass parsed documents to the splitter.


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(  # Create a boundary-aware chunking strategy.
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)  # Preserve document metadata while splitting text.
    print(f"Split into {len(chunks)} chunks")  # Report the number of indexed units.
    return chunks  # Pass chunks to embedding and persistence.


def build_index(chunks):
    embeddings = HuggingFaceEmbeddings(  # Initialize the local embedding model.
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
    )
    # langchain_chroma auto-persists when persist_directory is set -- no .persist() call
    vectordb = Chroma.from_documents(  # Embed chunks and write them to the Chroma directory.
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return vectordb  # Return the created index for callers that need it immediately.


def main():
    docs = load_documents()  # Load source files from data/.
    if not docs:  # Stop before creating an empty index.
        print(
            f"No documents found matching '{DATA_GLOB}' in {DATA_DIR}. "
            "Check the glob pattern in config.py and that data/ isn't empty."
        )
        return  # Exit cleanly and let the user correct the input data or glob.
    chunks = split_documents(docs)  # Convert source documents into searchable chunks.
    build_index(chunks)  # Embed and persist the chunks.
    print(f"Index persisted to {CHROMA_DIR}")  # Confirm the output location.


if __name__ == "__main__":  # Support `python -m src.ingest` execution.
    main()  # Start the indexing workflow.
