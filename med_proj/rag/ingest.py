"""
Document ingestion following RapidFire FiQA RAG tutorial:
DirectoryLoader -> RecursiveCharacterTextSplitter -> HuggingFaceEmbeddings -> FAISS.
Saves index to a directory (index.faiss + index.pkl) for use by retrieve().
Heavy deps (langchain, sentence-transformers) are imported lazily so the pipeline
can run with TF-IDF only if this stack is missing or broken.
"""
from pathlib import Path

from med_proj.common.logging import get_logger

log = get_logger("rag_ingest")

# FiQA-style chunking (tutorial: chunk_size 256, overlap 32)
CHUNK_SIZE = 256
CHUNK_OVERLAP = 32
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def run_ingest(kb_dir: str, out_dir: str) -> None:
    """
    Load markdown docs from kb_dir, split, embed, and persist FAISS index to out_dir.
    out_dir will contain index.faiss and index.pkl (e.g. artifacts/rag_faiss).
    If langchain/sentence-transformers are unavailable or fail, logs and returns without raising.
    """
    kb_path = Path(kb_dir)
    if not kb_path.is_dir():
        raise FileNotFoundError(f"KB dir not found: {kb_dir}")

    try:
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
    except Exception as e:
        log.warning("Skipping FAISS ingest (optional deps missing or incompatible): %s", e)
        return

    try:
        loader = DirectoryLoader(
            str(kb_path),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        raw_docs = loader.load()
    except Exception as e:
        log.warning("FAISS ingest failed during load: %s", e)
        return

    if not raw_docs:
        log.warning("No .md documents found under %s; skipping FAISS ingest", kb_dir)
        return

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        docs = splitter.split_documents(raw_docs)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        log.warning("FAISS ingest failed during split/embed: %s", e)
        return

    try:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(out_path))
        log.info(
            "RAG ingest done: %d raw docs -> %d chunks -> %s",
            len(raw_docs),
            len(docs),
            out_dir,
        )
    except Exception as e:
        log.warning("FAISS ingest failed during save: %s", e)
