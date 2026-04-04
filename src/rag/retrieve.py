import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

FAISS_INDEX_DIR = "rag_faiss"
JOBLIB_INDEX_FILE = "kb_index.joblib"

# Process-local caches: reloading HuggingFace embeddings + FAISS on every query
# dominated latency (seconds per request). Joblib TF-IDF index is also reloaded
# unnecessarily without caching.
_faiss_lock = threading.Lock()
_faiss_cache: Optional[Tuple[str, Any]] = None  # (resolved_dir, vectorstore)

_joblib_lock = threading.Lock()
_joblib_cache: Dict[str, Any] = {}


def _get_faiss_vectorstore(faiss_dir: str):
    global _faiss_cache
    resolved = os.path.abspath(faiss_dir)
    with _faiss_lock:
        if _faiss_cache is not None and _faiss_cache[0] == resolved:
            return _faiss_cache[1]
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.load_local(
            faiss_dir, embeddings, allow_dangerous_deserialization=True
        )
        _faiss_cache = (resolved, vectorstore)
        return vectorstore


def _get_joblib_index(index_path: str):
    resolved = os.path.abspath(index_path)
    with _joblib_lock:
        if resolved not in _joblib_cache:
            _joblib_cache[resolved] = joblib.load(index_path)
        return _joblib_cache[resolved]


def _faiss_retrieve(faiss_dir: str, query: str, top_k: int):
    vectorstore = _get_faiss_vectorstore(faiss_dir)
    # similarity_search_with_score returns (Document, distance); lower distance = better
    pairs = vectorstore.similarity_search_with_score(query, k=top_k)
    out = []
    for doc, dist in pairs:
        # Convert L2 distance to a similarity score in (0,1]; higher is better
        score = 1.0 / (1.0 + float(dist))
        source = doc.metadata.get("source", "unknown")
        if isinstance(source, str) and os.path.sep in source:
            source = Path(source).name
        out.append({"score": score, "source": source, "excerpt": doc.page_content})
    return out


def _joblib_retrieve(index_path: str, query: str, top_k: int):
    idx = _get_joblib_index(index_path)
    vec, X = idx["vectorizer"], idx["matrix"]
    docs, meta = idx["docs"], idx["meta"]
    q = vec.transform([query])
    sims = cosine_similarity(q, X).ravel()
    top = np.argsort(-sims)[:top_k]
    return [
        {
            "score": float(sims[i]),
            "source": meta[i]["source"],
            "excerpt": docs[i],
        }
        for i in top
    ]


def rag_available(artifacts_dir: str) -> bool:
    """True if either FAISS or joblib RAG index exists under artifacts_dir."""
    faiss_dir = os.path.join(artifacts_dir, FAISS_INDEX_DIR)
    joblib_path = os.path.join(artifacts_dir, JOBLIB_INDEX_FILE)
    return os.path.isdir(faiss_dir) or os.path.isfile(joblib_path)


def retrieve(artifacts_dir: str, query: str, top_k: int = 4):
    """
    Retrieve top_k chunks for query. Prefers FAISS index (rag_faiss) if present,
    else uses TF-IDF index (kb_index.joblib). Returns list of {score, source, excerpt}.
    """
    faiss_dir = os.path.join(artifacts_dir, FAISS_INDEX_DIR)
    joblib_path = os.path.join(artifacts_dir, JOBLIB_INDEX_FILE)
    if os.path.isdir(faiss_dir):
        return _faiss_retrieve(faiss_dir, query, top_k)
    if os.path.isfile(joblib_path):
        return _joblib_retrieve(joblib_path, query, top_k)
    return []
