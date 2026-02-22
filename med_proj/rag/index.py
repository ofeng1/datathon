from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from med_proj.common.logging import get_logger

log = get_logger("rag_index")

def build_index(kb_dir: str, out_path: str):
    kb = Path(kb_dir)
    docs, meta = [], []
    for p in kb.glob("*.md"):
        docs.append(p.read_text(encoding="utf-8"))
        meta.append({"source": p.name})

    if not docs:
        raise ValueError(f"No KB docs found at {kb_dir}")

    vec = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))
    X = vec.fit_transform(docs)

    joblib.dump({"vectorizer": vec, "matrix": X, "docs": docs, "meta": meta}, out_path)
    log.info("KB indexed: %d docs -> %s", len(docs), out_path)