import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def retrieve(index_path: str, query: str, top_k: int = 4):
    idx = joblib.load(index_path)
    vec, X = idx["vectorizer"], idx["matrix"]
    docs, meta = idx["docs"], idx["meta"]

    q = vec.transform([query])
    sims = cosine_similarity(q, X).ravel()
    top = np.argsort(-sims)[:top_k]

    out = []
    for i in top:
        out.append({
            "score": float(sims[i]),
            "source": meta[i]["source"],
            "excerpt": docs[i],
        })
    return out