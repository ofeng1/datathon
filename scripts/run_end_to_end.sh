#!/usr/bin/env bash
set -e

PYTHONPATH=src python3 - <<'PY'
import yaml, os, json
from common.io import ensure_dir
from data.data_loader import DataLoader
from data.stats import build_stats_from_raw
from rag.index import build_index
from rag.ingest import run_ingest

cfg = yaml.safe_load(open("config.yaml"))

art_dir = cfg["artifacts"]["dir"]
ensure_dir(art_dir)

raw = DataLoader().load_data(cfg["data"]["zip_filename"])

# Build stats by region and condition (for Stats tab and chatbot)
stats = build_stats_from_raw(raw)
with open(os.path.join(art_dir, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
print("Stats saved to", os.path.join(art_dir, "stats.json"))

# Build RAG knowledge base index (TF-IDF fallback)
kb_dir = cfg["rag"]["kb_dir"]
build_index(kb_dir, os.path.join(art_dir, "kb_index.joblib"))

# RapidFire-style ingestion: chunk + embed + FAISS (preferred by retrieve)
run_ingest(kb_dir, os.path.join(art_dir, "rag_faiss"))

print("\nDONE")
print("Artifacts saved to:", art_dir)
PY
