#!/usr/bin/env bash
set -e

python - <<'PY'
import yaml, os, json
from med_proj.common.io import ensure_dir
from med_proj.data.data_loader import DataLoader
from med_proj.data.stats import build_stats_from_raw
from med_proj.rag.index import build_index

cfg = yaml.safe_load(open("config.yaml"))

art_dir = cfg["artifacts"]["dir"]
ensure_dir(art_dir)

raw = DataLoader().load_data(cfg["data"]["zip_filename"])

# Build stats by region and condition (for Stats tab and chatbot)
stats = build_stats_from_raw(raw)
with open(os.path.join(art_dir, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
print("Stats saved to", os.path.join(art_dir, "stats.json"))

# Build RAG knowledge base index
kb_dir = cfg["rag"]["kb_dir"]
build_index(kb_dir, os.path.join(art_dir, "kb_index.joblib"))

print("\nDONE")
print("Artifacts saved to:", art_dir)
PY
