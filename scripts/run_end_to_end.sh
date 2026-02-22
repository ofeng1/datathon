#!/usr/bin/env bash
set -e

python - <<'PY'
import yaml, os, datetime
import numpy as np
import pandas as pd
from med_proj.common.io import ensure_dir
from med_proj.data.data_loader import DataLoader
from med_proj.data.normalize import MappingConfig, normalize_sas_to_encounters
from med_proj.data.labels import build_ed_revisit_labels, build_ed_to_admit_label
from med_proj.features.build_features import build_features
from med_proj.modeling.train import train_one, save
from med_proj.modeling.evaluate import evaluate
from med_proj.rag.index import build_index

cfg = yaml.safe_load(open("config.yaml"))

art_dir = cfg["artifacts"]["dir"]
ensure_dir(art_dir)

raw = DataLoader().load_data(cfg["data"]["zip_filename"])

# ---- NHAMCS preprocessing: create synthetic IDs and timestamps ----
raw["_ENCOUNTER_ID"] = np.arange(len(raw))
raw["_PATIENT_ID"] = np.arange(len(raw))

def _make_visit_date(row):
    year = int(row.get("YEAR", 2015))
    month = int(row.get("VMONTH", 1)) if not np.isnan(row.get("VMONTH", np.nan)) else 1
    month = max(1, min(12, month))
    return datetime.date(year, month, 15)

raw["_VISIT_DATE"] = raw.apply(_make_visit_date, axis=1)

def _decode_arrtime(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if isinstance(val, bytes):
        val = val.decode("utf-8", errors="ignore")
    return str(val).strip()

raw["_ARRTIME_STR"] = raw["ARRTIME"].apply(_decode_arrtime)

# ---- Build MappingConfig from config ----
m = cfg["mapping"]
mc = MappingConfig(
    patient_id_col=m["patient_id_col"],
    encounter_id_col=m["encounter_id_col"],
    start_time_col=m.get("start_time_col"),
    end_time_col=m.get("end_time_col"),
    start_date_col=m.get("start_date_col"),
    start_clock_col=m.get("start_clock_col"),
    end_date_col=m.get("end_date_col"),
    end_clock_col=m.get("end_clock_col"),
    admitted_col=m.get("admitted_col"),
    admitted_positive_values=m.get("admitted_positive_values"),
)

try:
    enc = normalize_sas_to_encounters(raw, mc)
except Exception:
    print("\n=== COLUMN DEBUG HELP ===")
    cols = list(raw.columns)
    print("Columns (first 200):")
    print(cols[:200])
    print("\nSample head:")
    print(raw.head(2))
    raise

# ---- Labels: use NHAMCS pre-computed flags (SEEN72, ADMITHOS) ----
nhamcs_cfg = cfg.get("nhamcs", {})
seen72_col = nhamcs_cfg.get("seen72_col", "SEEN72")
seen72_pos = set(nhamcs_cfg.get("seen72_positive", [1, 1.0]))
admithos_col = nhamcs_cfg.get("admithos_col", "ADMITHOS")
admithos_pos = set(nhamcs_cfg.get("admithos_positive", [1, 1.0]))

if seen72_col in raw.columns:
    seen72_flag = raw[seen72_col].apply(lambda x: 1 if x in seen72_pos else 0).values
    enc["label_ed_revisit_72h"] = seen72_flag[enc.index] if len(enc) == len(raw) else 0
    enc["label_ed_revisit_7d"] = enc["label_ed_revisit_72h"]
    enc["label_ed_revisit_30d"] = enc["label_ed_revisit_72h"]
    print(f"SEEN72 label positives: {int(enc['label_ed_revisit_72h'].sum())}")
else:
    enc["label_ed_revisit_72h"] = 0
    enc["label_ed_revisit_7d"] = 0
    enc["label_ed_revisit_30d"] = 0
    print("WARNING: SEEN72 column not found, revisit labels set to 0")

if admithos_col in raw.columns:
    admit_flag = raw[admithos_col].apply(lambda x: 1 if x in admithos_pos else 0).values
    enc["label_ed_admit"] = admit_flag[enc.index] if len(enc) == len(raw) else 0
    print(f"ADMITHOS label positives: {int(enc['label_ed_admit'].sum())}")
else:
    if "admitted" in enc.columns:
        enc["label_ed_admit"] = enc["admitted"].astype(int)
    else:
        enc["label_ed_admit"] = 0
        print("WARNING: ADMITHOS column not found, admit labels set to 0")

feat = build_features(enc)

seed = cfg["modeling"]["random_seed"]
ts = cfg["modeling"]["test_size"]
cal = cfg["modeling"]["calibrate_method"]

def fit(label_col, out_name):
    if label_col not in feat.columns:
        print(f"Skipping {out_name}: label not found ({label_col})")
        return
    if len(set(feat[label_col].astype(int).values)) < 2:
        print(f"Skipping {out_name}: label has only one class")
        return
    model, Xte, yte = train_one(feat, label_col, seed, ts, cal)
    save(model, os.path.join(art_dir, out_name))
    evaluate(model, Xte, yte, out_name)

fit("label_ed_revisit_72h", "model_ed72.joblib")
fit("label_ed_revisit_7d", "model_ed7d.joblib")
fit("label_ed_revisit_30d", "model_ed30d.joblib")
fit("label_ed_admit", "model_ed_admit.joblib")

kb_dir = cfg["rag"]["kb_dir"]
build_index(kb_dir, os.path.join(art_dir, "kb_index.joblib"))

print("\nDONE")
print("Artifacts saved to:", art_dir)
PY
