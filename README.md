# UCSB Datathon Project

## ED Revisit Risk: a RAG-backed chatbot that estimates readmission risk and surfaces evidence-based recommendations from a clinical knowledge base.

---

## What it does

- **Assess a patient** — Describe a patient in plain language (e.g. *72 year old male with COPD and CHF, temp 101, BP 135/85, pulse 110, pain 8/10*) or **upload an ED record PDF**; the system parses vitals and conditions, runs a readmission model, and returns a **Patient Summary**, **risk score**, and **recommendations**.
- **Ask the knowledge base** — Ask clinical questions (e.g. *What are risk factors for ED revisits?*) and get relevant excerpts from the indexed guidance.
- **Stats** — View precomputed NHAMCS-based stats (72-hour revisit rates, admission rates by condition and region).

---

## Architecture overview

```
config.yaml
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  OFFLINE PIPELINE (scripts/run_end_to_end.sh)                    │
│  • Load NHAMCS ED data → build stats.json                       │
│  • Index knowledge base (markdown) → TF-IDF (kb_index.joblib)    │
│  • Optional: same KB → chunk + embed + FAISS (rag_faiss/)       │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼  artifacts/  (stats.json, kb_index.joblib, rag_faiss/, readmission model)
     │
┌─────────────────────────────────────────────────────────────────┐
│  RUNTIME: FastAPI service (med_proj/service/api.py)             │
│  • Loads readmission model + stats at startup            │
│  • /chat → ChatEngine (state, intent, extract, predict, RAG)     │
│  • /parse-ed-document → PDF → parsed state for merge into chat   │
│  • /health, /stats → status and stats for UI                     │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
  Static UI (index.html): chat, PDF upload, Stats tab
```

---

## Model and ingestion

- **Data** — NHAMCS ED encounter data (e.g. SAS ZIP) is loaded from paths in `config.yaml`; the pipeline builds **stats** (regional and condition-level 72-hour revisit and admission percentages) and writes `artifacts/stats.json`.
- **Readmission model** — An **XGBoost** classifier (trained separately by the modeling pipeline) is loaded from `artifacts/` at service startup. It predicts a base readmission probability from features (age, sex, vitals, conditions, triage, etc.); the chatbot then applies an **evidence-based clinical risk adjustment** (log-odds shifts for conditions, abnormal vitals, age, triage) to produce the final probability shown to the user.
- **RAG knowledge base** — Markdown files under `med_proj/rag/knowledge_base/` are indexed in two ways:
  - **TF-IDF** (scikit-learn): document-level vectors → `artifacts/kb_index.joblib` (always built).
  - **FAISS** (optional): same docs are chunked (RecursiveCharacterTextSplitter), embedded with `sentence-transformers/all-MiniLM-L6-v2`, and stored as `artifacts/rag_faiss/`. Retrieval uses FAISS when present, otherwise falls back to TF-IDF.
- **ED form parsing** — Uploaded PDFs are converted to text (pypdf); regex and NLP extractors fill structured state (age, sex, vitals, ESI/triage, conditions, chief complaint, allergies, disposition, diagnosis) and that state is merged into the chat session for the next assessment.

---

## RAG and chatbot

- **RAG** does not generate text; it **retrieves** top-k passages from the knowledge base. The chatbot uses it for: (1) **Ask** — user’s question as query → “Knowledge Base Results”; (2) **Recommendations** — after an assessment, a synthetic query (risk level + conditions, e.g. “high risk readmission recommendations CHF elderly”) → “Recommendations” section.
- **Chatbot** — Per-session state (vitals, conditions, optional form fields). Each message is **intent-classified** (greeting, help, reset, ask, update, assess). For assess/update: **extractors** pull clinical values from text (or use merged PDF state), **infer** missing triage from conditions/vitals, **predict** with XGBoost + adjustment, then **assemble** reply: Patient Summary, risk score, RAG recommendations, and condition-level stats from `stats.json`.

---

## API

| Endpoint | Description |
|----------|-------------|
| `GET /` | Redirects to `/static/index.html` (single-page app). |
| `GET /health` | `{ status, models_loaded, rag_index_loaded }`. |
| `POST /chat` | Body: `{ message, session_id?, merge_state? }`. Returns `{ session_id, reply }` (markdown). |
| `POST /parse-ed-document` | Upload PDF; returns `{ parsed, summary }` for use with `/chat` and `merge_state`. |
| `GET /stats` | Precomputed stats JSON for the Stats tab. |

---

## Setup and run

1. **Config** — Edit `config.yaml` for data paths, `rag.kb_dir`, and `artifacts.dir` (default `artifacts`).
2. **Install** — `pip install -r requirements.txt` (optionally `numpy<2` and `sentence-transformers<3` if you hit dependency conflicts; see `med_proj/rag/ingest.py`).
3. **Pipeline** — From repo root: `bash scripts/run_end_to_end.sh` to build stats and RAG indexes. The readmission model (e.g. `artifacts/readmission_model.json` or your trained artifact) must already exist in `artifacts/` for full risk predictions.
4. **Run service** — `uvicorn med_proj.service.api:app --host 0.0.0.0 --port 8000`. Open `http://localhost:8000/` for the chat UI.

---

## Project layout (main pieces)

- `config.yaml` — Data paths, RAG and artifacts config.
- `scripts/run_end_to_end.sh` — Builds stats and RAG indexes.
- `med_proj/rag/` — Index (TF-IDF), ingest (FAISS), retrieve (FAISS or TF-IDF).
- `med_proj/chatbot/` — Engine (state, intent, extract, predict, format), intents, extractors.
- `med_proj/data/` — Data loading, stats, ED form parser.
- `med_proj/service/` — FastAPI app, schemas, static frontend.
- `artifacts/` — stats.json, kb_index.joblib, rag_faiss/, and the readmission model (from training).

For a deeper technical walkthrough, see **`docs/TECHNICAL_OVERVIEW.md`**.
