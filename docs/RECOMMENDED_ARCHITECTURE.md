# Recommended Repository Architecture

This repo is already close to a clean layered design, but several generated/data-heavy artifacts and notebooks are mixed into the source package (`med_proj/`).  
The recommendation below separates **source code**, **runtime artifacts**, **raw data**, and **experiments** so the project is easier to maintain and scale.

## Goals

- Keep importable app code in one place (`src/`)
- Keep large/static assets out of package modules
- Separate reproducible pipeline outputs from checked-in source
- Make notebooks and experiments discoverable and non-blocking
- Reduce noise (`__pycache__`, duplicate notebooks, mixed assets)

## Recommended Top-Level Structure

```text
.
├── src/
│   ├── api/                     # FastAPI routes, schemas
│   ├── chatbot/                 # Conversation engine, intents, extractors
│   ├── data/                    # Data parsing/normalization/stats code only
│   ├── features/                # Feature schemas + feature builders
│   ├── modeling/                # Train/eval/metrics/calibration code
│   ├── rag/                     # Retrieval + indexing logic
│   ├── common/                  # Shared utilities
│   └── cli/                     # CLI entrypoints
├── apps/
│   └── web/
│       └── static/              # index.html, logo.jpg, frontend assets
├── data/
│   ├── raw/                     # NHAMCS zips, immutable input files
│   ├── interim/                 # temporary transformed files
│   └── processed/               # cleaned tables/feature-ready data
├── artifacts/
│   ├── models/                  # readmission_model.json, model binaries
│   ├── rag/                     # kb_index.joblib, FAISS index
│   └── stats/                   # stats.json and analytics outputs
├── knowledge_base/              # source markdown docs used by RAG
├── notebooks/
│   ├── exploration/
│   └── modeling/
├── scripts/                     # orchestration scripts
├── docs/                        # architecture + technical docs
├── tests/                       # unit/integration tests
├── config/
│   ├── base.yaml
│   ├── dev.yaml
│   └── prod.yaml
└── README.md
```

## Move Plan (Current -> Recommended)

### Code modules

| Current path | Recommended path | Notes |
|---|---|---|
| `med_proj/service/api.py` | `src/api/routes.py` | Keep FastAPI app wiring here |
| `med_proj/service/schemas.py` | `src/api/schemas.py` | API request/response models |
| `med_proj/service/__init__.py` | `src/api/__init__.py` | Rename module to `api` for clarity |
| `med_proj/chatbot/*` | `src/chatbot/*` | No structure change needed |
| `med_proj/rag/*` (except knowledge base docs) | `src/rag/*` | Keep ingest/retrieve/index logic together |
| `med_proj/data/*.py` | `src/data/*.py` | Keep only code here |
| `med_proj/features/*` | `src/features/*` | Keep current structure |
| `med_proj/modeling/*` | `src/modeling/*` | Keep current structure |
| `med_proj/common/*` | `src/common/*` | Keep current structure |
| `med_proj/cli/*` | `src/cli/*` | Keep CLI package |
| `med_proj/plots.py` | `src/analysis/plots.py` | Optional: create `analysis` module |

### Frontend/static assets

| Current path | Recommended path | Notes |
|---|---|---|
| `med_proj/service/static/index.html` | `apps/web/static/index.html` | Decouple frontend from backend package |
| `med_proj/service/static/logo.jpg` | `apps/web/static/logo.jpg` | Same as above |

### RAG content and docs

| Current path | Recommended path | Notes |
|---|---|---|
| `med_proj/rag/knowledge_base/*.md` | `knowledge_base/*.md` | Treat as content, not package code |
| `med_proj/data/FEATURES_AND_RESPONSE.md` | `docs/FEATURES_AND_RESPONSE.md` | Project documentation |

### Notebooks

| Current path | Recommended path | Notes |
|---|---|---|
| `med_proj/model.ipynb` | `notebooks/modeling/model.ipynb` | Keep one canonical modeling notebook |
| `med_proj/graphs.ipynb` | `notebooks/exploration/graphs.ipynb` | EDA/visualization notebook |
| `med_proj/data/model.ipynb` | `notebooks/archive/model_data_legacy.ipynb` or delete | Appears duplicate; archive or remove |

### Data and model artifacts

| Current path | Recommended path | Notes |
|---|---|---|
| `med_proj/data/ed2015-sas.sas7bdat.zip` | `data/raw/ed2015-sas.sas7bdat.zip` | Raw data should live outside package |
| `med_proj/data/ed2016_sas.zip` | `data/raw/ed2016_sas.zip` | Same |
| `med_proj/data/ed2017_sas.zip` | `data/raw/ed2017_sas.zip` | Same |
| `med_proj/data/ed2018_sas.zip` | `data/raw/ed2018_sas.zip` | Same |
| `med_proj/data/ed2019_sas.zip` | `data/raw/ed2019_sas.zip` | Same |
| `med_proj/data/ed2020_sas.zip` | `data/raw/ed2020_sas.zip` | Same |
| `med_proj/data/ed2021_sas.zip` | `data/raw/ed2021_sas.zip` | Same |
| `med_proj/data/readmission_model.json` | `artifacts/models/readmission_model.json` | Model artifact |

## Cleanup Rules (Strongly Recommended)

- Add `__pycache__/` and `*.pyc` to `.gitignore` (if missing)
- Keep all generated artifacts under `artifacts/` only
- Keep all raw datasets under `data/raw/` only
- Do not commit notebook checkpoints or duplicate notebooks
- Use absolute config keys for data/artifact paths in YAML config files

## Migration Order (Low Risk)

1. **Create new directories** (`src/`, `apps/web/static`, `data/raw`, `artifacts/models`, `notebooks/*`)
2. **Move non-code assets first** (notebooks, zips, model json, knowledge base markdown)
3. **Move service package** (`service` -> `api`) and update imports
4. **Move remaining code to `src/`** and update entrypoints
5. **Update runtime config** (`config.yaml`) and script paths
6. **Run full smoke test** (`run_end_to_end.sh`, API startup, chat endpoint)

## Optional Naming Improvements

- Rename `scripts/run_end_to_end.sh` to `scripts/build_pipeline.sh`
- Add `Makefile` targets for `make setup`, `make pipeline`, `make serve`, `make test`

---

If useful, this can be followed by a second markdown file with an **exact import rewrite checklist** (`from med_proj.service...` -> `from api...`) and a one-shot shell migration script.
