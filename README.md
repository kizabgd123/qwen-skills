# qwen-skills

Agent Skills + Pre-Commit Hooks + ML Pipeline Orchestration for Kaggle competitions, ML pipelines, and deterministic AI workflows.

---

## Architecture

```
DATA → EDA → SCIENCE → TRAIN → ENSEMBLE → DEPLOY → VERIFY → SUBMIT
  │       │        │         │          │         │        │         │
  ▼       ▼        ▼         ▼          ▼         ▼        ▼         ▼
data   kaggle   data    machine-    ml-     machine-  judge-   kaggle
audit   -eda   scientist learning  ensemble learning  guard    -submit
                         engineer   verify   -ops-ml
                                    │        pipeline
                                    ▼
                          .pre-commit-config.yaml
                          (automated gates on every commit)
```

---

## Installed Skills (10 Total)

| # | Skill | Category | Triggers on |
|---|-------|----------|-------------|
| 1 | `data-audit` | Data Quality | "audit data", "check data quality", "duplicates", "outliers" |
| 2 | `kaggle-eda` | Exploration | "do EDA", "explore data", "check for leakage", "data profiling" |
| 3 | `data-scientist` | ML Science | algorithm selection, feature engineering, A/B tests |
| 4 | `machine-learning` | ML Core | JAX/functional ML patterns, training loops |
| 5 | `machine-learning-engineer` | MLE | model deployment, ONNX, FastAPI, Kubernetes |
| 6 | `machine-learning-ops-ml-pipeline` | MLOps | 4-phase ML pipeline orchestration |
| 7 | `ml-ensemble` | Ensemble | "ensemble", "stacking", "blending", "combine models" |
| 8 | `judge-guard-verify` | Verification | "judge-guard", "verify output", "hash check" |
| 9 | `kaggle-submit` | Submission | "submit to Kaggle", "pre-flight", "check submission" |
| 10 | `data-analysis-jupyter` | Analysis | data analysis, visualization, notebook development |

### External Skills (from npx skills ecosystem)

| Skill | Source | Installs |
|-------|--------|----------|
| `machine-learning` | mindrally/skills | 145 |
| `data-analysis-jupyter` | mindrally/skills | 269 |
| `data-scientist` | borghei/claude-skills | 194 |
| `machine-learning-ops-ml-pipeline` | sickn33/antigravity-awesome-skills | 241 |
| `machine-learning-engineer` | 404kidwiz/claude-supercode-skills | 685 |

---

## Pre-Commit Hooks (5 Custom Hooks)

Every commit triggers relevant skill-based checks automatically.

| Hook | File Pattern | Skill | Checks |
|------|-------------|-------|--------|
| `data-audit` | `*.csv`, `*.parquet` | data-audit | 7 quality checks (duplicates, missing, outliers, etc.) |
| `submission-guard` | `submission*.csv` | kaggle-submit | 5 gates (row count, nulls, range, duplicates, size) |
| `judge-guard` | `outputs/*.json` | judge-guard-verify | 3 levels (schema, hash, assertions) |
| `ensemble-diversity` | `oof/*.npy` | ml-ensemble | Pearson r < 0.95 between model pairs |
| `notebook-check` | `*.ipynb` | data-analysis-jupyter | kernelspec, paths, execution order, docs |

### Install Hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files    # First-time scan
```

---

## Pipeline Orchestrator

Run the full ML lifecycle pipeline from one command:

```bash
python pipeline/orchestrate.py --full              # All 9 phases
python pipeline/orchestrate.py --phase eda         # Single phase
python pipeline/orchestrate.py --phase train,ensemble  # Multiple
python pipeline/orchestrate.py --list              # List phases
python pipeline/orchestrate.py --dry-run           # Preview
```

### Phases

| Phase | Skill | Output |
|-------|-------|--------|
| `data-audit` | data-audit | `data/audit_report.md` |
| `eda` | kaggle-eda | `outputs/eda_report.md` |
| `features` | data-scientist | `data/processed/feature_spec.yaml` |
| `train` | machine-learning | `outputs/oof/*.npy` |
| `serving` | machine-learning-engineer | `serving/` |
| `mlops` | machine-learning-ops-ml-pipeline | `mlops/pipeline.yaml` |
| `ensemble` | ml-ensemble | `outputs/ensemble_oof.npy` |
| `verify` | judge-guard-verify | `outputs/guard_report.md` |
| `submit` | kaggle-submit | `outputs/submissions/submission_*.csv` |
| `notebook` | data-analysis-jupyter | cleaned notebooks |

---

## CI/CD (GitHub Actions)

Automated pipeline runs on every push/PR:

```yaml
# Triggers on data/notebook/submission changes
jobs:
  data-audit          # ✅ Data quality gates
  notebook-check      # ✅ Notebook reproducibility
  ensemble-diversity  # ✅ Model diversity check
  submission-guard    # ✅ Submission pre-flight
  judge-guard         # ✅ Output verification
  full-pipeline       # 🌙 Nightly full pipeline
```

---

## Directory Structure

```
qwen-skills/
├── .pre-commit-config.yaml      # Master hook config
├── hooks/                       # Pre-commit hook scripts
│   ├── run_data_audit.py        # → data-audit skill
│   ├── run_submission_guard.py  # → kaggle-submit skill
│   ├── run_judge_guard.py       # → judge-guard-verify skill
│   ├── check_ensemble_diversity.py  # → ml-ensemble skill
│   └── check_notebook.py        # → data-analysis-jupyter skill
├── pipeline/
│   └── orchestrate.py           # Full pipeline runner
├── .github/workflows/
│   └── ml-pipeline.yml          # CI/CD automation
├── IMPLEMENTATION_PLAN.md       # Full architecture plan
└── README.md                    # This file
```

---

## Quick Start

```bash
# 1. Install skills
bash install.sh

# 2. Install pre-commit hooks
pip install pre-commit
pre-commit install

# 3. Run pipeline
python pipeline/orchestrate.py --full

# 4. Check your data
/skills data-audit

# 5. Submit (with all gates enforced)
/skills kaggle-submit
```

---

## License

Apache 2.0
