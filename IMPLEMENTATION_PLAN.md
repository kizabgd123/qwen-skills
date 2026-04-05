# 🔗 MASTER IMPLEMENTATION PLAN — Skills + Hooks Integration

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    FULL ML LIFECYCLE — HOOKED PIPELINE                       │
│                                                                              │
│  DATA → EDA → SCIENCE → TRAIN → ENSEMBLE → DEPLOY → VERIFY → SUBMIT         │
│    │       │        │         │          │         │        │         │      │
│    ▼       ▼        ▼         ▼          ▼         ▼        ▼         ▼      │
│  data   kaggle   data    machine-    ml-     machine-  judge-   kaggle       │
│  audit   -eda   scientist learning  ensemble learning  guard    -submit      │
│                         engineer    verify   -ops-ml                         │
│                                     │        pipeline                        │
│                                     ▼                                        │
│                              .pre-commit-config.yaml                         │
│                              (automated gates on every commit)               │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Installed Skills Inventory

| # | Skill | Category | Purpose |
|---|-------|----------|---------|
| 1 | `data-audit` | Data Quality | 7-check dataset audit (duplicates, missing, outliers, schema) |
| 2 | `kaggle-eda` | Exploration | 6-step EDA + leakage detection + distribution shift |
| 3 | `data-scientist` | ML Science | Algorithm selection, feature engineering, A/B tests |
| 4 | `machine-learning` | ML Core | JAX/functional ML patterns, training loops |
| 5 | `machine-learning-engineer` | MLE | Model serving, ONNX, FastAPI, Kubernetes deployment |
| 6 | `machine-learning-ops-ml-pipeline` | MLOps | 4-phase ML pipeline orchestration |
| 7 | `ml-ensemble` | Ensemble | Weighted avg, rank avg, stacking |
| 8 | `judge-guard-verify` | Verification | Schema/Hash/Assertion guards |
| 9 | `kaggle-submit` | Submission | 5-gate pre-flight + experiment logging |
| 10 | `data-analysis-jupyter` | Analysis | Pandas, matplotlib, notebook best practices |

---

## Phase 1: Pre-Commit Hooks Infrastructure

### 1.1 Create `.pre-commit-config.yaml`

Hooks wired to skill domains:

```yaml
repos:
  # ─── FORMAT & LINT (universal) ───
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: check-added-large-files    # block >500MB files
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json

  # ─── PYTHON CODE QUALITY ───
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.9.0
    hooks:
      - id: ruff           # lint + auto-fix
      - id: ruff-format    # formatter

  # ─── NOTEBOOK QUALITY (data-analysis-jupyter skill) ───
  - repo: https://github.com/nbQA-dev/nbQA
    rev: 1.9.1
    hooks:
      - id: nbqa-ruff
      - id: nbqa-check-ast

  # ─── SECRET DETECTION (always-on) ───
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.22.0
    hooks:
      - id: gitleaks

  # ─── SKILL-TRIGGERED HOOKS (custom scripts) ───
  - repo: local
    hooks:
      # DATA AUDIT: triggers on CSV/parquet changes
      - id: data-audit
        name: "📊 Data Audit (data-audit skill)"
        entry: scripts/hooks/run_data_audit.py
        language: python
        files: \.(csv|parquet)$
        additional_dependencies: [pandas, numpy, scipy]

      # SUBMISSION GUARD: triggers on submission*.csv
      - id: submission-guard
        name: "🛡️ Submission Guard (kaggle-submit skill)"
        entry: scripts/hooks/run_submission_guard.py
        language: python
        files: submission.*\.csv$
        additional_dependencies: [pandas, sklearn]

      # JUDGE GUARD VERIFY: triggers on JSON output files
      - id: judge-guard
        name: "🔒 Judge Guard Verify (judge-guard-verify skill)"
        entry: scripts/hooks/run_judge_guard.py
        language: python
        files: outputs/.*\.json$
        additional_dependencies: []

      # ENSEMBLE DIVERSITY: triggers on OOF prediction files
      - id: ensemble-diversity
        name: "🎯 Ensemble Diversity (ml-ensemble skill)"
        entry: scripts/hooks/check_ensemble_diversity.py
        language: python
        files: oof/.*\.npy$
        additional_dependencies: [numpy, scipy]

      # NOTEBOOK REPRODUCIBILITY: triggers on .ipynb changes
      - id: notebook-check
        name: "📓 Notebook Reproducibility (data-analysis-jupyter skill)"
        entry: scripts/hooks/check_notebook.py
        language: python
        files: \.ipynb$
        additional_dependencies: [nbformat]

  # ─── COMMIT MESSAGE FORMAT ───
  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: v4.0.0
    hooks:
      - id: conventional-pre-commit
        stages: [commit-msg]
```

### 1.2 Custom Hook Scripts

Each script invokes its corresponding skill's logic:

| Script | Skill It Invokes | Trigger |
|--------|-----------------|---------|
| `scripts/hooks/run_data_audit.py` | `data-audit` | Any `.csv`/`.parquet` change |
| `scripts/hooks/run_submission_guard.py` | `kaggle-submit` | `submission*.csv` change |
| `scripts/hooks/run_judge_guard.py` | `judge-guard-verify` | `outputs/*.json` change |
| `scripts/hooks/check_ensemble_diversity.py` | `ml-ensemble` | `oof/*.npy` change |
| `scripts/hooks/check_notebook.py` | `data-analysis-jupyter` | `.ipynb` change |

---

## Phase 2: Skill Wiring — Data Flow Pipeline

### 2.1 Pipeline Orchestration Script

```
scripts/pipeline/orchestrate.py
```

Connects all 10 skills in order:

```
┌─ INPUT ──────────────────────────────────────────────────────────────┐
│  train.csv, test.csv → data-audit (Skill #1)                         │
│  → 7 checks: duplicates, missing, outliers, cardinality,             │
│     dtype mismatches, constant features, schema consistency           │
│  → audit_report.md                                                    │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ EXPLORATION ────────────────────────────────────────────────────────┐
│  audit_report.md → kaggle-eda (Skill #2)                             │
│  → 6 steps: load/shape, missing, target, leakage, shift, types       │
│  → eda_report.md + leakage flags + adversarial AUC                    │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ FEATURE ENGINEERING ────────────────────────────────────────────────┐
│  eda_report.md → data-scientist (Skill #3)                           │
│  → Algorithm selection matrix, feature transforms, experiment design │
│  → feature_spec.yaml + experiment_tracker.json                       │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ MODEL TRAINING ─────────────────────────────────────────────────────┐
│  feature_spec.yaml → machine-learning (Skill #4)                     │
│  → JAX/functional training loops, JIT compilation,                   │
│     functional programming patterns, memory management                │
│  → trained_models/ + training_metrics.json                            │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ MODEL SERVING ──────────────────────────────────────────────────────┐
│  trained_models/ → machine-learning-engineer (Skill #5)              │
│  → ONNX conversion, FastAPI/gRPC serving, auto-scaling,              │
│     quantization (FP32→INT8), Kubernetes manifests                     │
│  → serving/ + docker-compose.yml + k8s/                               │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ MLOps PIPELINE ─────────────────────────────────────────────────────┐
│  serving/ → machine-learning-ops-ml-pipeline (Skill #6)              │
│  → 4-phase orchestration: Data→Train→Deploy→Monitor                   │
│  → MLflow/W&B tracking, Feast feature store,                          │
│     KServe serving, Prometheus monitoring                             │
│  → pipeline.yaml + monitoring/ + ci-cd/                               │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ ENSEMBLE ───────────────────────────────────────────────────────────┐
│  trained_models/ → ml-ensemble (Skill #7)                            │
│  → Weighted average, rank averaging, stacking meta-learner            │
│  → Diversity check (Pearson r < 0.95 between models)                 │
│  → ensemble_oof.npy + ensemble_test.npy + blend_weights.json          │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ VERIFICATION ───────────────────────────────────────────────────────┐
│  ensemble_oof.npy → judge-guard-verify (Skill #8)                    │
│  → 3-level guard: Schema + Hash + Assertion                          │
│  → verification_hash.txt + guard_report.md                            │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ SUBMISSION ─────────────────────────────────────────────────────────┐
│  guard_report.md → kaggle-submit (Skill #9)                          │
│  → 5-gate pre-flight: OOF/LB gap, file integrity, GPU,               │
│     sample pipeline test, experiment logging                          │
│  → submission_YYYYMMDD_HHMM.csv + experiment_log.csv                  │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 Shared Data Contracts

All skills pass typed artifacts between phases:

| From → To | Artifact | Schema |
|-----------|----------|--------|
| data-audit → kaggle-eda | `audit_report.md` | Markdown with ✅/❌ flags |
| kaggle-eda → data-scientist | `eda_report.md` | Shape, leakage flags, adversarial AUC |
| data-scientist → machine-learning | `feature_spec.yaml` | Feature list + transforms |
| machine-learning → mlops | `training_metrics.json` | `{model, cv_score, params}` |
| mlops → ml-ensemble | `oof/*.npy` | Per-model OOF predictions |
| ml-ensemble → judge-guard | `ensemble_oof.npy` | Blended predictions |
| judge-guard → kaggle-submit | `guard_report.md` | Hash verification + assertion pass |
| kaggle-submit → FINAL | `submission_*.csv` | `{id, target}` format |

---

## Phase 3: Directory Structure

```
project-root/
├── .pre-commit-config.yaml          # Master hook config
├── skills/                          # Skill definitions (symlinks to ~/.qwen/skills)
│   ├── data-audit → ~/.qwen/skills/data-audit/
│   ├── kaggle-eda → ~/.qwen/skills/kaggle-eda/
│   ├── data-scientist → ~/.qwen/skills/data-scientist/
│   ├── machine-learning → ~/.qwen/skills/machine-learning/
│   ├── machine-learning-engineer → ~/.qwen/skills/machine-learning-engineer/
│   ├── machine-learning-ops-ml-pipeline → ~/.qwen/skills/machine-learning-ops-ml-pipeline/
│   ├── ml-ensemble → ~/.qwen/skills/ml-ensemble/
│   ├── judge-guard-verify → ~/.qwen/skills/judge-guard-verify/
│   ├── kaggle-submit → ~/.qwen/skills/kaggle-submit/
│   └── data-analysis-jupyter → ~/.qwen/skills/data-analysis-jupyter/
├── scripts/
│   ├── hooks/                       # Pre-commit hook scripts
│   │   ├── run_data_audit.py
│   │   ├── run_submission_guard.py
│   │   ├── run_judge_guard.py
│   │   ├── check_ensemble_diversity.py
│   │   └── check_notebook.py
│   ├── pipeline/                    # Main pipeline orchestration
│   │   ├── orchestrate.py           # Full pipeline runner
│   │   ├── feature_engineer.py      # data-scientist feature logic
│   │   ├── train.py                 # machine-learning training loop
│   │   ├── ensemble.py              # ml-ensemble blending
│   │   └── verify.py                # judge-guard-verify runner
│   ├── serving/                     # MLE deployment artifacts
│   │   ├── api.py                   # FastAPI serving endpoint
│   │   ├── onnx_export.py           # Model conversion
│   │   └── docker-compose.yml
│   └── mlops/                       # MLOps infrastructure
│       ├── mlflow_setup.py
│       ├── pipeline.yaml
│       └── monitoring/
│           ├── prometheus.yml
│           └── grafana_dashboards/
├── data/
│   ├── raw/                         # Original train.csv, test.csv
│   ├── processed/                   # Cleaned/feature-engineered data
│   └── audit_report.md              # Generated by data-audit
├── outputs/
│   ├── eda_report.md                # Generated by kaggle-eda
│   ├── oof/                         # OOF predictions per model
│   ├── submissions/                 # Final submission files
│   └── guard_report.md              # Generated by judge-guard-verify
├── notebooks/                       # Jupyter notebooks
├── experiment_log.csv               # All experiments tracked here
└── README.md                        # Project overview
```

---

## Phase 4: Hook Script Implementations

### 4.1 `scripts/hooks/run_data_audit.py`
Invokes `data-audit` skill's 7 checks:
- Duplicates → ❌ if > 1%
- Missing values → ❌ if critical > 50%
- Cardinality → ⚠️ if > 1000 unique on object cols
- Outliers → ⚠️ if z > 3.5 count > 5%
- Data type mismatches → ❌ if numeric stored as string
- Constant features → ❌ if n_unique == 1
- Schema consistency → ❌ if train/test columns differ

### 4.2 `scripts/hooks/run_submission_guard.py`
Invokes `kaggle-submit` skill's 5 gates:
- Gate 1: OOF/LB gap < 5%
- Gate 2: submission.csv shape, nulls, ID alignment
- Gate 3: GPU availability check
- Gate 4: Sample pipeline test
- Gate 5: Experiment log entry

### 4.3 `scripts/hooks/run_judge_guard.py`
Invokes `judge-guard-verify` skill's 3 levels:
- Level 1: Schema guard (structure match)
- Level 2: Hash guard (determinism check)
- Level 3: Assertion guard (domain assertions)

### 4.4 `scripts/hooks/check_ensemble_diversity.py`
Invokes `ml-ensemble` skill's diversity check:
- Pearson r between all model pairs
- ❌ if r > 0.95 (too similar, no ensemble value)

### 4.5 `scripts/hooks/check_notebook.py`
Invokes `data-analysis-jupyter` skill's reproducibility:
- Check kernelspec exists
- Check no hardcoded paths
- Check cell execution order is valid
- Check imports are standard

---

## Phase 5: Automation & CI/CD

### 5.1 GitHub Actions Workflow

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  data-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python scripts/hooks/run_data_audit.py

  model-train:
    needs: data-audit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python scripts/pipeline/train.py

  ensemble:
    needs: model-train
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python scripts/hooks/check_ensemble_diversity.py
      - run: python scripts/pipeline/ensemble.py

  verify:
    needs: ensemble
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python scripts/hooks/run_judge_guard.py
```

### 5.2 Local Development Workflow

```bash
# Full pipeline run
python scripts/pipeline/orchestrate.py --full

# Single phase
python scripts/pipeline/orchestrate.py --phase eda
python scripts/pipeline/orchestrate.py --phase train
python scripts/pipeline/orchestrate.py --phase ensemble

# Pre-commit hooks
pre-commit install
pre-commit run --all-files     # First-time scan
pre-commit run data-audit      # Run single hook

# Verify & submit
python scripts/pipeline/verify.py && kaggle submit
```

---

## Phase 6: Skill-to-Agent Mapping

| Skill | Matching Qwen Agent | Handoff Protocol |
|-------|-------------------|------------------|
| data-audit | `kaggle-forensics` | audit_report.md |
| kaggle-eda | `kaggle-eda` | eda_report.md |
| data-scientist | `kaggle-fe` | feature_spec.yaml |
| machine-learning | `kaggle-nn` | training_metrics.json |
| machine-learning-engineer | `code-architect` | serving/ |
| machine-learning-ops-ml-pipeline | `trinity-orchestrator` | pipeline.yaml |
| ml-ensemble | `kaggle-ensemble` | ensemble_oof.npy |
| judge-guard-verify | `kaggle-judge` | guard_report.md |
| kaggle-submit | `kaggle-submit-safe` | submission_*.csv |
| data-analysis-jupyter | `code-jupyter` | notebooks/ |

---

## Execution Order

1. ✅ **Install pre-commit** → `pip install pre-commit`
2. ✅ **Create .pre-commit-config.yaml** → Phase 1
3. ✅ **Create hook scripts** → Phase 4
4. ✅ **Create pipeline scripts** → Phase 2
5. ✅ **Create directory structure** → Phase 3
6. ✅ **Wire skill-to-agent mapping** → Phase 6
7. ✅ **Set up CI/CD** → Phase 5
8. ✅ **Run full pipeline test** → `python scripts/pipeline/orchestrate.py --full`

---

## Success Criteria

- [ ] Every commit triggers relevant skill-based hooks
- [ ] Data changes auto-trigger data-audit → eda pipeline
- [ ] Model OOF files auto-trigger ensemble diversity check
- [ ] Submission files auto-trigger kaggle-submit 5-gate check
- [ ] All outputs pass judge-guard-verify (schema + hash + assertions)
- [ ] Experiment log tracks every run with params + metrics + hash
- [ ] Pre-commit completes in < 10s for normal commits
- [ ] Full pipeline runs in < 30min for s6e3 dataset
