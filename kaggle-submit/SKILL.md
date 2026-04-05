---
name: kaggle-submit
description: >
  Pre-flight submission checklist for Kaggle competitions. Use this skill
  whenever the user mentions submitting to Kaggle, generating a submission.csv,
  checking OOF/LB gap, validating a notebook before submission, or any phrase
  like "ready to submit", "submit to Kaggle", "check my submission", "pre-flight",
  or "is this ready". Always run through every gate before allowing a submit.
---

# Kaggle Submission Pre-Flight

Run every gate in order. Do not skip. A failed gate stops the process — fix first, then continue.

---

## Gate 1 — OOF / LB Gap

```python
import numpy as np
from sklearn.metrics import roc_auc_score  # swap for competition metric

oof_score   = roc_auc_score(y_train, oof_predictions)
lb_score    = float(input("Paste your current public LB score: "))
gap         = abs(oof_score - lb_score)
gap_pct     = gap / oof_score * 100

print(f"OOF:  {oof_score:.5f}")
print(f"LB:   {lb_score:.5f}")
print(f"Gap:  {gap:.5f}  ({gap_pct:.1f}%)")

assert gap_pct < 5, f"GAP TOO LARGE ({gap_pct:.1f}%) — investigate leakage or CV mismatch before submitting"
print("✅ Gate 1 passed")
```

**If gap > 5%:** Check for target leakage, incorrect CV strategy (use GroupKFold for time-series), or test-set distribution shift.

---

## Gate 2 — Submission File Integrity

```python
import pandas as pd

sub  = pd.read_csv("submission.csv")
test = pd.read_csv("data/raw/test.csv")

id_col     = "id"          # ← change to actual ID column
target_col = "target"      # ← change to actual target column

# Shape
assert len(sub) == len(test), f"Row count mismatch: {len(sub)} vs {len(test)}"

# Columns
assert id_col in sub.columns,     f"Missing ID column: {id_col}"
assert target_col in sub.columns, f"Missing target column: {target_col}"
assert len(sub.columns) == 2,     f"Extra columns: {list(sub.columns)}"

# No nulls
assert sub[target_col].isnull().sum() == 0, "Null predictions found"

# ID alignment
assert (sub[id_col].values == test[id_col].values).all(), "ID order mismatch"

# Prediction range (classification)
if sub[target_col].between(0, 1).all():
    print(f"Prediction range: [{sub[target_col].min():.4f}, {sub[target_col].max():.4f}]")
    assert sub[target_col].max() > 0.5, "All predictions below 0.5 — model may be broken"

print(f"✅ Gate 2 passed — {len(sub)} rows, {len(sub.columns)} columns")
```

---

## Gate 3 — GPU Sanity (Kaggle Notebooks)

Before committing the notebook:
- [ ] Session type: **GPU P100 or T4 x2** — not CPU
- [ ] Run `!nvidia-smi` in a cell and confirm GPU is available
- [ ] Peak VRAM < 90% of available (check with `!nvidia-smi --query-gpu=memory.used,memory.total --format=csv`)
- [ ] No `CUDA out of memory` in any cell output

---

## Gate 4 — Sample Pipeline Test

Run the full pipeline on a 500-row subset to catch runtime errors:

```python
import pandas as pd

N_SAMPLE = 500
train_sample = pd.read_csv("data/raw/train.csv").sample(N_SAMPLE, random_state=42)
test_sample  = pd.read_csv("data/raw/test.csv").head(N_SAMPLE)

# Run your full pipeline here on train_sample / test_sample
# If it completes without error → Gate 4 passed
print("✅ Gate 4 passed — sample pipeline ran clean")
```

---

## Gate 5 — Submit & Log

```bash
# Kaggle CLI submit
kaggle competitions submit \
  -c COMPETITION_SLUG \
  -f submission.csv \
  -m "MODEL_NAME | CV=0.XXXX | Notes: ..."
```

Immediately log the result:

```csv
# experiment_log.csv row to add:
timestamp,experiment,cv_score,lb_score,cv_lb_delta,model,notes
2025-XX-XX,lgbm_v3,0.8821,0.8794,-0.0027,LightGBM,feature_set_v3 no leaky cols
```

---

## Checklist Summary

- [ ] Gate 1: OOF/LB gap < 5%
- [ ] Gate 2: submission.csv shape, nulls, ID alignment
- [ ] Gate 3: GPU enabled in Kaggle session
- [ ] Gate 4: sample pipeline runs clean
- [ ] Gate 5: submitted + logged in experiment_log.csv
