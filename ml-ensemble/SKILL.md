---
name: ml-ensemble
description: >
  Build production-grade ML ensembles for Kaggle competitions: stacking, blending,
  weighted averaging, and rank averaging. Use this skill when the user mentions
  "ensemble", "stacking", "blending", "combine models", "meta-learner", "level 2 model",
  "weighted average", "model fusion", "rank average", or asks how to combine
  LightGBM / XGBoost / CatBoost predictions. Always enforce proper OOF stacking
  to prevent leakage.
---

# ML Ensemble Builder

Three ensemble strategies ranked by complexity and typical Kaggle gain.
Choose based on time remaining and number of base models.

---

## Strategy 1 — Weighted Average (fastest, +0.001–0.003 AUC typical)

Best when: < 3 models, limited time, models have similar CV scores.

```python
import numpy as np

# Load OOF predictions (out-of-fold on train)
oof_lgbm   = np.load("oof/lgbm_oof.npy")
oof_xgb    = np.load("oof/xgb_oof.npy")
oof_cat    = np.load("oof/catboost_oof.npy")

# Load test predictions
test_lgbm  = np.load("preds/lgbm_test.npy")
test_xgb   = np.load("preds/xgb_test.npy")
test_cat   = np.load("preds/catboost_test.npy")

# Grid search weights on OOF
from sklearn.metrics import roc_auc_score
from itertools import product

best_score, best_weights = 0, (1/3, 1/3, 1/3)

for w1, w2 in product(np.arange(0, 1.05, 0.05), repeat=2):
    w3 = 1 - w1 - w2
    if w3 < 0:
        continue
    blend = w1 * oof_lgbm + w2 * oof_xgb + w3 * oof_cat
    score = roc_auc_score(y_train, blend)
    if score > best_score:
        best_score, best_weights = score, (w1, w2, w3)

w1, w2, w3 = best_weights
print(f"Best weights: LGBM={w1:.2f}  XGB={w2:.2f}  CAT={w3:.2f}")
print(f"Ensemble OOF AUC: {best_score:.5f}")

test_blend = w1 * test_lgbm + w2 * test_xgb + w3 * test_cat
```

---

## Strategy 2 — Rank Averaging (robust to scale differences)

Best when: models predict on different scales, or mixing classifiers with regressors.

```python
from scipy.stats import rankdata

def rank_avg(*arrays):
    ranks = [rankdata(a) / len(a) for a in arrays]
    return np.mean(ranks, axis=0)

oof_blend  = rank_avg(oof_lgbm, oof_xgb, oof_cat)
test_blend = rank_avg(test_lgbm, test_xgb, test_cat)

score = roc_auc_score(y_train, oof_blend)
print(f"Rank average OOF AUC: {score:.5f}")
```

---

## Strategy 3 — Stacking / Meta-Learner (most powerful, +0.002–0.008 AUC typical)

Best when: 4+ diverse base models, enough time for full CV.

**Critical rule: OOF meta-features only — never train on full train predictions.**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# Base model OOF predictions become features for the meta-learner
meta_train = np.column_stack([oof_lgbm, oof_xgb, oof_cat])  # shape: (n_train, n_models)
meta_test  = np.column_stack([test_lgbm, test_xgb, test_cat])

# Optional: add raw features to meta-learner
# meta_train = np.column_stack([meta_train, X_train_raw])
# meta_test  = np.column_stack([meta_test, X_test_raw])

# Meta-learner CV (same folds as base models)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_oof   = np.zeros(len(meta_train))
meta_preds = np.zeros(len(meta_test))

for fold, (tr_idx, val_idx) in enumerate(cv.split(meta_train, y_train)):
    X_tr, X_val = meta_train[tr_idx], meta_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    # Simple meta-learner — LogReg or LightGBM
    meta = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    meta.fit(X_tr, y_tr)

    meta_oof[val_idx] = meta.predict_proba(X_val)[:, 1]
    meta_preds        += meta.predict_proba(meta_test)[:, 1] / 5

stack_score = roc_auc_score(y_train, meta_oof)
print(f"Stacking OOF AUC: {stack_score:.5f}")
```

---

## Diversity Checklist

Before building an ensemble, verify model diversity:

```python
from scipy.stats import pearsonr

pairs = [
    ("LGBM vs XGB", oof_lgbm, oof_xgb),
    ("LGBM vs CAT", oof_lgbm, oof_cat),
    ("XGB vs CAT",  oof_xgb, oof_cat),
]

for name, a, b in pairs:
    r, _ = pearsonr(a, b)
    print(f"{name}: r={r:.3f}", "✅ diverse" if r < 0.95 else "⚠️  too similar")
```

If r > 0.95 between two models — the second adds negligible value. Drop it or replace with a genuinely different architecture (neural net, linear model, tree with different feature set).

---

## Save & Version

```python
import pandas as pd
from datetime import datetime

ts  = datetime.now().strftime("%Y%m%d_%H%M")
sub = pd.DataFrame({"id": test_ids, "target": test_blend})
sub.to_csv(f"submissions/{ts}_ensemble_cv{best_score:.4f}.csv", index=False)
print(f"Saved: submissions/{ts}_ensemble_cv{best_score:.4f}.csv")
```
