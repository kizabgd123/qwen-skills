---
name: kaggle-eda
description: >
  Systematic Exploratory Data Analysis for Kaggle competitions. Use this skill
  whenever the user says "do EDA", "explore the data", "analyze the dataset",
  "what's in the data", "check for leakage", "understand the features",
  "data profiling", or uploads/references a train.csv for a competition.
  Always run the full pipeline — never skip leakage check.
---

# Kaggle EDA Pipeline

Run in order. Every section outputs findings to `eda_report.md` for session persistence.

---

## Step 1 — Load & Shape Overview

```python
import pandas as pd
import numpy as np

train = pd.read_csv("data/raw/train.csv")
test  = pd.read_csv("data/raw/test.csv")

TARGET = "target"  # ← set this
ID_COL = "id"      # ← set this

print(f"Train: {train.shape}  |  Test: {test.shape}")
print(f"Target distribution:\n{train[TARGET].value_counts(normalize=True).round(3)}")
print(f"\nDtype breakdown:\n{train.dtypes.value_counts()}")
```

---

## Step 2 — Missing Values

```python
missing = pd.DataFrame({
    "train_missing":     train.isnull().sum(),
    "train_missing_pct": train.isnull().mean().round(3) * 100,
    "test_missing":      test.isnull().sum(),
    "test_missing_pct":  test.isnull().mean().round(3) * 100,
})
missing = missing[missing["train_missing"] > 0].sort_values("train_missing_pct", ascending=False)
print(missing)

# Flag columns with high missing in test but not train — potential leakage source
suspicious = missing[(missing["test_missing_pct"] > 50) & (missing["train_missing_pct"] < 10)]
if len(suspicious):
    print(f"\n🚨 SUSPICIOUS (high test missing, low train missing):\n{suspicious}")
```

---

## Step 3 — Target Analysis

```python
import matplotlib.pyplot as plt

# Classification
if train[TARGET].nunique() <= 20:
    print("Class counts:")
    print(train[TARGET].value_counts())
    imbalance_ratio = train[TARGET].value_counts().max() / train[TARGET].value_counts().min()
    if imbalance_ratio > 5:
        print(f"\n⚠️  Imbalance ratio: {imbalance_ratio:.1f}x — use StratifiedKFold + class weights")

# Regression
else:
    print(train[TARGET].describe())
    skew = train[TARGET].skew()
    if abs(skew) > 1:
        print(f"\n⚠️  Target skew: {skew:.2f} — consider log1p transform")
```

---

## Step 4 — Leakage Detection

```python
# Correlation-based leakage check
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
if TARGET in numeric_cols:
    corr = train[numeric_cols].corr()[TARGET].abs().sort_values(ascending=False)
    high = corr[(corr > 0.9) & (corr.index != TARGET)]
    if len(high):
        print(f"🚨 HIGH CORRELATION WITH TARGET (possible leakage):\n{high}")
    else:
        print("✅ No obvious numeric leakage detected")

# ID-based leakage: does ID correlate with target?
if ID_COL in train.columns:
    id_corr = train[ID_COL].corr(train[TARGET])
    if abs(id_corr) > 0.1:
        print(f"🚨 ID correlates with target ({id_corr:.3f}) — do NOT use as feature")
```

---

## Step 5 — Train/Test Distribution Shift

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# If a model can distinguish train from test — there's distribution shift
train_test = pd.concat([
    train[numeric_cols].drop(columns=[TARGET], errors="ignore").assign(_is_test=0),
    test[numeric_cols].assign(_is_test=1)
], ignore_index=True).fillna(-999)

X_adv = train_test.drop(columns=["_is_test"])
y_adv = train_test["_is_test"]

rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
auc = cross_val_score(rf, X_adv, y_adv, cv=3, scoring="roc_auc").mean()
print(f"Adversarial AUC (train vs test): {auc:.3f}")
if auc > 0.7:
    importances = pd.Series(rf.fit(X_adv, y_adv).feature_importances_, index=X_adv.columns)
    print(f"⚠️  Shift detected. Top shifted features:\n{importances.sort_values(ascending=False).head(10)}")
else:
    print("✅ No significant distribution shift")
```

---

## Step 6 — Feature Type Summary

```python
# Categorize each feature
high_card_cats = []
low_card_cats  = []
numerics       = []
datetime_cols  = []
text_cols      = []

for col in train.drop(columns=[TARGET, ID_COL], errors="ignore").columns:
    dtype = train[col].dtype
    nuniq = train[col].nunique()
    if dtype == "object":
        if nuniq > 50:
            high_card_cats.append((col, nuniq))
        else:
            low_card_cats.append((col, nuniq))
    elif "datetime" in str(dtype):
        datetime_cols.append(col)
    else:
        numerics.append(col)

print(f"Numeric: {len(numerics)}")
print(f"Low-cardinality categorical (<= 50): {len(low_card_cats)}")
print(f"High-cardinality categorical (> 50): {len(high_card_cats)}")
print(f"Datetime: {len(datetime_cols)}")
```

---

## EDA Report Template

Save findings to `eda_report.md`:

```markdown
## EDA Summary

**Train shape:** X rows × Y cols  
**Test shape:**  X rows × Y cols  
**Target:** binary / multiclass / regression  
**Imbalance:** N:1 ratio  

### Red Flags
- [ ] Leakage: ...
- [ ] Distribution shift: ...
- [ ] Suspicious missing: ...

### Feature Engineering Opportunities
- ...

### Recommended CV Strategy
- StratifiedKFold / GroupKFold / TimeSeriesSplit
- n_splits = 5, random_state = 42
```
