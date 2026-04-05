---
name: data-audit
description: >
  Systematic data quality audit for any dataset before ML training.
  Use this skill when the user says "audit the data", "check data quality",
  "validate dataset", "data QA", "check before training", "is the data clean",
  "duplicates", "outliers", "data integrity", or uploads any CSV/parquet file
  and asks what's wrong with it. Always complete every check — do not skip outliers
  or cardinality checks just because earlier checks passed.
---

# Data Audit

Seven systematic checks. Run all. Log every finding to `audit_report.md`.

---

## Setup

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("path/to/data.csv")  # or pd.read_parquet(...)

print(f"Shape: {df.shape}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
```

---

## Check 1 — Duplicates

```python
n_dup = df.duplicated().sum()
n_dup_id = df.duplicated(subset=["id"]).sum() if "id" in df.columns else "N/A"

print(f"Full duplicates: {n_dup} ({n_dup / len(df) * 100:.2f}%)")
print(f"ID duplicates:   {n_dup_id}")

if n_dup > 0:
    print("Sample duplicates:")
    print(df[df.duplicated(keep=False)].sort_values(df.columns.tolist()).head(6))
```

---

## Check 2 — Missing Values

```python
missing = pd.DataFrame({
    "count":   df.isnull().sum(),
    "pct":     df.isnull().mean().round(4) * 100,
    "dtype":   df.dtypes,
}).query("count > 0").sort_values("pct", ascending=False)

print(missing.to_string())

# Categorize by severity
critical = missing[missing["pct"] > 50]
moderate = missing[(missing["pct"] > 10) & (missing["pct"] <= 50)]
low      = missing[missing["pct"] <= 10]

print(f"\nCritical (>50%): {len(critical)} cols")
print(f"Moderate (10-50%): {len(moderate)} cols")
print(f"Low (<10%): {len(low)} cols")
```

---

## Check 3 — Cardinality

```python
cat_cols = df.select_dtypes(include=["object", "category"]).columns

cardinality = pd.DataFrame({
    "unique": df[cat_cols].nunique(),
    "total":  len(df),
    "ratio":  (df[cat_cols].nunique() / len(df)).round(4),
})

# High cardinality (quasi-identifier risk or needs embedding)
high = cardinality[cardinality["unique"] > 1000]
# Single value (useless)
single = cardinality[cardinality["unique"] == 1]

print(f"High cardinality (>1000 unique): {len(high)}")
if len(high): print(high)

print(f"\nSingle-value columns (drop): {list(single.index)}")
```

---

## Check 4 — Outliers (Numeric)

```python
from scipy import stats

numeric = df.select_dtypes(include=[np.number]).columns
outlier_report = []

for col in numeric:
    series = df[col].dropna()
    z_scores = np.abs(stats.zscore(series))
    n_outliers = (z_scores > 3.5).sum()
    iqr = series.quantile(0.75) - series.quantile(0.25)
    lower = series.quantile(0.25) - 1.5 * iqr
    upper = series.quantile(0.75) + 1.5 * iqr
    n_iqr = ((series < lower) | (series > upper)).sum()

    if n_outliers > 0 or n_iqr > 0:
        outlier_report.append({
            "column":    col,
            "z>3.5":     n_outliers,
            "iqr_fence": n_iqr,
            "min":       series.min(),
            "max":       series.max(),
            "mean":      series.mean().round(3),
            "std":       series.std().round(3),
        })

pd.DataFrame(outlier_report).sort_values("z>3.5", ascending=False).head(20)
```

---

## Check 5 — Data Type Mismatches

```python
# Detect numerics stored as strings
for col in df.select_dtypes(include="object").columns:
    sample = df[col].dropna().head(100)
    try:
        pd.to_numeric(sample)
        print(f"⚠️  '{col}' looks numeric but stored as object — consider pd.to_numeric()")
    except:
        pass

# Detect dates stored as strings
for col in df.select_dtypes(include="object").columns:
    sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else ""
    if any(x in str(sample) for x in ["-", "/", ":"]) and len(str(sample)) in range(8, 25):
        print(f"⚠️  '{col}' may be datetime — consider pd.to_datetime()")
```

---

## Check 6 — Constant & Near-Constant Features

```python
for col in df.columns:
    n_unique = df[col].nunique(dropna=False)
    top_freq = df[col].value_counts(dropna=False).iloc[0] / len(df)

    if n_unique == 1:
        print(f"🗑️  CONSTANT: '{col}' — drop immediately")
    elif top_freq > 0.99:
        print(f"⚠️  NEAR-CONSTANT: '{col}' — {top_freq:.1%} single value")
```

---

## Check 7 — Schema Consistency (Train vs Test)

```python
def compare_schemas(train: pd.DataFrame, test: pd.DataFrame):
    train_cols = set(train.columns)
    test_cols  = set(test.columns)

    only_train = train_cols - test_cols
    only_test  = test_cols  - train_cols

    if only_train:
        print(f"Columns only in train: {only_train}")
    if only_test:
        print(f"Columns only in test:  {only_test}")

    # Dtype mismatches
    shared = train_cols & test_cols
    for col in shared:
        if train[col].dtype != test[col].dtype:
            print(f"Dtype mismatch '{col}': train={train[col].dtype}, test={test[col].dtype}")

# compare_schemas(train, test)
```

---

## Audit Report Template

```markdown
## Data Audit Report — DATASET_NAME

**Date:** YYYY-MM-DD  
**Shape:** X rows × Y cols  

### Critical Issues (must fix before training)
- [ ] ...

### Warnings (investigate)
- [ ] ...

### Recommendations
- Drop: [col1, col2]
- Convert dtype: [col3 → datetime, col4 → int]
- Impute: [col5 → median, col6 → mode]
- Cap outliers: [col7 at p99]
```
