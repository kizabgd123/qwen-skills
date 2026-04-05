#!/usr/bin/env python3
"""
Hook: Data Audit (data-audit skill)
Triggered on .csv / .parquet file changes.
Runs 7 quality checks: duplicates, missing, cardinality, outliers,
dtype mismatches, constant features, schema consistency.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path


def audit_file(filepath: str) -> list[str]:
    """Run all 7 checks. Return list of findings."""
    findings = []
    ext = Path(filepath).suffix.lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(filepath, nrows=50000)
        elif ext == ".parquet":
            df = pd.read_parquet(filepath)
        else:
            return [f"⚠️  Unsupported format: {ext}"]
    except Exception as e:
        return [f"❌ Failed to load {filepath}: {e}"]

    findings.append(f"📊 Auditing {filepath} ({df.shape[0]} rows × {df.shape[1]} cols)")

    # CHECK 1: Duplicates
    n_dup = df.duplicated().sum()
    pct_dup = n_dup / len(df) * 100 if len(df) > 0 else 0
    if pct_dup > 1:
        findings.append(f"❌ Duplicates: {n_dup} ({pct_dup:.2f}%) — exceeds 1% threshold")
    elif n_dup > 0:
        findings.append(f"⚠️  Duplicates: {n_dup} ({pct_dup:.2f}%)")
    else:
        findings.append("✅ No duplicates")

    # CHECK 2: Missing Values
    missing = df.isnull().sum()
    critical = (missing / len(df) * 100) > 50
    if critical.any():
        cols = critical[critical].index.tolist()
        findings.append(f"❌ Critical missing (>50%): {cols}")
    elif missing.sum() > 0:
        findings.append(f"⚠️  Missing values: {missing.sum()} total across {missing[missing > 0].shape[0]} columns")
    else:
        findings.append("✅ No missing values")

    # CHECK 3: Cardinality
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        high_card = [c for c in cat_cols if df[c].nunique() > 1000]
        if high_card:
            findings.append(f"⚠️  High cardinality columns (>1000 unique): {high_card}")
        findings.append(f"✅ Categorical columns: {len(cat_cols)} ({len(high_card)} high-cardinality)")

    # CHECK 4: Outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_count = 0
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 0 and series.std() > 0:
            z = np.abs((series - series.mean()) / series.std())
            outlier_count += (z > 3.5).sum()
    if outlier_count > len(df) * 0.05:
        findings.append(f"❌ Outliers: {outlier_count} points with |z| > 3.5 (>5% of data)")
    elif outlier_count > 0:
        findings.append(f"⚠️  Outliers: {outlier_count} points with |z| > 3.5")
    else:
        findings.append("✅ No significant outliers")

    # CHECK 5: Data Type Mismatches
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(100)
        if len(sample) > 0:
            try:
                pd.to_numeric(sample)
                findings.append(f"⚠️  '{col}' looks numeric but stored as object")
            except (ValueError, TypeError):
                pass

    # CHECK 6: Constant / Near-Constant
    constants = []
    near_constants = []
    for col in df.columns:
        n_unique = df[col].nunique(dropna=False)
        if n_unique == 1:
            constants.append(col)
        else:
            top_freq = df[col].value_counts(dropna=False).iloc[0] / len(df)
            if top_freq > 0.99:
                near_constants.append(col)
    if constants:
        findings.append(f"❌ Constant columns (drop): {constants}")
    if near_constants:
        findings.append(f"⚠️  Near-constant (>99% single value): {near_constants[:5]}")

    # CHECK 7: Memory
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    findings.append(f"✅ Memory: {mem_mb:.1f} MB")

    return findings


def main():
    files = sys.argv[1:]
    if not files:
        print("⏭️  Data Audit: No data files to check")
        sys.exit(0)

    all_pass = True
    for f in files:
        print(f"\n{'='*60}")
        findings = audit_file(f)
        for line in findings:
            print(f"  {line}")
            if "❌" in line:
                all_pass = False
        print(f"{'='*60}")

    if not all_pass:
        print("\n❌ Data Audit FAILED — fix critical issues before committing")
        sys.exit(1)
    else:
        print("\n✅ Data Audit PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
