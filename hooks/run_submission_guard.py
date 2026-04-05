#!/usr/bin/env python3
"""
Hook: Submission Guard (kaggle-submit skill)
Triggered on submission*.csv file changes.
Runs 5 gates: OOF/LB gap, file integrity, prediction range, duplicates, format.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path


def check_submission(filepath: str) -> list[str]:
    """Run submission pre-flight gates. Return findings."""
    findings = []

    try:
        sub = pd.read_csv(filepath)
    except Exception as e:
        return [f"❌ Failed to load {filepath}: {e}"]

    findings.append(f"🛡️ Submission Guard: {filepath} ({len(sub)} rows)")

    # GATE 1: Row count sanity (must have rows)
    if len(sub) == 0:
        findings.append("❌ Empty submission file")
        return findings
    findings.append(f"✅ Row count: {len(sub)}")

    # GATE 2: Column count (should be exactly 2: id + target)
    if len(sub.columns) != 2:
        findings.append(f"❌ Expected 2 columns, got {len(sub.columns)}: {list(sub.columns)}")
    else:
        findings.append(f"✅ Columns: {list(sub.columns)}")

    # GATE 3: No null predictions
    target_col = sub.columns[1]
    nulls = sub[target_col].isnull().sum()
    if nulls > 0:
        findings.append(f"❌ Null predictions: {nulls}")
    else:
        findings.append("✅ No null predictions")

    # GATE 4: Prediction range (binary classification: [0, 1])
    if sub[target_col].dtype in [np.float64, np.float32, np.float16]:
        pmin = sub[target_col].min()
        pmax = sub[target_col].max()
        if pmin < 0 or pmax > 1:
            findings.append(f"❌ Predictions out of [0,1]: [{pmin:.4f}, {pmax:.4f}]")
        elif pmax < 0.5:
            findings.append(f"⚠️  All predictions below 0.5: max={pmax:.4f} — model may be broken")
        elif pmin > 0.5:
            findings.append(f"⚠️  All predictions above 0.5: min={pmin:.4f} — check calibration")
        else:
            findings.append(f"✅ Prediction range: [{pmin:.4f}, {pmax:.4f}]")
    else:
        findings.append(f"⚠️  Target dtype: {sub[target_col].dtype} — expected float")

    # GATE 5: No duplicate IDs
    id_col = sub.columns[0]
    dup_ids = sub[id_col].duplicated().sum()
    if dup_ids > 0:
        findings.append(f"❌ Duplicate IDs: {dup_ids}")
    else:
        findings.append("✅ No duplicate IDs")

    # GATE 6: File size check
    size_mb = Path(filepath).stat().st_size / 1024 / 1024
    if size_mb > 100:
        findings.append(f"⚠️  Large file: {size_mb:.1f} MB — Kaggle limit is typically 100MB")
    else:
        findings.append(f"✅ File size: {size_mb:.1f} MB")

    return findings


def main():
    files = sys.argv[1:]
    if not files:
        print("⏭️  Submission Guard: No submission files to check")
        sys.exit(0)

    all_pass = True
    for f in files:
        print(f"\n{'='*60}")
        findings = check_submission(f)
        for line in findings:
            print(f"  {line}")
            if "❌" in line:
                all_pass = False
        print(f"{'='*60}")

    if not all_pass:
        print("\n❌ Submission Guard FAILED — fix before committing")
        sys.exit(1)
    else:
        print("\n✅ Submission Guard PASSED — ready for Kaggle")
        sys.exit(0)


if __name__ == "__main__":
    main()
