#!/usr/bin/env python3
"""
Hook: Ensemble Diversity Check (ml-ensemble skill)
Triggered on oof/*.npy file changes.
Checks Pearson correlation between model pairs — r > 0.95 means
the models are too similar and the ensemble won't add value.
"""
import sys
import numpy as np
from pathlib import Path
from itertools import combinations


def check_diversity(oof_dir: str) -> list[str]:
    """Check diversity of all OOF prediction files in directory."""
    findings = []
    oof_path = Path(oof_dir)

    if not oof_path.exists():
        return [f"❌ OOF directory not found: {oof_dir}"]

    npy_files = sorted(oof_path.glob("*.npy"))
    if len(npy_files) < 2:
        findings.append(f"⚠️  Only {len(npy_files)} OOF file — need 2+ for ensemble diversity check")
        return findings

    findings.append(f"🎯 Ensemble Diversity: {len(npy_files)} models found")

    # Load all OOF predictions
    models = {}
    for f in npy_files:
        try:
            data = np.load(f, allow_pickle=True)
            if data.ndim > 1:
                data = data.flatten()
            models[f.stem] = data
        except Exception as e:
            findings.append(f"❌ Failed to load {f.name}: {e}")

    if len(models) < 2:
        findings.append("❌ Less than 2 valid models to compare")
        return findings

    # Check pairwise correlation
    all_pass = True
    names = list(models.keys())
    for i, j in combinations(range(len(names)), 2):
        name_a, name_b = names[i], names[j]
        a, b = models[name_a], models[name_b]

        # Align lengths
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

        if np.std(a) == 0 or np.std(b) == 0:
            findings.append(f"❌ {name_a} vs {name_b}: zero variance — model is constant")
            all_pass = False
            continue

        r = np.corrcoef(a, b)[0, 1]

        if r > 0.99:
            findings.append(f"❌ {name_a} vs {name_b}: r={r:.4f} — IDENTICAL (drop one)")
            all_pass = False
        elif r > 0.95:
            findings.append(f"⚠️  {name_a} vs {name_b}: r={r:.4f} — too similar, minimal ensemble gain")
        elif r > 0.85:
            findings.append(f"✅ {name_a} vs {name_b}: r={r:.4f} — good diversity")
        else:
            findings.append(f"✅ {name_a} vs {name_b}: r={r:.4f} — excellent diversity!")

    # Summary stats
    if all_pass:
        findings.append(f"\n✅ Ensemble diversity check PASSED")
        findings.append(f"💡 Tip: r < 0.95 = good, r < 0.85 = excellent for ensembling")
    else:
        findings.append(f"\n❌ Some model pairs are too correlated — consider replacing similar models")

    return findings


def main():
    files = sys.argv[1:]
    if not files:
        print("⏭️  Ensemble Diversity: No OOF files to check")
        sys.exit(0)

    # Group files by directory
    dirs = set()
    for f in files:
        dirs.add(str(Path(f).parent))

    all_pass = True
    for d in sorted(dirs):
        print(f"\n{'='*60}")
        findings = check_diversity(d)
        for line in findings:
            print(f"  {line}")
            if "❌" in line:
                all_pass = False
        print(f"{'='*60}")

    if not all_pass:
        print("\n❌ Ensemble Diversity FAILED — models too similar")
        sys.exit(1)
    else:
        print("\n✅ Ensemble Diversity PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
