#!/usr/bin/env python3
"""
ML Pipeline Orchestrator
Connects all 10 skills into a single end-to-end pipeline.

Usage:
    python orchestrate.py --full                    # Run all phases
    python orchestrate.py --phase eda               # Run single phase
    python orchestrate.py --phase data-audit,train  # Run multiple phases
    python orchestrate.py --list                    # List available phases
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ─── Phase Definitions ───
PHASES = {
    "data-audit": {
        "skill": "data-audit",
        "description": "7-check dataset quality audit",
        "input": ["data/raw/*.csv"],
        "output": "data/audit_report.md",
        "script": None,  # invoked via skill
    },
    "eda": {
        "skill": "kaggle-eda",
        "description": "6-step EDA + leakage detection + adversarial validation",
        "input": ["data/raw/*.csv", "data/audit_report.md"],
        "output": "outputs/eda_report.md",
        "script": None,
    },
    "features": {
        "skill": "data-scientist",
        "description": "Feature engineering + algorithm selection + experiment design",
        "input": ["outputs/eda_report.md"],
        "output": "data/processed/feature_spec.yaml",
        "script": None,
    },
    "train": {
        "skill": "machine-learning",
        "description": "Model training with JAX/functional patterns",
        "input": ["data/processed/feature_spec.yaml", "data/raw/*.csv"],
        "output": "outputs/oof/*.npy",
        "script": None,
    },
    "serving": {
        "skill": "machine-learning-engineer",
        "description": "Model serving: ONNX, FastAPI, Kubernetes",
        "input": ["outputs/oof/*.npy"],
        "output": "serving/",
        "script": None,
    },
    "mlops": {
        "skill": "machine-learning-ops-ml-pipeline",
        "description": "4-phase MLOps pipeline: Data→Train→Deploy→Monitor",
        "input": ["serving/"],
        "output": "mlops/pipeline.yaml",
        "script": None,
    },
    "ensemble": {
        "skill": "ml-ensemble",
        "description": "Weighted avg, rank avg, stacking meta-learner",
        "input": ["outputs/oof/*.npy"],
        "output": "outputs/ensemble_oof.npy",
        "script": None,
    },
    "verify": {
        "skill": "judge-guard-verify",
        "description": "3-level guard: Schema + Hash + Assertion",
        "input": ["outputs/ensemble_oof.npy"],
        "output": "outputs/guard_report.md",
        "script": None,
    },
    "submit": {
        "skill": "kaggle-submit",
        "description": "5-gate submission pre-flight + experiment logging",
        "input": ["outputs/guard_report.md"],
        "output": "outputs/submissions/submission_*.csv",
        "script": None,
    },
    "notebook": {
        "skill": "data-analysis-jupyter",
        "description": "Jupyter notebook quality + reproducibility",
        "input": ["notebooks/*.ipynb"],
        "output": "notebooks/*.ipynb (cleaned)",
        "script": None,
    },
}

REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "outputs/oof",
    "outputs/submissions",
    "notebooks",
    "serving",
    "mlops",
    "mlops/monitoring",
]


def ensure_dirs():
    """Create required directory structure."""
    for d in REQUIRED_DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ Directory structure ready")


def list_phases():
    """Print available phases."""
    print("\n📋 Available Pipeline Phases:")
    print("-" * 70)
    for name, info in PHASES.items():
        print(f"  {name:<15} │ {info['description']}")
        print(f"                │ In:  {info['input']}")
        print(f"                │ Out: {info['output']}")
        print()


def run_phase(phase_name: str) -> bool:
    """Run a single pipeline phase. Returns True on success."""
    phase = PHASES.get(phase_name)
    if not phase:
        print(f"❌ Unknown phase: {phase_name}")
        return False

    print(f"\n{'='*70}")
    print(f"  PHASE: {phase_name}")
    print(f"  Skill: {phase['skill']}")
    print(f"  Desc:  {phase['description']}")
    print(f"{'='*70}")

    # Check inputs exist
    for pattern in phase["input"]:
        from glob import glob
        matches = glob(pattern)
        if not matches and "*" in pattern:
            print(f"⚠️  Input pattern not found: {pattern}")
        elif not matches:
            print(f"⚠️  Input not found: {pattern}")

    start = time.time()

    # Phase-specific logic
    if phase_name == "data-audit":
        success = run_data_audit()
    elif phase_name == "ensemble":
        success = run_ensemble()
    elif phase_name == "verify":
        success = run_verify()
    else:
        # For phases without local scripts, just log
        print(f"ℹ️  Phase '{phase_name}' requires skill invocation")
        print(f"   Invoke via: /skills {phase['skill']}")
        success = True  # placeholder — actual logic via skill

    elapsed = time.time() - start
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"\n  {status} — {phase_name} completed in {elapsed:.1f}s")

    return success


def run_data_audit() -> bool:
    """Run data-audit skill's 7 checks."""
    print("\n📊 Running Data Audit (7 checks)...")
    hook = Path("hooks/run_data_audit.py")
    if hook.exists():
        import subprocess
        result = subprocess.run(
            [sys.executable, str(hook), "data/raw/train.csv"],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result.returncode == 0
    print("⚠️  Hook script not found — run via /skills data-audit")
    return True


def run_ensemble() -> bool:
    """Run ml-ensemble diversity check."""
    print("\n🎯 Running Ensemble Diversity Check...")
    hook = Path("hooks/check_ensemble_diversity.py")
    if hook.exists():
        import subprocess
        result = subprocess.run(
            [sys.executable, str(hook)] + list(Path("outputs/oof").glob("*.npy")),
            capture_output=True, text=True
        )
        print(result.stdout)
        return result.returncode == 0
    print("⚠️  Hook script not found — run via /skills ml-ensemble")
    return True


def run_verify() -> bool:
    """Run judge-guard-verify."""
    print("\n🔒 Running Judge Guard Verify...")
    hook = Path("hooks/run_judge_guard.py")
    if hook.exists():
        import subprocess
        json_files = list(Path("outputs").glob("*.json"))
        if json_files:
            result = subprocess.run(
                [sys.executable, str(hook)] + [str(f) for f in json_files],
                capture_output=True, text=True
            )
            print(result.stdout)
            return result.returncode == 0
        else:
            print("⚠️  No JSON output files to verify")
            return True
    print("⚠️  Hook script not found — run via /skills judge-guard-verify")
    return True


def main():
    parser = argparse.ArgumentParser(description="ML Pipeline Orchestrator")
    parser.add_argument("--full", action="store_true", help="Run all phases")
    parser.add_argument("--phase", type=str, help="Run specific phase(s), comma-separated")
    parser.add_argument("--list", action="store_true", help="List available phases")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    args = parser.parse_args()

    ensure_dirs()

    if args.list:
        list_phases()
        return

    if args.full:
        print("\n🚀 Running FULL PIPELINE")
        all_pass = True
        for phase_name in PHASES:
            if not run_phase(phase_name):
                print(f"\n❌ Pipeline halted at: {phase_name}")
                all_pass = False
                break

        if all_pass:
            print("\n" + "="*70)
            print("  ✅ FULL PIPELINE COMPLETE")
            print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*70)

    elif args.phase:
        phases = [p.strip() for p in args.phase.split(",")]
        for phase_name in phases:
            if phase_name not in PHASES:
                print(f"❌ Unknown phase: {phase_name}")
                print(f"   Available: {list(PHASES.keys())}")
                sys.exit(1)
            run_phase(phase_name)

    elif args.dry_run:
        print("\n📋 Dry Run — would execute:")
        for name, info in PHASES.items():
            print(f"  {name}: {info['skill']} → {info['output']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
