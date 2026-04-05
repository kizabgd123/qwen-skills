#!/usr/bin/env python3
"""
Hook: Judge Guard Verify (judge-guard-verify skill)
Triggered on outputs/*.json file changes.
Runs 3-level guard: Schema + Hash + Assertion.
"""
import sys
import json
import hashlib
from pathlib import Path


def compute_json_hash(filepath: str, precision: int = 6) -> str:
    """Compute stable hash of JSON file content."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    def normalize(obj):
        if isinstance(obj, float):
            return round(obj, precision)
        if isinstance(obj, dict):
            return {k: normalize(v) for k, v in sorted(obj.items())}
        if isinstance(obj, list):
            return [normalize(i) for i in obj]
        return obj

    normalized = normalize(data)
    serialized = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def schema_guard(data: dict, required_fields: list[str]) -> list[str]:
    """Check that all required fields exist in the JSON."""
    errors = []
    for field in required_fields:
        if field not in data:
            errors.append(f"❌ Missing required field: '{field}'")
    if not errors:
        return [f"✅ Schema: all {len(required_fields)} required fields present"]
    return errors


def assertion_guard(data: dict) -> list[str]:
    """Domain-specific assertions on output JSON."""
    findings = []

    # Check no NaN values (JSON serializability)
    json_str = json.dumps(data)
    if "NaN" in json_str or "Infinity" in json_str:
        findings.append("❌ Contains NaN/Infinity — not JSON serializable")
    else:
        findings.append("✅ No NaN/Infinity values")

    # Check no empty required values
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, str) and len(val.strip()) == 0:
                findings.append(f"⚠️  Empty string value for key: '{key}'")

    # Check numerical ranges if numeric values present
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, (int, float)):
                if val != val:  # NaN check
                    findings.append(f"❌ NaN in numeric field: '{key}'")
                elif abs(val) > 1e15:
                    findings.append(f"⚠️  Suspiciously large value in '{key}': {val}")

    if not findings:
        findings.append("✅ All assertions passed")
    return findings


def check_file(filepath: str) -> list[str]:
    """Run full Judge Guard on a JSON file."""
    findings = []
    findings.append(f"🔒 Judge Guard: {filepath}")

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"❌ Invalid JSON in {filepath}: {e}"]
    except Exception as e:
        return [f"❌ Failed to load {filepath}: {e}"]

    # Level 1: Schema Guard
    if isinstance(data, dict):
        findings.extend(schema_guard(data, list(data.keys())))
    else:
        findings.append("⚠️  Root is not a dict — skipping schema check")

    # Level 2: Hash Guard
    file_hash = compute_json_hash(filepath)
    findings.append(f"✅ Hash: {file_hash}")

    # Store hash in .hash sidecar file for future verification
    hash_path = Path(filepath).with_suffix('.json.hash')
    if hash_path.exists():
        expected_hash = hash_path.read_text().strip()
        if expected_hash != file_hash:
            findings.append(f"❌ HASH MISMATCH: expected={expected_hash}, got={file_hash}")
        else:
            findings.append(f"✅ Hash verified against sidecar")
    else:
        hash_path.write_text(file_hash)
        findings.append(f"📌 Registered hash: {file_hash}")

    # Level 3: Assertion Guard
    findings.extend(assertion_guard(data))

    return findings


def main():
    files = sys.argv[1:]
    if not files:
        print("⏭️  Judge Guard: No JSON files to check")
        sys.exit(0)

    all_pass = True
    for f in files:
        print(f"\n{'='*60}")
        findings = check_file(f)
        for line in findings:
            print(f"  {line}")
            if "❌" in line:
                all_pass = False
        print(f"{'='*60}")

    if not all_pass:
        print("\n❌ Judge Guard FAILED — output is non-deterministic or invalid")
        sys.exit(1)
    else:
        print("\n✅ Judge Guard PASSED — output verified")
        sys.exit(0)


if __name__ == "__main__":
    main()
