---
name: judge-guard-verify
description: >
  Enforce Judge_Guard deterministic verification on AI outputs, commits, and agent results.
  Use this skill when the user mentions "judge-guard", "verify output", "hash check",
  "deterministic verification", "guard check", "validate agent output", "commit verification",
  "reproducibility check", or asks to enforce strict output contracts on any AI-generated result.
  This skill implements the Judge_Guard protocol from the judge-guard-core system.
---

# Judge_Guard Verification Protocol

Enforces hash-verified, deterministic output contracts on AI-generated results.
No output is accepted without passing all gates.

---

## Core Concept

Judge_Guard treats every AI output as **untrusted until proven deterministic**.
Verification runs on three levels:

```
Level 1 — Schema Guard    → output matches expected structure
Level 2 — Hash Guard      → output is reproducible (same input → same hash)
Level 3 — Assertion Guard → output passes domain-specific assertions
```

---

## Level 1 — Schema Guard

```python
import json
from typing import Any

def schema_guard(output: Any, expected_schema: dict) -> bool:
    """
    Validate that output matches the expected schema.
    expected_schema: {field_name: expected_type}
    """
    errors = []

    if not isinstance(output, dict):
        raise ValueError(f"Output must be dict, got {type(output)}")

    for field, expected_type in expected_schema.items():
        if field not in output:
            errors.append(f"Missing field: {field}")
            continue
        if not isinstance(output[field], expected_type):
            errors.append(
                f"Field '{field}': expected {expected_type.__name__}, "
                f"got {type(output[field]).__name__}"
            )

    if errors:
        raise AssertionError("Schema Guard FAILED:\n" + "\n".join(f"  - {e}" for e in errors))

    print("✅ Schema Guard passed")
    return True


# Example usage
SUBMISSION_SCHEMA = {
    "id":         str,
    "prediction": float,
    "confidence": float,
    "model_version": str,
}

schema_guard(agent_output, SUBMISSION_SCHEMA)
```

---

## Level 2 — Hash Guard

```python
import hashlib
import json

def compute_output_hash(output: dict, precision: int = 6) -> str:
    """
    Compute a stable hash of the output.
    Floats rounded to `precision` decimals for numerical stability.
    """
    def normalize(obj):
        if isinstance(obj, float):
            return round(obj, precision)
        if isinstance(obj, dict):
            return {k: normalize(v) for k, v in sorted(obj.items())}
        if isinstance(obj, list):
            return [normalize(i) for i in obj]
        return obj

    normalized = normalize(output)
    serialized = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def hash_guard(output: dict, expected_hash: str = None) -> str:
    """
    If expected_hash provided: verify output matches it.
    If not: compute and return hash for registration.
    """
    current_hash = compute_output_hash(output)

    if expected_hash is None:
        print(f"📌 Register this hash: {current_hash}")
        return current_hash

    if current_hash != expected_hash:
        raise AssertionError(
            f"Hash Guard FAILED\n"
            f"  Expected: {expected_hash}\n"
            f"  Got:      {current_hash}\n"
            f"  Output has changed — non-deterministic or tampered."
        )

    print(f"✅ Hash Guard passed: {current_hash}")
    return current_hash


# First run — register hash
registered_hash = hash_guard(agent_output)

# All subsequent runs — verify
hash_guard(agent_output, expected_hash=registered_hash)
```

---

## Level 3 — Assertion Guard

```python
from dataclasses import dataclass
from typing import Callable, List

@dataclass
class Assertion:
    name: str
    fn: Callable
    description: str

def assertion_guard(output: dict, assertions: List[Assertion]) -> bool:
    """
    Run domain-specific assertions against the output.
    All must pass.
    """
    failures = []

    for assertion in assertions:
        try:
            result = assertion.fn(output)
            if not result:
                failures.append(f"{assertion.name}: returned False — {assertion.description}")
        except Exception as e:
            failures.append(f"{assertion.name}: raised {type(e).__name__}: {e}")

    if failures:
        raise AssertionError("Assertion Guard FAILED:\n" + "\n".join(f"  - {f}" for f in failures))

    print(f"✅ Assertion Guard passed ({len(assertions)} assertions)")
    return True


# Example: Kaggle submission assertions
KAGGLE_ASSERTIONS = [
    Assertion(
        name="prediction_range",
        fn=lambda o: 0.0 <= o["prediction"] <= 1.0,
        description="Prediction must be in [0, 1]"
    ),
    Assertion(
        name="confidence_positive",
        fn=lambda o: o["confidence"] > 0,
        description="Confidence must be positive"
    ),
    Assertion(
        name="id_not_empty",
        fn=lambda o: len(o["id"]) > 0,
        description="ID must not be empty"
    ),
]

assertion_guard(agent_output, KAGGLE_ASSERTIONS)
```

---

## Full Verification Pipeline

```python
def judge_guard_verify(output: dict, schema: dict, assertions: list, expected_hash: str = None) -> str:
    """
    Run all three levels. Returns hash on success. Raises on any failure.
    """
    print("Running Judge_Guard verification...")
    schema_guard(output, schema)
    h = hash_guard(output, expected_hash)
    assertion_guard(output, assertions)
    print(f"\n✅ ALL GATES PASSED — hash: {h}")
    return h
```

---

## Commit Hook Integration

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Judge_Guard pre-commit verification
python scripts/verify_outputs.py || {
    echo "❌ Judge_Guard verification failed — commit blocked"
    exit 1
}
echo "✅ Judge_Guard passed — commit allowed"
```

Make executable: `chmod +x .git/hooks/pre-commit`
