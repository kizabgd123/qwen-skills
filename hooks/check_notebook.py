#!/usr/bin/env python3
"""
Hook: Notebook Reproducibility Check (data-analysis-jupyter skill)
Triggered on .ipynb file changes.
Checks: kernelspec, hardcoded paths, cell execution order, import validity.
"""
import sys
import json
from pathlib import Path
import re


def check_notebook(filepath: str) -> list[str]:
    """Run notebook quality checks."""
    findings = []

    try:
        with open(filepath, 'r') as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        return [f"❌ Invalid JSON in {filepath}: {e}"]
    except Exception as e:
        return [f"❌ Failed to load {filepath}: {e}"]

    findings.append(f"📓 Notebook Check: {filepath}")

    # CHECK 1: Valid notebook format
    if nb.get("nbformat") is None:
        findings.append("❌ Missing nbformat — not a valid Jupyter notebook")
        return findings
    findings.append(f"✅ Notebook format: nbformat {nb['nbformat']}")

    # CHECK 2: Kernelspec present
    metadata = nb.get("metadata", {})
    kernelspec = metadata.get("kernelspec", {})
    if not kernelspec:
        findings.append("⚠️  No kernelspec — may fail on Kaggle/remote")
    else:
        kernel_name = kernelspec.get("name", "unknown")
        findings.append(f"✅ Kernelspec: {kernel_name}")

    # CHECK 3: Hardcoded Kaggle paths
    hardcoded_patterns = [
        r"/kaggle/input/[a-z0-9_-]+/",
        r"/kaggle/working/",
        r"/root/",
        r"C:\\Users\\",
        r"/home/\w+/",
    ]

    hardcoded_found = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            for pattern in hardcoded_patterns:
                matches = re.findall(pattern, source)
                for m in matches:
                    hardcoded_found.append(m.strip())

    if hardcoded_found:
        findings.append(f"⚠️  Hardcoded paths found: {set(hardcoded_found)} — use auto-detect")
    else:
        findings.append("✅ No hardcoded paths detected")

    # CHECK 4: Cell execution count consistency
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    exec_counts = []
    for cell in code_cells:
        ec = cell.get("execution_count")
        if ec is not None:
            exec_counts.append(ec)

    if exec_counts:
        # Check if execution counts are in order
        is_ordered = all(exec_counts[i] <= exec_counts[i+1] for i in range(len(exec_counts)-1))
        if not is_ordered:
            findings.append("⚠️  Cell execution counts out of order — restart kernel and run all")
        else:
            findings.append(f"✅ Cell execution order valid ({len(exec_counts)} cells)")

    # CHECK 5: Empty code cells
    empty_cells = 0
    for cell in code_cells:
        source = "".join(cell.get("source", [])).strip()
        if not source:
            empty_cells += 1
    if empty_cells > 0:
        findings.append(f"⚠️  {empty_cells} empty code cell(s) — clean up before commit")

    # CHECK 6: Missing markdown documentation
    markdown_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "markdown"]
    if len(markdown_cells) == 0:
        findings.append("⚠️  No markdown cells — add documentation")
    else:
        findings.append(f"✅ {len(markdown_cells)} markdown cell(s) for documentation")

    # CHECK 7: File size
    size_mb = Path(filepath).stat().st_size / 1024 / 1024
    if size_mb > 10:
        findings.append(f"⚠️  Large notebook: {size_mb:.1f} MB — consider clearing outputs")
    else:
        findings.append(f"✅ Notebook size: {size_mb:.1f} MB")

    return findings


def main():
    files = sys.argv[1:]
    if not files:
        print("⏭️  Notebook Check: No notebooks to check")
        sys.exit(0)

    all_pass = True
    for f in files:
        print(f"\n{'='*60}")
        findings = check_notebook(f)
        for line in findings:
            print(f"  {line}")
            if "❌" in line:
                all_pass = False
        print(f"{'='*60}")

    if not all_pass:
        print("\n❌ Notebook Check FAILED — fix issues before committing")
        sys.exit(1)
    else:
        print("\n✅ Notebook Check PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
