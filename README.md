# qwen-skills

Agent Skills for [Qwen Code](https://github.com/QwenLM/qwen-code) — tailored for Kaggle competitions, ML pipelines, and deterministic AI orchestration.

---

## Skills

| Skill | Triggers on |
|---|---|
| `kaggle-submit` | "submit to Kaggle", "pre-flight", "check submission", OOF/LB gap |
| `kaggle-eda` | "do EDA", "explore data", "check for leakage", "data profiling" |
| `ml-ensemble` | "ensemble", "stacking", "blending", "combine models", "meta-learner" |
| `judge-guard-verify` | "judge-guard", "verify output", "hash check", "deterministic verification" |
| `data-audit` | "audit data", "data quality", "check before training", "duplicates", "outliers" |

---

## Install

**Personal (available in all projects):**
```bash
git clone https://github.com/kizabgd123/qwen-skills.git
cp -r qwen-skills/kaggle-submit  ~/.qwen/skills/
cp -r qwen-skills/kaggle-eda     ~/.qwen/skills/
cp -r qwen-skills/ml-ensemble    ~/.qwen/skills/
cp -r qwen-skills/judge-guard-verify ~/.qwen/skills/
cp -r qwen-skills/data-audit     ~/.qwen/skills/
```

**Project-level (shared with team via git):**
```bash
mkdir -p .qwen/skills
cp -r path/to/qwen-skills/kaggle-submit .qwen/skills/
# etc.
```

**Or install all at once:**
```bash
git clone https://github.com/kizabgd123/qwen-skills.git ~/.qwen/skills-repo
for skill in kaggle-submit kaggle-eda ml-ensemble judge-guard-verify data-audit; do
  cp -r ~/.qwen/skills-repo/$skill ~/.qwen/skills/
done
```

---

## Structure

```
qwen-skills/
├── kaggle-submit/
│   └── SKILL.md          pre-submission gates (OOF/LB, integrity, GPU, sample test)
├── kaggle-eda/
│   └── SKILL.md          EDA pipeline: missing, leakage, distribution shift, feature types
├── ml-ensemble/
│   └── SKILL.md          weighted avg, rank avg, stacking with proper OOF meta-features
├── judge-guard-verify/
│   └── SKILL.md          schema guard → hash guard → assertion guard pipeline
└── data-audit/
    └── SKILL.md          7-check audit: duplicates, missing, cardinality, outliers, drift
```

---

## Skill Format

Qwen Code uses the same skill format as Claude Code:

```
skill-name/
└── SKILL.md              ← required
    ├── YAML frontmatter  ← name + description (triggering)
    └── Markdown body     ← instructions, code, examples
```

Skills are **model-invoked** — Qwen Code autonomously decides when to use them based on the description.  
To invoke explicitly: `/skills kaggle-submit`

---

## Requirements

- [Qwen Code](https://github.com/QwenLM/qwen-code) installed (`qwen` CLI)
- Python 3.9+
- `pandas`, `numpy`, `scikit-learn`, `lightgbm` (for Kaggle skills)
- `kaggle` CLI configured with `~/.kaggle/kaggle.json`

---

## License

Apache 2.0
