#!/usr/bin/env bash
# ============================================================
# install-qwen-skills.sh
# One-shot installer for kizabgd123/qwen-skills
#
# What it does:
#   1. Installs Qwen Code CLI (if not present)
#   2. Clones the skills repo directly into ~/.qwen/skills/
#   3. Verifies every SKILL.md is correctly placed
#   4. Prints a summary of installed skills
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/kizabgd123/qwen-skills/main/install.sh | bash
#
#   Or locally:
#   chmod +x install.sh && ./install.sh
# ============================================================

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

ok()   { echo -e "${GREEN}✓${RESET} $*"; }
warn() { echo -e "${YELLOW}⚠${RESET}  $*"; }
fail() { echo -e "${RED}✗${RESET} $*"; exit 1; }
info() { echo -e "  $*"; }

SKILLS_REPO="https://github.com/kizabgd123/qwen-skills.git"
SKILLS_DIR="$HOME/.qwen/skills"
REPO_DIR="$HOME/.qwen/skills-repo"

SKILLS=(
  kaggle-submit
  kaggle-eda
  ml-ensemble
  judge-guard-verify
  data-audit
)

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   Qwen Code Skills — Installer               ║"
echo "║   github.com/kizabgd123/qwen-skills          ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Step 1: Check / Install Qwen Code ─────────────────────
echo "── Step 1: Qwen Code CLI"

if command -v qwen &>/dev/null; then
  QWEN_VERSION=$(qwen --version 2>/dev/null || echo "unknown")
  ok "Already installed — $QWEN_VERSION"
else
  warn "qwen not found — installing now..."
  if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    fail "Windows detected. Run the installer manually: https://github.com/QwenLM/qwen-code#installation"
  fi
  bash -c "$(curl -fsSL https://qwen-code-assets.oss-cn-hangzhou.aliyuncs.com/installation/install-qwen.sh)"
  ok "Qwen Code installed"
  warn "Restart your terminal after this script finishes to ensure PATH is set correctly"
fi

echo ""

# ── Step 2: Create skills directory ───────────────────────
echo "── Step 2: Skills directory"

mkdir -p "$SKILLS_DIR"
ok "~/.qwen/skills/ ready"

echo ""

# ── Step 3: Clone or update the repo ──────────────────────
echo "── Step 3: Fetching skills from GitHub"

if [[ -d "$REPO_DIR/.git" ]]; then
  info "Repo exists — pulling latest..."
  git -C "$REPO_DIR" pull --quiet
  ok "Updated to latest"
else
  info "Cloning $SKILLS_REPO ..."
  git clone --quiet "$SKILLS_REPO" "$REPO_DIR"
  ok "Cloned successfully"
fi

echo ""

# ── Step 4: Copy each skill into ~/.qwen/skills/ ──────────
echo "── Step 4: Installing skills"

INSTALLED=0
FAILED=0

for skill in "${SKILLS[@]}"; do
  src="$REPO_DIR/$skill"
  dst="$SKILLS_DIR/$skill"

  if [[ ! -d "$src" ]]; then
    warn "Skill not found in repo: $skill"
    ((FAILED++)) || true
    continue
  fi

  if [[ ! -f "$src/SKILL.md" ]]; then
    warn "Missing SKILL.md in: $skill"
    ((FAILED++)) || true
    continue
  fi

  # Copy (overwrite)
  rm -rf "$dst"
  cp -r "$src" "$dst"
  ok "$skill → ~/.qwen/skills/$skill/"
  ((INSTALLED++)) || true
done

echo ""

# ── Step 5: Verify ────────────────────────────────────────
echo "── Step 5: Verification"

ALL_OK=true
for skill in "${SKILLS[@]}"; do
  skill_file="$SKILLS_DIR/$skill/SKILL.md"
  if [[ -f "$skill_file" ]]; then
    NAME=$(grep "^name:" "$skill_file" | head -1 | sed 's/name: //')
    ok "$skill  →  name: $NAME"
  else
    warn "$skill  →  SKILL.md missing"
    ALL_OK=false
  fi
done

echo ""

# ── Step 6: Auth check ────────────────────────────────────
echo "── Step 6: Authentication"

SETTINGS="$HOME/.qwen/settings.json"
if [[ -f "$SETTINGS" ]]; then
  ok "~/.qwen/settings.json found"
else
  warn "~/.qwen/settings.json not found"
  info "Run: qwen"
  info "Then in the session: /auth"
  info "Choose Qwen OAuth (free, 1000 req/day) or paste your API key"
fi

echo ""

# ── Summary ───────────────────────────────────────────────
echo "╔══════════════════════════════════════════════╗"
echo "║   Install complete                           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
info "Installed : $INSTALLED skills"
[[ "$FAILED" -gt 0 ]] && info "Failed    : $FAILED skills"
echo ""
info "── How to use ──────────────────────────────────"
info ""
info "  Start interactive session:"
info "    qwen"
info ""
info "  Explicit skill invoke:"
info "    /skills kaggle-submit"
info "    /skills kaggle-eda"
info "    /skills ml-ensemble"
info "    /skills judge-guard-verify"
info "    /skills data-audit"
info ""
info "  Auto-triggered by natural language:"
info "    \"check my submission\"     → kaggle-submit"
info "    \"do EDA on train.csv\"    → kaggle-eda"
info "    \"blend my 3 models\"      → ml-ensemble"
info "    \"verify agent output\"    → judge-guard-verify"
info "    \"audit the dataset\"      → data-audit"
info ""
info "  Headless (scripts / CI):"
info "    qwen -p \"run pre-flight check on submission.csv\""
info "    qwen -p \"do EDA on data/raw/train.csv and report findings\""
info ""
info "  List all skills:"
info "    ls ~/.qwen/skills/"
info ""
info "  Update skills later:"
info "    git -C ~/.qwen/skills-repo pull"
info "    bash ~/.qwen/skills-repo/install.sh"
echo ""

if $ALL_OK; then
  ok "All skills verified. You're ready."
else
  warn "Some skills had issues — check output above."
fi

echo ""
