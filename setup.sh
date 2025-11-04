#!/usr/bin/env bash
set -euo pipefail

# setup.sh - Create virtualenv, install Python deps, and optionally set up Ollama (gpt-oss:20b)
# Usage:
#   ./setup.sh            # create venv, install deps, attempt to setup Ollama if available
#   ./setup.sh --skip-ollama  # skip Ollama installation/pull (safe for CI/local)
#   ./setup.sh --ollama-port 11435  # set custom Ollama HTTP port

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

SKIP_OLLAMA=0
OLLAMA_PORT=11434

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-ollama)
      SKIP_OLLAMA=1
      shift
      ;;
    --ollama-port)
      OLLAMA_PORT="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "[setup] Starting setup in $SCRIPT_DIR"

# 1) Create virtualenv
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BIN=${PYTHON_BIN:-python3}

if [[ -d "$VENV_DIR" ]]; then
  echo "[setup] Virtualenv already exists at $VENV_DIR. Skipping creation." 
else
  echo "[setup] Creating virtualenv at $VENV_DIR using $PYTHON_BIN"
  $PYTHON_BIN -m venv "$VENV_DIR"
fi

# 2) Activate venv and upgrade pip
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "[setup] Upgrading pip, setuptools, wheel"
python -m pip install --upgrade pip setuptools wheel

# 3) Install python requirements
REQ_FILE="$SCRIPT_DIR/requirements.txt"
if [[ -f "$REQ_FILE" ]]; then
  echo "[setup] Installing Python packages from $REQ_FILE"
  pip install -r "$REQ_FILE"
else
  echo "[setup] No requirements.txt found at $REQ_FILE"
fi

# 4) Optional: install Ollama (macOS/Homebrew) and pull gpt-oss:20b
if [[ "$SKIP_OLLAMA" -eq 1 ]]; then
  echo "[setup] Skipping Ollama installation/pull as requested (--skip-ollama)"
else
  if command -v ollama >/dev/null 2>&1; then
    echo "[setup] ollama CLI found"
  else
    if command -v brew >/dev/null 2>&1; then
      echo "[setup] ollama not found. Installing via Homebrew..."
      brew install ollama || { echo "[setup] brew install ollama failed. Please install Ollama manually: https://ollama.com/docs"; }
    else
      echo "[setup] brew not found and ollama not installed. Please install Ollama manually: https://ollama.com/docs";
    fi
  fi

  if command -v ollama >/dev/null 2>&1; then
    echo "[setup] Pulling model gpt-oss:20b (this may take time and disk space)"
    ollama pull gpt-oss:20b || echo "[setup] ollama pull failed or model already available"
    echo "[setup] Ensure Ollama daemon is running. To serve HTTP API, Ollama usually listens on port 11434 by default."
    echo "[setup] You can run a model with: ollama run gpt-oss:20b --http"
    echo "[setup] We'll set OLLAMA_URL to http://localhost:$OLLAMA_PORT in your environment instructions."
  fi
fi

# 5) Create a helper env file with recommended exports
ENV_FILE="$SCRIPT_DIR/.env.local"
cat > "$ENV_FILE" <<EOF
# Local environment for System-Dynamics-Bot
# Activate venv first: source .venv/bin/activate
export USE_OLLAMA=1
export OLLAMA_URL="http://localhost:$OLLAMA_PORT"
# If you prefer OpenAI instead:
# export USE_OLLAMA=0
# export OPENAI_API_KEY="sk-..."
EOF

echo "[setup] Wrote helper env file to $ENV_FILE"

echo "[setup] Done. Quick start:
  source .venv/bin/activate
  export USE_OLLAMA=1
  export OLLAMA_URL=\"http://localhost:$OLLAMA_PORT\"
  cd cld
  python main.py --verbose
"

exit 0
