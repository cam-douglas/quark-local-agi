#!/usr/bin/env bash
set -e

# ─────────────────────────────────────────────────────────────────────────────
# Quark AI Assistant Launcher
# ─────────────────────────────────────────────────────────────────────────────

cd /Users/camdouglas/quark

# Check if virtual environment exists
if [[ ! -d venv ]]; then
    echo "❌ Virtual environment not found. Running setup..."
    ./scripts/setup_env.sh
fi

# Activate the virtual environment
source venv/bin/activate

# Run the startup script which ensures everything is ready
python3 scripts/startup.py

