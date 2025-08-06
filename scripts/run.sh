#!/usr/bin/env bash
# run.sh — launcher for Meta-Model Assistant

# 1) cd into this script’s directory (your project root)
cd "$(dirname "$0")"

# 2) ensure our package is on PYTHONPATH
export PYTHONPATH="$(pwd)"

# 3) activate your venv if you like:
# source venv/bin/activate

# 4) now invoke the packaged CLI
exec python3 -m meta_model.cli "$@"

