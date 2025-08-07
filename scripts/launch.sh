#!/usr/bin/env bash
set -e

# ─────────────────────────────────────────────────────────────────────────────
# Quark AI Assistant Launcher
# ─────────────────────────────────────────────────────────────────────────────

echo "🚀 Quark AI Assistant Launcher"
echo "=" * 40

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION is too old. Need Python $REQUIRED_VERSION+"
    echo ""
    echo "💡 To fix this, install Python 3.10 or later:"
    echo ""
    echo "   Option 1: Using Homebrew (recommended)"
    echo "   brew install python@3.11"
    echo ""
    echo "   Option 2: Download from python.org"
    echo "   https://www.python.org/downloads/"
    echo ""
    echo "   After installing, you may need to update your PATH:"
    echo "   export PATH=\"/usr/local/bin:\$PATH\""
    echo ""
    echo "   Then run this script again."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION is compatible"

# Check if virtual environment exists
if [[ ! -d venv ]]; then
    echo "❌ Virtual environment not found."
    echo "🔧 Running setup..."
    ./scripts/setup_env.sh
fi

# Activate the virtual environment
source venv/bin/activate

# Run the startup script which ensures everything is ready
python3 scripts/startup.py 