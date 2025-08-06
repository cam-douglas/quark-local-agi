#!/usr/bin/env bash
set -e

# ─────────────────────────────────────────────────────────────────────────────
# File: setup_env.sh
# One-time environment setup for Meta-Model AI Assistant
# ─────────────────────────────────────────────────────────────────────────────

echo "🚀 Setting up Meta-Model AI Assistant environment..."

# 0) Go to your project root
cd /Users/camdouglas/meta_model

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION is too old. Need Python $REQUIRED_VERSION+"
    echo "💡 Please install Python 3.10 or later:"
    echo "   brew install python@3.11"
    echo "   or download from https://www.python.org/downloads/"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION is compatible"

# 1) Remove existing venv if it has issues
if [[ -d venv ]]; then
    echo "🗑️  Removing existing virtual environment (may have compatibility issues)..."
    rm -rf venv
fi

# 2) Create fresh venv
echo "📦 Creating fresh virtual environment..."
python3 -m venv venv
echo "✅ Created virtualenv"

# 3) Activate the venv
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# 4) Upgrade pip and install core dependencies
echo "📥 Installing core dependencies..."
pip install --upgrade pip setuptools wheel

# Install dependencies one by one to catch issues
echo "📦 Installing PyTorch..."
pip install torch torchvision torchaudio

echo "📦 Installing Transformers..."
pip install transformers

echo "📦 Installing other ML libraries..."
pip install sentence-transformers spacy chromadb

echo "📦 Installing CLI and utilities..."
pip install click requests prometheus-client

echo "📦 Installing development tools..."
pip install pytest pytest-cov

echo "📦 Installing additional utilities..."
pip install numpy pandas scikit-learn

# 5) Install the package in development mode
echo "🔧 Installing package in development mode..."
pip install -e .

# 6) Download SpaCy English model
echo "📚 Downloading SpaCy English model..."
python3 -m spacy download en_core_web_sm

# 7) Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p memory_db
mkdir -p models

# 8) Preload all HF models to warm the cache
echo "🔥 Preloading models for faster startup..."
python3 scripts/preload_models.py

echo "✅ Environment is fully set up and ready!"
echo ""
echo "🎯 To start the AI assistant, run:"
echo "   python scripts/ai_assistant.sh"
echo ""
echo "🔧 To activate the environment manually:"
echo "   source venv/bin/activate"
echo ""
echo "🔍 To check environment status:"
echo "   python scripts/check_env.py"

