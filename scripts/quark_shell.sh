#!/bin/bash
# Quark Shell Wrapper
# Provides immediate terminal access with optimized startup

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_ROOT"

# Set up environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export QUARK_HOME="$PROJECT_ROOT"
export QUARK_ENV="production"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Please run: python3 -m venv venv"
    exit 1
fi

# Check if Quark is already running
if [ -f "logs/quark.pid" ]; then
    PID=$(cat logs/quark.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "🤖 Quark is already running (PID: $PID)"
        echo "Use './scripts/quark status' to check status"
        exit 0
    else
        echo "🧹 Cleaning up stale PID file..."
        rm -f logs/quark.pid logs/quark_ready.flag
    fi
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Start Quark with optimized startup
echo "🚀 Starting Quark with optimized startup..."
echo "📁 Project: $PROJECT_ROOT"
echo "🐍 Python: $(which python3)"
echo "📦 Environment: $QUARK_ENV"
echo ""

       # Run the chat interface
       exec python3 scripts/quark_chat.py "$@" 