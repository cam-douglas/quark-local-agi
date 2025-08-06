#!/bin/bash
# Meta-Model AI Assistant Startup Daemon
# This script ensures the AI assistant is ready before completing startup

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Meta-Model project directory
META_MODEL_DIR="/Users/camdouglas/meta_model"
STARTUP_LOG="$META_MODEL_DIR/logs/startup.log"
READY_FLAG="$META_MODEL_DIR/logs/model_ready.flag"
PID_FILE="$META_MODEL_DIR/logs/meta_model.pid"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$STARTUP_LOG"
}

# Function to check if model is ready
check_model_ready() {
    if [[ -f "$READY_FLAG" ]]; then
        return 0
    fi
    return 1
}

# Function to mark model as ready
mark_model_ready() {
    echo "$(date '+%Y-%m-%d %H:%M:%S')" > "$READY_FLAG"
    log_message "✅ Model marked as ready"
}

# Function to wait for model readiness
wait_for_model_ready() {
    local max_wait=300  # 5 minutes max wait
    local wait_time=0
    
    log_message "⏳ Waiting for model to be ready..."
    
    while [[ $wait_time -lt $max_wait ]]; do
        if check_model_ready; then
            log_message "✅ Model is ready!"
            return 0
        fi
        
        sleep 5
        wait_time=$((wait_time + 5))
        log_message "⏳ Still waiting... ($wait_time seconds)"
    done
    
    log_message "❌ Timeout waiting for model to be ready"
    return 1
}

# Function to start the model in background
start_model_background() {
    log_message "🚀 Starting Meta-Model AI Assistant in background..."
    
    # Change to project directory
    cd "$META_MODEL_DIR"
    
    # Start the model in background and capture PID
    nohup python3 main.py --daemon > "$META_MODEL_DIR/logs/assistant.log" 2>&1 &
    local pid=$!
    echo $pid > "$PID_FILE"
    
    log_message "📝 Model started with PID: $pid"
    
    # Wait a moment for startup
    sleep 10
    
    # Check if process is still running
    if kill -0 $pid 2>/dev/null; then
        log_message "✅ Model process is running"
        mark_model_ready
        return 0
    else
        log_message "❌ Model process failed to start"
        return 1
    fi
}

# Function to setup environment
setup_environment() {
    log_message "🔧 Setting up environment..."
    
    cd "$META_MODEL_DIR"
    
    # Check if virtual environment exists
    if [[ ! -d "venv" ]]; then
        log_message "📦 Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Check dependencies
    log_message "🔍 Checking dependencies..."
    python3 scripts/check_env.py
    
    # Check models
    if [[ ! -d "models" ]] || [[ -z "$(ls -A models 2>/dev/null)" ]]; then
        log_message "📥 Downloading models..."
        python3 scripts/download_models.py
    fi
    
    log_message "✅ Environment setup complete"
}

# Function to create necessary directories
create_directories() {
    mkdir -p "$META_MODEL_DIR/logs"
    mkdir -p "$META_MODEL_DIR/models"
    mkdir -p "$META_MODEL_DIR/memory_db"
}

# Function to cleanup on exit
cleanup() {
    log_message "🧹 Cleaning up..."
    
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 $pid 2>/dev/null; then
            log_message "🛑 Stopping model process (PID: $pid)"
            kill $pid
        fi
        rm -f "$PID_FILE"
    fi
    
    rm -f "$READY_FLAG"
}

# Set up signal handlers
trap cleanup EXIT
trap 'log_message "🛑 Received interrupt signal"; exit 1' INT TERM

# Main startup sequence
main() {
    log_message "🚀 Meta-Model AI Assistant Startup Daemon Starting"
    log_message "📁 Working directory: $META_MODEL_DIR"
    
    # Create necessary directories
    create_directories
    
    # Setup environment
    setup_environment
    
    # Start model in background
    if start_model_background; then
        log_message "✅ Startup sequence completed successfully"
        log_message "🤖 Meta-Model AI Assistant is ready for input"
        
        # Keep the daemon running to maintain the model
        while true; do
            if [[ -f "$PID_FILE" ]]; then
                local pid=$(cat "$PID_FILE")
                if ! kill -0 $pid 2>/dev/null; then
                    log_message "❌ Model process died, restarting..."
                    start_model_background
                fi
            else
                log_message "❌ PID file missing, restarting..."
                start_model_background
            fi
            
            sleep 30  # Check every 30 seconds
        done
    else
        log_message "❌ Startup sequence failed"
        exit 1
    fi
}

# Run main function
main "$@" 