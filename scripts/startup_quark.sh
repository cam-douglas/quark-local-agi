#!/bin/bash

# Quark AI System Startup Script
# Handles starting, stopping, and status of Quark

QUARK_DIR="/Users/camdouglas/quark"
PID_FILE="$QUARK_DIR/logs/quark.pid"
READY_FILE="$QUARK_DIR/logs/quark_ready.flag"
LOG_FILE="$QUARK_DIR/logs/quark_startup.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Function to check if models exist
check_models_exist() {
    local models_dir="$QUARK_DIR/models"
    local essential_models=("gpt2" "bert-base-uncased" "sentence-transformers_all-MiniLM-L6-v2")
    
    for model in "${essential_models[@]}"; do
        if [ ! -d "$models_dir/$model" ]; then
            return 1
        fi
    done
    return 0
}

# Function to start Quark
start_quark() {
    log_message "Starting Quark AI System..."
    
    # Check if already running
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            log_message "Quark is already running (PID: $pid)"
            return 0
        else
            log_message "Removing stale PID file"
            rm -f "$PID_FILE"
        fi
    fi
    
    # Activate virtual environment
    log_message "Virtual environment activated"
    cd "$QUARK_DIR"
    source venv/bin/activate
    
    # Launch Quark with instant startup
    log_message "Launching Quark with instant startup..."
    nohup python3 scripts/instant_startup.py > "$QUARK_DIR/logs/quark_output.log" 2>&1 &
    local quark_pid=$!
    
    # Save PID
    echo "$quark_pid" > "$PID_FILE"
    log_message "Quark started with PID: $quark_pid"
    
    # Wait for Quark to be ready
    log_message "Waiting for Quark to be ready..."
    local attempts=0
    local max_attempts=60
    
    while [ $attempts -lt $max_attempts ]; do
        if [ -f "$READY_FILE" ]; then
            log_message "✅ Quark is ready!"
            return 0
        fi
        
        sleep 5
        attempts=$((attempts + 1))
        log_message "Waiting for Quark... (attempt $attempts/$max_attempts)"
    done
    
    log_message "Error: Quark failed to start within timeout"
    return 1
}

# Function to stop Quark
stop_quark() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        log_message "Stopping Quark (PID: $pid)..."
        kill "$pid" 2>/dev/null || true
        rm -f "$PID_FILE"
        rm -f "$READY_FILE"
        log_message "Quark stopped"
    fi
}

# Function to restart Quark
restart_quark() {
    log_message "Restarting Quark..."
    stop_quark
    sleep 2
    start_quark
}

# Function to show Quark status
show_status() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        echo -e "${GREEN}✅ Quark is running (PID: $pid)${NC}"
        
        if [ -f "$READY_FILE" ]; then
            echo -e "${GREEN}✅ Quark is ready for user input${NC}"
        else
            echo -e "${YELLOW}⏳ Quark is starting up...${NC}"
        fi
        
        # Show recent logs
        echo -e "${BLUE}Recent logs:${NC}"
        tail -n 5 "$LOG_FILE" 2>/dev/null || echo "No logs available"
    else
        echo -e "${RED}❌ Quark is not running${NC}"
    fi
}

# Main script logic
case "${1:-start}" in
    start)
        if [ -f "$PID_FILE" ]; then
            log_message "Quark is already running"
            show_status
        else
            start_quark
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✅ Quark started successfully${NC}"
                show_status
            else
                echo -e "${RED}❌ Failed to start Quark${NC}"
                exit 1
            fi
        fi
        ;;
    stop)
        stop_quark
        echo -e "${GREEN}✅ Quark stopped${NC}"
        ;;
    restart)
        restart_quark
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Quark restarted successfully${NC}"
            show_status
        else
            echo -e "${RED}❌ Failed to restart Quark${NC}"
            exit 1
        fi
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo "  start   - Start Quark (default)"
        echo "  stop    - Stop Quark"
        echo "  restart - Restart Quark"
        echo "  status  - Show Quark status"
        exit 1
        ;;
esac 