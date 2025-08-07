#!/bin/bash
# Quark Daemon Script
# Runs Quark automatically on system startup

# Configuration
QUARK_DIR="/Users/camdouglas/quark"
LOG_FILE="$QUARK_DIR/logs/quark_daemon.log"
PID_FILE="$QUARK_DIR/logs/quark_daemon.pid"
USER="camdouglas"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
    echo "$1"
}

# Function to check if Quark is running
is_quark_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Function to start Quark
start_quark() {
    log_message "🚀 Starting Quark daemon..."
    
    # Create necessary directories
    mkdir -p "$QUARK_DIR/logs"
    mkdir -p "$QUARK_DIR/data"
    
    # Change to Quark directory
    cd "$QUARK_DIR"
    
    # Activate virtual environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        log_message "❌ Virtual environment not found"
        return 1
    fi
    
    # Set environment variables
    export PYTHONPATH="$QUARK_DIR:$PYTHONPATH"
    export QUARK_HOME="$QUARK_DIR"
    export QUARK_ENV="daemon"
    
    # Start Quark in background
    nohup python3 scripts/quark_optimized_startup.py > "$LOG_FILE" 2>&1 &
    DAEMON_PID=$!
    
    # Save PID
    echo $DAEMON_PID > "$PID_FILE"
    
    log_message "✅ Quark daemon started (PID: $DAEMON_PID)"
    return 0
}

# Function to stop Quark
stop_quark() {
    log_message "🛑 Stopping Quark daemon..."
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            kill $PID
            log_message "✅ Quark daemon stopped (PID: $PID)"
        else
            log_message "⚠️ Quark daemon not running"
        fi
        rm -f "$PID_FILE"
    else
        log_message "⚠️ PID file not found"
    fi
    
    # Clean up ready flag
    rm -f "$QUARK_DIR/logs/quark_ready.flag"
}

# Function to restart Quark
restart_quark() {
    log_message "🔄 Restarting Quark daemon..."
    stop_quark
    sleep 2
    start_quark
}

# Function to show status
show_status() {
    if is_quark_running; then
        PID=$(cat "$PID_FILE")
        log_message "✅ Quark daemon is running (PID: $PID)"
        
        # Check if Quark is ready
        if [ -f "$QUARK_DIR/logs/quark_ready.flag" ]; then
            log_message "✅ Quark is ready for interaction"
        else
            log_message "⏳ Quark is still starting up"
        fi
    else
        log_message "❌ Quark daemon is not running"
    fi
}

# Main script logic
case "$1" in
    start)
        if is_quark_running; then
            log_message "⚠️ Quark daemon is already running"
        else
            start_quark
        fi
        ;;
    stop)
        stop_quark
        ;;
    restart)
        restart_quark
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo ""
        echo "Commands:"
        echo "  start   - Start Quark daemon"
        echo "  stop    - Stop Quark daemon"
        echo "  restart - Restart Quark daemon"
        echo "  status  - Show daemon status"
        exit 1
        ;;
esac 