#!/bin/bash

# Quark AI System Minimal Terminal Profile
# Shows status without displaying function definitions

# Configuration
QUARK_DIR="/Users/camdouglas/quark"
QUARK_PID_FILE="$QUARK_DIR/logs/quark.pid"
QUARK_READY_FILE="$QUARK_DIR/logs/quark_ready.flag"

# Function to check if Quark is running (silent)
quark_is_running() {
    if [ -f "$QUARK_PID_FILE" ]; then
        local pid=$(cat "$QUARK_PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Function to check if Quark is ready (silent)
quark_is_ready() {
    [ -f "$QUARK_READY_FILE" ]
}

# Function to show Quark status (silent)
show_quark_status() {
    echo -e "\033[0;36m🤖 Quark AI System Status\033[0m"
    echo "=================================="
    
    if quark_is_running; then
        local pid=$(cat "$QUARK_PID_FILE")
        echo -e "\033[0;32m✅ Quark is running (PID: $pid)\033[0m"
        
        if quark_is_ready; then
            echo -e "\033[0;32m✅ Ready for user input\033[0m"
            echo -e "\033[0;34m🌐 Web interface: http://localhost:8000\033[0m"
            echo -e "\033[0;34m📊 Metrics: http://localhost:8001\033[0m"
        else
            echo -e "\033[1;33m⏳ Starting up...\033[0m"
        fi
    else
        echo -e "\033[0;31m❌ Quark is not running\033[0m"
        echo -e "\033[1;33m💡 Run 'quark start' to start Quark\033[0m"
    fi
    
    echo ""
}

# Function to create quark command (silent)
quark_command() {
    case "$1" in
        start)
            "$QUARK_DIR/scripts/startup_quark.sh" start
            ;;
        stop)
            "$QUARK_DIR/scripts/startup_quark.sh" stop
            ;;
        restart)
            "$QUARK_DIR/scripts/startup_quark.sh" restart
            ;;
        status)
            "$QUARK_DIR/scripts/startup_quark.sh" status
            ;;
        web)
            if quark_is_ready; then
                open http://localhost:8000
                echo -e "\033[0;32m✅ Opening Quark web interface...\033[0m"
            else
                echo -e "\033[0;31m❌ Quark is not ready. Run 'quark start' first.\033[0m"
            fi
            ;;
        metrics)
            if quark_is_ready; then
                open http://localhost:8001
                echo -e "\033[0;32m✅ Opening Quark metrics dashboard...\033[0m"
            else
                echo -e "\033[0;31m❌ Quark is not ready. Run 'quark start' first.\033[0m"
            fi
            ;;
        cli)
            if quark_is_ready; then
                echo -e "\033[0;32m🚀 Starting Quark CLI...\033[0m"
                cd "$QUARK_DIR"
                source venv/bin/activate
                python3 -m cli.cli
            else
                echo -e "\033[0;31m❌ Quark is not ready. Run 'quark start' first.\033[0m"
            fi
            ;;
        logs)
            if [ -f "$QUARK_DIR/logs/quark_startup.log" ]; then
                echo -e "\033[0;34m📋 Recent Quark logs:\033[0m"
                tail -n 20 "$QUARK_DIR/logs/quark_startup.log"
            else
                echo -e "\033[1;33mNo logs available\033[0m"
            fi
            ;;
        intelligence)
            if [ -n "$2" ]; then
                cd "$QUARK_DIR"
                source venv/bin/activate
                python3 cli/intelligence_cli.py "$2"
            else
                echo -e "\033[0;31m❌ Intelligence command requires subcommand\033[0m"
                echo "Available: stats, patterns, gains, agents, optimize"
            fi
            ;;
        *)
            echo -e "\033[0;31m❌ Unknown command: $1\033[0m"
            echo -e "\033[0;35m🚀 Available Quark Commands:\033[0m"
            echo "quark start/stop/restart/status"
            echo "quark web/metrics/cli/logs"
            echo "quark intelligence stats/patterns/gains/agents/optimize"
            ;;
    esac
}

# Create quark command alias
alias quark='quark_command'

# Only show status and auto-start when this is sourced in an interactive terminal
if [ -t 0 ] && [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    show_quark_status
    
    # Auto-start Quark if not running
    if ! quark_is_running; then
        echo -e "\033[1;33m🚀 Auto-starting Quark...\033[0m"
        "$QUARK_DIR/scripts/startup_quark.sh" start > /dev/null 2>&1 &
        
        # Wait a moment for startup
        sleep 2
        
        # Show updated status
        show_quark_status
    elif ! quark_is_ready; then
        echo -e "\033[1;33m⏳ Quark is starting up...\033[0m"
    fi
fi

# Export functions and variables (silently)
export -f quark_is_running quark_is_ready show_quark_status quark_command 2>/dev/null || true 