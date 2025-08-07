#!/bin/bash

# Quark Commands Script
# Defines quark command without displaying function definitions

QUARK_DIR="/Users/camdouglas/quark"
QUARK_PID_FILE="$QUARK_DIR/logs/quark.pid"
QUARK_READY_FILE="$QUARK_DIR/logs/quark_ready.flag"

# Define quark command function (silent)
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
            if [ -f "$QUARK_READY_FILE" ]; then
                open http://localhost:8000
                echo -e "\033[0;32m✅ Opening Quark web interface...\033[0m"
            else
                echo -e "\033[0;31m❌ Quark is not ready. Run 'quark start' first.\033[0m"
            fi
            ;;
        metrics)
            if [ -f "$QUARK_READY_FILE" ]; then
                open http://localhost:8001
                echo -e "\033[0;32m✅ Opening Quark metrics dashboard...\033[0m"
            else
                echo -e "\033[0;31m❌ Quark is not ready. Run 'quark start' first.\033[0m"
            fi
            ;;
        cli)
            if [ -f "$QUARK_READY_FILE" ]; then
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

# Export the function silently
export -f quark_command 2>/dev/null || true 