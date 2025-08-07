#!/bin/bash

# Quark AI System Terminal Profile
# This script runs when you open a new terminal window

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
QUARK_DIR="/Users/camdouglas/quark"
QUARK_PID_FILE="$QUARK_DIR/logs/quark.pid"
QUARK_READY_FILE="$QUARK_DIR/logs/quark_ready.flag"
QUARK_STATUS_FILE="$QUARK_DIR/logs/quark.status"

# Function to check if Quark is running
quark_is_running() {
    if [ -f "$QUARK_PID_FILE" ]; then
        local pid=$(cat "$QUARK_PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Function to check if Quark is ready
quark_is_ready() {
    [ -f "$QUARK_READY_FILE" ]
}

# Function to show Quark status
show_quark_status() {
    echo -e "${CYAN}🤖 Quark AI System Status${NC}"
    echo "=================================="
    
    if quark_is_running; then
        local pid=$(cat "$QUARK_PID_FILE")
        echo -e "${GREEN}✅ Quark is running (PID: $pid)${NC}"
        
        if quark_is_ready; then
            echo -e "${GREEN}✅ Ready for user input${NC}"
            echo -e "${BLUE}🌐 Web interface: http://localhost:8000${NC}"
            echo -e "${BLUE}📊 Metrics: http://localhost:8001${NC}"
        else
            echo -e "${YELLOW}⏳ Starting up...${NC}"
        fi
    else
        echo -e "${RED}❌ Quark is not running${NC}"
        echo -e "${YELLOW}💡 Run 'quark start' to start Quark${NC}"
    fi
    
    echo ""
}

# Function to show Quark commands
show_quark_commands() {
    echo -e "${PURPLE}🚀 Available Quark Commands:${NC}"
    echo "quark start    - Start Quark AI System"
    echo "quark stop     - Stop Quark AI System"
    echo "quark restart  - Restart Quark AI System"
    echo "quark status   - Show Quark status"
    echo "quark web      - Open Quark web interface"
    echo "quark metrics  - Open Quark metrics dashboard"
    echo "quark cli      - Open Quark CLI interface"
    echo "quark logs     - Show recent Quark logs"
    echo ""
    echo -e "${CYAN}🧠 Intelligence Commands:${NC}"
    echo "quark intelligence stats     - Show intelligence improvement stats"
    echo "quark intelligence patterns  - Show learning patterns"
    echo "quark intelligence gains     - Show efficiency gains"
    echo "quark intelligence agents    - Show agent optimizations"
    echo "quark intelligence optimize  - Run optimization cycle"
    echo ""
}

# Function to create quark command
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
                echo -e "${GREEN}✅ Opening Quark web interface...${NC}"
            else
                echo -e "${RED}❌ Quark is not ready. Run 'quark start' first.${NC}"
            fi
            ;;
        metrics)
            if quark_is_ready; then
                open http://localhost:8001
                echo -e "${GREEN}✅ Opening Quark metrics dashboard...${NC}"
            else
                echo -e "${RED}❌ Quark is not ready. Run 'quark start' first.${NC}"
            fi
            ;;
        cli)
            if quark_is_ready; then
                echo -e "${GREEN}🚀 Starting Quark CLI...${NC}"
                cd "$QUARK_DIR"
                source venv/bin/activate
                python3 -m cli.cli
            else
                echo -e "${RED}❌ Quark is not ready. Run 'quark start' first.${NC}"
            fi
            ;;
        logs)
            if [ -f "$QUARK_DIR/logs/quark_startup.log" ]; then
                echo -e "${BLUE}📋 Recent Quark logs:${NC}"
                tail -n 20 "$QUARK_DIR/logs/quark_startup.log"
            else
                echo -e "${YELLOW}No logs available${NC}"
            fi
            ;;
        intelligence)
            if [ -n "$2" ]; then
                cd "$QUARK_DIR"
                source venv/bin/activate
                python3 cli/intelligence_cli.py "$2"
            else
                echo -e "${RED}❌ Intelligence command requires subcommand${NC}"
                echo "Available: stats, patterns, gains, agents, optimize"
            fi
            ;;
        *)
            echo -e "${RED}❌ Unknown command: $1${NC}"
            show_quark_commands
            ;;
    esac
}

# Create quark command alias
alias quark='quark_command'

# Only show status and auto-start when this is sourced in an interactive terminal
if [ -t 0 ] && [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    # Use the clean status script to avoid displaying function definitions
    "$QUARK_DIR/scripts/quark_status.sh"
    
    # Auto-start Quark if not running
    if ! quark_is_running; then
        echo -e "${YELLOW}🚀 Auto-starting Quark...${NC}"
        "$QUARK_DIR/scripts/startup_quark.sh" start > /dev/null 2>&1 &
        
        # Wait a moment for startup
        sleep 2
        
        # Show updated status
        "$QUARK_DIR/scripts/quark_status.sh"
    elif ! quark_is_ready; then
        echo -e "${YELLOW}⏳ Quark is starting up...${NC}"
        show_quark_commands
    fi
fi

# Export functions and variables (silently)
export -f quark_is_running quark_is_ready show_quark_status show_quark_commands quark_command 2>/dev/null || true 