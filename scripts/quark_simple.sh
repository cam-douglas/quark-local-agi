#!/bin/bash

# Quark Simple Status Display
# Shows status and auto-starts chat when ready

QUARK_DIR="/Users/camdouglas/quark"
QUARK_PID_FILE="$QUARK_DIR/logs/quark.pid"
QUARK_READY_FILE="$QUARK_DIR/logs/quark_ready.flag"

# Colors for output
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

quark_is_running() {
    if [ -f "$QUARK_PID_FILE" ]; then
        local pid=$(cat "$QUARK_PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

quark_is_ready() {
    [ -f "$QUARK_READY_FILE" ]
}

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
            echo ""
            echo -e "${GREEN}🚀 Starting Quark Chat Interface...${NC}"
            echo ""
            # Start the interactive chat
            cd "$QUARK_DIR"
            source venv/bin/activate
            python3 cli/interactive_cli.py
        else
            echo -e "${YELLOW}⏳ Starting up...${NC}"
            echo -e "${YELLOW}💡 Chat will start automatically when ready${NC}"
        fi
    else
        echo -e "${RED}❌ Quark is not running${NC}"
        echo -e "${YELLOW}💡 Run 'quark start' to start Quark${NC}"
        echo ""
        echo -e "${YELLOW}🚀 Auto-starting Quark...${NC}"
        "$QUARK_DIR/scripts/startup_quark.sh" start > /dev/null 2>&1 &
        echo -e "${YELLOW}⏳ Please wait a moment for Quark to start...${NC}"
        echo -e "${YELLOW}💡 Run 'quark chat' when ready${NC}"
    fi
    echo ""
}

# Show status and auto-start chat if ready
show_quark_status 