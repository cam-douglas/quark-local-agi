#!/bin/bash

# Quark AI System Status Script
# Shows status without displaying function definitions

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

# Show status
show_quark_status 