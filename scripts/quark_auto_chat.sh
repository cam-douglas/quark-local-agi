#!/bin/bash

# Quark Auto-Chat Script
# Waits for Quark to be ready and then starts chat interface

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

wait_for_quark() {
    local max_wait=30  # Maximum wait time in seconds
    local wait_time=0
    
    echo -e "${YELLOW}⏳ Waiting for Quark to be ready...${NC}"
    
    while [ $wait_time -lt $max_wait ]; do
        if quark_is_ready; then
            echo -e "${GREEN}✅ Quark is ready!${NC}"
            return 0
        fi
        
        sleep 1
        wait_time=$((wait_time + 1))
        
        # Show progress every 5 seconds
        if [ $((wait_time % 5)) -eq 0 ]; then
            echo -e "${YELLOW}⏳ Still waiting... (${wait_time}s/${max_wait}s)${NC}"
        fi
    done
    
    echo -e "${RED}❌ Timeout waiting for Quark to be ready${NC}"
    return 1
}

start_quark_if_needed() {
    if ! quark_is_running; then
        echo -e "${YELLOW}🚀 Starting Quark...${NC}"
        "$QUARK_DIR/scripts/startup_quark.sh" start > /dev/null 2>&1 &
        
        # Wait a moment for startup
        sleep 3
        
        if quark_is_running; then
            echo -e "${GREEN}✅ Quark started successfully${NC}"
        else
            echo -e "${RED}❌ Failed to start Quark${NC}"
            return 1
        fi
    else
        echo -e "${GREEN}✅ Quark is already running${NC}"
    fi
    
    return 0
}

show_status() {
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
    fi
    echo ""
}

main() {
    # Show initial status
    show_status
    
    # Start Quark if needed
    if ! start_quark_if_needed; then
        echo -e "${RED}❌ Cannot start Quark. Please check logs.${NC}"
        exit 1
    fi
    
    # Wait for Quark to be ready
    if wait_for_quark; then
        echo ""
        echo -e "${GREEN}🚀 Starting Quark Chat Interface...${NC}"
        echo ""
        
        # Start the interactive chat
        cd "$QUARK_DIR"
        source venv/bin/activate
        python3 cli/interactive_cli.py
    else
        echo -e "${RED}❌ Quark is not ready. Please try again later.${NC}"
        echo -e "${YELLOW}💡 You can run 'quark chat' manually when ready${NC}"
    fi
}

# Run the main function
main 