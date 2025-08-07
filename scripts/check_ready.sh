#!/bin/bash
# Quick script to check if Quark AI Assistant is ready

META_MODEL_DIR="/Users/camdouglas/quark"
READY_FLAG="$META_MODEL_DIR/logs/model_ready.flag"
PID_FILE="$META_MODEL_DIR/logs/quark.pid"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "🤖 Quark AI Assistant Status Check"
echo "======================================"

# Check if ready flag exists
if [[ -f "$READY_FLAG" ]]; then
    echo -e "${GREEN}✅ Model is ready for input${NC}"
    echo "📅 Ready since: $(cat "$READY_FLAG")"
else
    echo -e "${YELLOW}⚠️  Model is not ready${NC}"
fi

# Check if process is running
if [[ -f "$PID_FILE" ]]; then
    pid=$(cat "$PID_FILE")
    if kill -0 $pid 2>/dev/null; then
        echo -e "${GREEN}✅ Model process is running (PID: $pid)${NC}"
    else
        echo -e "${RED}❌ Model process is not running${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  No PID file found${NC}"
fi

# Check LaunchAgent status
if launchctl list | grep -q "com.metamodel.ai.assistant"; then
    echo -e "${GREEN}✅ LaunchAgent is loaded${NC}"
else
    echo -e "${YELLOW}⚠️  LaunchAgent is not loaded${NC}"
fi

echo ""
echo "💡 To start manually: ./scripts/install_startup.sh start"
echo "💡 To stop: ./scripts/install_startup.sh stop"
echo "💡 To check status: ./scripts/install_startup.sh status" 