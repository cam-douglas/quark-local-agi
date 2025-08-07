#!/bin/bash
# Quark AI System Status Check
# This script checks the status of the Quark AI System

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🤖 Quark AI System Status${NC}"
echo "=================================="

# Check if Quark process is running
QUARK_PID=$(ps aux | grep "python.*main.py" | grep -v grep | awk '{print $2}' | head -1)

if [ -n "$QUARK_PID" ]; then
    echo -e "${GREEN}✅ Quark is running (PID: $QUARK_PID)${NC}"
else
    echo -e "${RED}❌ Quark is not running${NC}"
fi

# Check health endpoint
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Health endpoint responding${NC}"
    HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
    echo -e "${BLUE}📊 Health status: $HEALTH_RESPONSE${NC}"
else
    echo -e "${RED}❌ Health endpoint not responding${NC}"
fi

# Check port usage
if lsof -i :8000 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Port 8000 is in use${NC}"
else
    echo -e "${RED}❌ Port 8000 is not in use${NC}"
fi

# Check virtual environment
if [ -d "venv" ]; then
    echo -e "${GREEN}✅ Virtual environment exists${NC}"
else
    echo -e "${RED}❌ Virtual environment not found${NC}"
fi

# Check main.py
if [ -f "main.py" ]; then
    echo -e "${GREEN}✅ main.py exists${NC}"
else
    echo -e "${RED}❌ main.py not found${NC}"
fi

echo ""
echo -e "${BLUE}📡 Health endpoint: http://localhost:8000/health${NC}"
echo -e "${BLUE}🛠️  To start Quark: ./scripts/start_quark.sh${NC}"
echo -e "${BLUE}🛑 To stop Quark: pkill -f 'python.*main.py'${NC}" 