#!/bin/bash
# Quark AI System Startup Script
# This script starts the Quark AI System with proper environment setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🤖 Quark AI System Startup${NC}"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo -e "${RED}❌ Error: main.py not found. Please run this script from the Quark project root.${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Error: Virtual environment not found. Please run setup first.${NC}"
    exit 1
fi

# Check if Quark is already running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Quark is already running on port 8000${NC}"
    echo -e "${GREEN}✅ Health check: http://localhost:8000/health${NC}"
    exit 0
fi

echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source venv/bin/activate

echo -e "${BLUE}📦 Checking dependencies...${NC}"
python3 -c "import flask, transformers, torch" 2>/dev/null || {
    echo -e "${YELLOW}⚠️  Installing missing dependencies...${NC}"
    pip install flask fastapi uvicorn transformers torch
}

echo -e "${BLUE}🚀 Starting Quark AI System...${NC}"
echo -e "${BLUE}📡 Health endpoint will be available at: http://localhost:8000/health${NC}"
echo -e "${BLUE}🛑 Press Ctrl+C to stop Quark${NC}"
echo ""

# Start Quark
python3 main.py 