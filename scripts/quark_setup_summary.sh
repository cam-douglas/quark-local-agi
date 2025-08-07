#!/bin/bash
# Quark Setup Summary
# Shows the complete Quark setup and provides usage instructions

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

QUARK_DIR="/Users/camdouglas/quark"

echo -e "${PURPLE}🤖 Quark AI System - Complete Setup Summary${NC}"
echo "================================================"
echo ""

# Check if Quark is running
if [ -f "$QUARK_DIR/logs/quark.pid" ]; then
    PID=$(cat "$QUARK_DIR/logs/quark.pid")
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Quark is running (PID: $PID)${NC}"
    else
        echo -e "${YELLOW}⚠️ Quark PID file exists but process not running${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ Quark is not running${NC}"
fi

echo ""
echo -e "${BLUE}📁 Installation Locations:${NC}"
echo "  • Quark Directory: $QUARK_DIR"
echo "  • System Launcher: /Users/camdouglas/.local/bin/quark"
echo "  • Auto-start: $HOME/Library/LaunchAgents/com.camdouglas.quark.plist"
echo "  • Desktop Shortcut: $HOME/Desktop/Quark.command"
echo "  • Shell Profile: $HOME/.bashrc"

echo ""
echo -e "${BLUE}🚀 Quick Start Commands:${NC}"
echo "  • quark - Start Quark interactive shell"
echo "  • quark status - Check Quark status"
echo "  • ./scripts/quark_daemon.sh start - Start Quark daemon"
echo "  • ./scripts/quark_daemon.sh stop - Stop Quark daemon"
echo "  • ./scripts/quark_daemon.sh restart - Restart Quark daemon"

echo ""
echo -e "${BLUE}⚙️ Management Commands:${NC}"
echo "  • launchctl load $HOME/Library/LaunchAgents/com.camdouglas.quark.plist - Enable auto-start"
echo "  • launchctl unload $HOME/Library/LaunchAgents/com.camdouglas.quark.plist - Disable auto-start"
echo "  • launchctl list | grep quark - Check auto-start status"

echo ""
echo -e "${BLUE}📊 System Status:${NC}"

# Check LaunchAgent status
if launchctl list | grep -q "com.camdouglas.quark"; then
    echo -e "  ${GREEN}✅ Auto-start enabled${NC}"
else
    echo -e "  ${YELLOW}⚠️ Auto-start disabled${NC}"
fi

# Check if launcher is in PATH
if command -v quark >/dev/null 2>&1; then
    echo -e "  ${GREEN}✅ System launcher available${NC}"
else
    echo -e "  ${YELLOW}⚠️ System launcher not in PATH${NC}"
fi

# Check desktop shortcut
if [ -f "$HOME/Desktop/Quark.command" ]; then
    echo -e "  ${GREEN}✅ Desktop shortcut available${NC}"
else
    echo -e "  ${YELLOW}⚠️ Desktop shortcut not found${NC}"
fi

echo ""
echo -e "${BLUE}🎯 How to Use:${NC}"
echo "  1. Open a new terminal window"
echo "  2. Type 'quark' to start Quark interactive shell"
echo "  3. Or double-click 'Quark.command' on desktop"
echo "  4. Quark will start automatically on system login"

echo ""
echo -e "${BLUE}🔧 Troubleshooting:${NC}"
echo "  • If 'quark' command not found: source ~/.bashrc"
echo "  • If Quark won't start: ./scripts/quark_daemon.sh restart"
echo "  • If auto-start not working: launchctl load $HOME/Library/LaunchAgents/com.camdouglas.quark.plist"
echo "  • Check logs: tail -f $QUARK_DIR/logs/quark_optimized_startup.log"

echo ""
echo -e "${CYAN}💡 Tips:${NC}"
echo "  • Quark pre-loads all agents in parallel for faster startup"
echo "  • The interactive shell starts immediately while agents load in background"
echo "  • Use 'status' command in Quark to see loading progress"
echo "  • All pillars are operational by default"

echo ""
echo -e "${GREEN}🎉 Quark is ready for use!${NC}"
echo "================================================" 