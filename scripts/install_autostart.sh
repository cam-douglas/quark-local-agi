#!/bin/bash

# Quark AI System Auto-Start Installation Script
# This script sets up Quark to run automatically on macOS startup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
QUARK_DIR="/Users/camdouglas/quark"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST_FILE="$LAUNCH_AGENTS_DIR/com.camdouglas.quark.plist"
SHELL_PROFILE="$HOME/.zshrc"

echo -e "${PURPLE}🚀 Quark AI System Auto-Start Installation${NC}"
echo "================================================"

# Check if running as the correct user
if [ "$USER" != "camdouglas" ]; then
    echo -e "${RED}❌ This script must be run as user 'camdouglas'${NC}"
    echo "Current user: $USER"
    exit 1
fi

# Check if Quark directory exists
if [ ! -d "$QUARK_DIR" ]; then
    echo -e "${RED}❌ Quark directory not found at $QUARK_DIR${NC}"
    exit 1
fi

echo -e "${BLUE}📁 Quark directory: $QUARK_DIR${NC}"

# Make startup script executable
echo -e "${YELLOW}🔧 Making startup script executable...${NC}"
chmod +x "$QUARK_DIR/scripts/startup_quark.sh"
chmod +x "$QUARK_DIR/scripts/quark_profile.sh"

# Create LaunchAgents directory if it doesn't exist
if [ ! -d "$LAUNCH_AGENTS_DIR" ]; then
    echo -e "${YELLOW}📁 Creating LaunchAgents directory...${NC}"
    mkdir -p "$LAUNCH_AGENTS_DIR"
fi

# Copy plist file to LaunchAgents
echo -e "${YELLOW}📋 Installing LaunchAgent...${NC}"
cp "$QUARK_DIR/scripts/com.camdouglas.quark.plist" "$PLIST_FILE"

# Load the LaunchAgent
echo -e "${YELLOW}🔄 Loading LaunchAgent...${NC}"
launchctl unload "$PLIST_FILE" 2>/dev/null || true
launchctl load "$PLIST_FILE"

# Check if shell profile exists
if [ ! -f "$SHELL_PROFILE" ]; then
    echo -e "${YELLOW}📝 Creating shell profile...${NC}"
    touch "$SHELL_PROFILE"
fi

# Add Quark profile to shell configuration
echo -e "${YELLOW}🔧 Adding Quark profile to shell configuration...${NC}"
if ! grep -q "quark_profile.sh" "$SHELL_PROFILE"; then
    echo "" >> "$SHELL_PROFILE"
    echo "# Quark AI System Profile" >> "$SHELL_PROFILE"
    echo "source $QUARK_DIR/scripts/quark_profile.sh" >> "$SHELL_PROFILE"
    echo "" >> "$SHELL_PROFILE"
fi

# Create necessary directories
echo -e "${YELLOW}📁 Creating necessary directories...${NC}"
mkdir -p "$QUARK_DIR/logs"
mkdir -p "$QUARK_DIR/models"

# Test the startup script
echo -e "${YELLOW}🧪 Testing startup script...${NC}"
if "$QUARK_DIR/scripts/startup_quark.sh" status > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Startup script is working${NC}"
else
    echo -e "${YELLOW}⚠️  Startup script test inconclusive (this is normal if Quark isn't running)${NC}"
fi

# Show installation summary
echo ""
echo -e "${GREEN}✅ Installation Complete!${NC}"
echo "================================"
echo -e "${BLUE}📋 What was installed:${NC}"
echo "  • LaunchAgent for automatic startup"
echo "  • Terminal profile integration"
echo "  • Quark command aliases"
echo "  • Status checking on terminal open"
echo ""
echo -e "${BLUE}🚀 Next steps:${NC}"
echo "  1. Restart your Mac to test auto-startup"
echo "  2. Open a new terminal window to see Quark status"
echo "  3. Use 'quark start' to manually start Quark"
echo "  4. Use 'quark web' to open the web interface"
echo ""
echo -e "${BLUE}📝 Available commands:${NC}"
echo "  quark start    - Start Quark"
echo "  quark stop     - Stop Quark"
echo "  quark restart  - Restart Quark"
echo "  quark status   - Show status"
echo "  quark web      - Open web interface"
echo "  quark metrics  - Open metrics dashboard"
echo "  quark cli      - Open CLI interface"
echo "  quark logs     - Show logs"
echo ""
echo -e "${YELLOW}💡 Tip: Quark will automatically start when you log in to your Mac${NC}"
echo -e "${YELLOW}💡 Tip: Open a new terminal window to see Quark status${NC}"

# Test the profile script
echo ""
echo -e "${BLUE}🧪 Testing profile script...${NC}"
source "$QUARK_DIR/scripts/quark_profile.sh" 