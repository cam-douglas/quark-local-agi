#!/usr/bin/env bash
# Auto-start script for Quark AI Assistant
# This script can be called to automatically start the quark CLI

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Quark project directory
META_MODEL_DIR="/Users/camdouglas/quark"

# Function to check if we should auto-start
should_auto_start() {
    # Check if auto-start is enabled
    if [[ "$META_MODEL_AUTO_START" == "true" ]]; then
        return 0
    fi
    
    # Check if we're in an interactive terminal
    if [[ ! -t 0 ]]; then
        return 1
    fi
    
    # Check if we're already in a quark session
    if [[ -n "$META_MODEL_SESSION" ]]; then
        return 1
    fi
    
    # Check if quark is available
    if [[ ! -d "$META_MODEL_DIR" ]] || [[ ! -f "$META_MODEL_DIR/scripts/meta_shell.sh" ]]; then
        return 1
    fi
    
    return 0
}

# Function to start quark
start_quark() {
    echo -e "${CYAN}🚀 Auto-starting Quark AI Assistant...${NC}"
    echo -e "${YELLOW}💡 Press Ctrl+C to exit${NC}"
    echo ""
    
    # Set session flag
    export META_MODEL_SESSION=1
    
    # Change to project directory
    cd "$META_MODEL_DIR"
    
    # Start the quark CLI
    ./scripts/meta_shell.sh
}

# Main function
main() {
    # Check if we should auto-start
    if should_auto_start; then
        start_quark
    fi
}

# Run main function
main "$@" 