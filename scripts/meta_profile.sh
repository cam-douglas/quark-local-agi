#!/usr/bin/env bash
# Quark AI Assistant Shell Profile
# Add this to your ~/.bashrc, ~/.zshrc, or ~/.bash_profile

# Colors for output
META_GREEN='\033[0;32m'
META_YELLOW='\033[1;33m'
META_BLUE='\033[0;34m'
META_MAGENTA='\033[0;35m'
META_CYAN='\033[0;36m'
META_NC='\033[0m' # No Color

# Quark project directory
export META_MODEL_DIR="/Users/camdouglas/quark"

# Auto-start configuration (set to "true" to enable auto-start)
export META_MODEL_AUTO_START="${META_MODEL_AUTO_START:-true}"

# Function to check if quark is available
quark_available() {
    [[ -d "$META_MODEL_DIR" && -f "$META_MODEL_DIR/scripts/meta_shell.sh" ]]
}

# Function to activate quark environment
activate_quark() {
    if [[ -d "$META_MODEL_DIR/venv" ]]; then
        source "$META_MODEL_DIR/venv/bin/activate"
        return 0
    else
        return 1
    fi
}

# Function to show quark status
quark_status() {
    if quark_available; then
        echo -e "${META_GREEN}✅ Quark AI Assistant is available${META_NC}"
        echo -e "${META_CYAN}📁 Project: $META_MODEL_DIR${META_NC}"
        
        if activate_quark; then
            echo -e "${META_GREEN}✅ Virtual environment is active${META_NC}"
        else
            echo -e "${META_YELLOW}⚠️  Virtual environment not found${META_NC}"
        fi
        
        if [[ -d "$META_MODEL_DIR/models" && -n "$(ls -A "$META_MODEL_DIR/models" 2>/dev/null)" ]]; then
            echo -e "${META_GREEN}✅ Models are downloaded${META_NC}"
        else
            echo -e "${META_YELLOW}⚠️  Models not found${META_NC}"
        fi
        
        # Show auto-start status
        if [[ "$META_MODEL_AUTO_START" == "true" ]]; then
            echo -e "${META_GREEN}✅ Auto-start is enabled${META_NC}"
        else
            echo -e "${META_YELLOW}⚠️  Auto-start is disabled${META_NC}"
        fi
    else
        echo -e "${META_YELLOW}⚠️  Quark AI Assistant not found at $META_MODEL_DIR${META_NC}"
    fi
}

# Function to start quark assistant
quark() {
    if ! quark_available; then
        echo -e "${META_YELLOW}⚠️  Quark AI Assistant not found${META_NC}"
        echo -e "${META_CYAN}💡 Please ensure the project is at: $META_MODEL_DIR${META_NC}"
        return 1
    fi
    
    cd "$META_MODEL_DIR"
    ./scripts/meta_shell.sh "$@"
}

# Function to setup quark environment
quark_setup() {
    if ! quark_available; then
        echo -e "${META_YELLOW}⚠️  Quark AI Assistant not found${META_NC}"
        return 1
    fi
    
    cd "$META_MODEL_DIR"
    ./scripts/setup_env.sh
}

# Function to check quark environment
quark_check() {
    if ! quark_available; then
        echo -e "${META_YELLOW}⚠️  Quark AI Assistant not found${META_NC}"
        return 1
    fi
    
    cd "$META_MODEL_DIR"
    python3 scripts/check_env.py
}

# Function to download quark models
quark_download() {
    if ! quark_available; then
        echo -e "${META_YELLOW}⚠️  Quark AI Assistant not found${META_NC}"
        return 1
    fi
    
    cd "$META_MODEL_DIR"
    python3 scripts/download_models.py
}

# Function to enable auto-start
quark_enable_auto_start() {
    export META_MODEL_AUTO_START="true"
    echo -e "${META_GREEN}✅ Auto-start enabled${META_NC}"
    echo -e "${META_YELLOW}💡 Restart your terminal or run: source ~/.bashrc${META_NC}"
}

# Function to disable auto-start
quark_disable_auto_start() {
    export META_MODEL_AUTO_START="false"
    echo -e "${META_YELLOW}⚠️  Auto-start disabled${META_NC}"
}

# Function to show quark help
quark_help() {
    echo -e "${META_MAGENTA}🤖 Quark AI Assistant Commands${META_NC}"
    echo -e "${META_CYAN}================================${META_NC}"
    echo ""
    echo -e "${META_GREEN}quark${META_NC}          - Start the AI assistant"
    echo -e "${META_GREEN}quark help${META_NC}      - Show help"
    echo -e "${META_GREEN}quark status${META_NC}    - Check environment status"
    echo -e "${META_GREEN}quark models${META_NC}    - List available models"
    echo -e "${META_GREEN}quark setup${META_NC}     - Run environment setup"
    echo -e "${META_GREEN}quark download${META_NC}  - Download models"
    echo -e "${META_GREEN}quark test${META_NC}      - Run a quick test"
    echo -e "${META_GREEN}quark_check${META_NC}     - Check environment"
    echo -e "${META_GREEN}quark_status${META_NC}    - Show status"
    echo ""
    echo -e "${META_GREEN}quark_enable_auto_start${META_NC}  - Enable auto-start"
    echo -e "${META_GREEN}quark_disable_auto_start${META_NC} - Disable auto-start"
    echo ""
    echo -e "${META_YELLOW}Examples:${META_NC}"
    echo -e "  quark                    # Start interactive assistant"
    echo -e "  quark 'What is AI?'      # Ask a question"
    echo -e "  quark help               # Show help"
    echo ""
}

# Function to auto-start quark (optional)
auto_start_quark() {
    # Only auto-start if enabled and we're in an interactive shell
    if [[ "$META_MODEL_AUTO_START" == "true" ]] && [[ -t 0 ]] && [[ -z "$META_MODEL_AUTO_STARTED" ]]; then
        export META_MODEL_AUTO_STARTED=1
        
        # Check if quark is available and ready
        if quark_available && [[ -d "$META_MODEL_DIR/venv" ]]; then
            echo -e "${META_CYAN}🤖 Auto-starting Quark AI Assistant...${META_NC}"
            echo -e "${META_YELLOW}💡 Press Ctrl+C to exit${META_NC}"
            echo ""
            
            # Change to project directory and start
            cd "$META_MODEL_DIR"
            python3 scripts/startup.py
        fi
    fi
}

# Auto-activate environment if available
if quark_available; then
    # Try to activate the environment
    if activate_quark; then
        # Set up aliases
        alias meta='quark'
        alias meta-help='quark_help'
        alias meta-status='quark_status'
        alias meta-check='quark_check'
        alias meta-setup='quark_setup'
        alias meta-download='quark_download'
        alias meta-enable-auto='quark_enable_auto_start'
        alias meta-disable-auto='quark_disable_auto_start'
        alias meta-ready='$META_MODEL_DIR/scripts/check_ready.sh'
        
        # Auto-start quark if enabled
        auto_start_quark
    fi
fi

# Export functions so they're available in subshells
export -f quark
export -f quark_help
export -f quark_status
export -f quark_check
export -f quark_setup
export -f quark_download
export -f quark_available
export -f activate_quark
export -f auto_start_quark
export -f quark_enable_auto_start
export -f quark_disable_auto_start 