#!/usr/bin/env bash
# Meta-Model AI Assistant Shell Profile
# Add this to your ~/.bashrc, ~/.zshrc, or ~/.bash_profile

# Colors for output
META_GREEN='\033[0;32m'
META_YELLOW='\033[1;33m'
META_BLUE='\033[0;34m'
META_MAGENTA='\033[0;35m'
META_CYAN='\033[0;36m'
META_NC='\033[0m' # No Color

# Meta-Model project directory
export META_MODEL_DIR="/Users/camdouglas/meta_model"

# Auto-start configuration (set to "true" to enable auto-start)
export META_MODEL_AUTO_START="${META_MODEL_AUTO_START:-true}"

# Function to check if meta_model is available
meta_model_available() {
    [[ -d "$META_MODEL_DIR" && -f "$META_MODEL_DIR/scripts/meta_shell.sh" ]]
}

# Function to activate meta_model environment
activate_meta_model() {
    if [[ -d "$META_MODEL_DIR/venv" ]]; then
        source "$META_MODEL_DIR/venv/bin/activate"
        return 0
    else
        return 1
    fi
}

# Function to show meta_model status
meta_model_status() {
    if meta_model_available; then
        echo -e "${META_GREEN}✅ Meta-Model AI Assistant is available${META_NC}"
        echo -e "${META_CYAN}📁 Project: $META_MODEL_DIR${META_NC}"
        
        if activate_meta_model; then
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
        echo -e "${META_YELLOW}⚠️  Meta-Model AI Assistant not found at $META_MODEL_DIR${META_NC}"
    fi
}

# Function to start meta_model assistant
meta_model() {
    if ! meta_model_available; then
        echo -e "${META_YELLOW}⚠️  Meta-Model AI Assistant not found${META_NC}"
        echo -e "${META_CYAN}💡 Please ensure the project is at: $META_MODEL_DIR${META_NC}"
        return 1
    fi
    
    cd "$META_MODEL_DIR"
    ./scripts/meta_shell.sh "$@"
}

# Function to setup meta_model environment
meta_model_setup() {
    if ! meta_model_available; then
        echo -e "${META_YELLOW}⚠️  Meta-Model AI Assistant not found${META_NC}"
        return 1
    fi
    
    cd "$META_MODEL_DIR"
    ./scripts/setup_env.sh
}

# Function to check meta_model environment
meta_model_check() {
    if ! meta_model_available; then
        echo -e "${META_YELLOW}⚠️  Meta-Model AI Assistant not found${META_NC}"
        return 1
    fi
    
    cd "$META_MODEL_DIR"
    python3 scripts/check_env.py
}

# Function to download meta_model models
meta_model_download() {
    if ! meta_model_available; then
        echo -e "${META_YELLOW}⚠️  Meta-Model AI Assistant not found${META_NC}"
        return 1
    fi
    
    cd "$META_MODEL_DIR"
    python3 scripts/download_models.py
}

# Function to enable auto-start
meta_model_enable_auto_start() {
    export META_MODEL_AUTO_START="true"
    echo -e "${META_GREEN}✅ Auto-start enabled${META_NC}"
    echo -e "${META_YELLOW}💡 Restart your terminal or run: source ~/.bashrc${META_NC}"
}

# Function to disable auto-start
meta_model_disable_auto_start() {
    export META_MODEL_AUTO_START="false"
    echo -e "${META_YELLOW}⚠️  Auto-start disabled${META_NC}"
}

# Function to show meta_model help
meta_model_help() {
    echo -e "${META_MAGENTA}🤖 Meta-Model AI Assistant Commands${META_NC}"
    echo -e "${META_CYAN}================================${META_NC}"
    echo ""
    echo -e "${META_GREEN}meta_model${META_NC}          - Start the AI assistant"
    echo -e "${META_GREEN}meta_model help${META_NC}      - Show help"
    echo -e "${META_GREEN}meta_model status${META_NC}    - Check environment status"
    echo -e "${META_GREEN}meta_model models${META_NC}    - List available models"
    echo -e "${META_GREEN}meta_model setup${META_NC}     - Run environment setup"
    echo -e "${META_GREEN}meta_model download${META_NC}  - Download models"
    echo -e "${META_GREEN}meta_model test${META_NC}      - Run a quick test"
    echo -e "${META_GREEN}meta_model_check${META_NC}     - Check environment"
    echo -e "${META_GREEN}meta_model_status${META_NC}    - Show status"
    echo ""
    echo -e "${META_GREEN}meta_model_enable_auto_start${META_NC}  - Enable auto-start"
    echo -e "${META_GREEN}meta_model_disable_auto_start${META_NC} - Disable auto-start"
    echo ""
    echo -e "${META_YELLOW}Examples:${META_NC}"
    echo -e "  meta_model                    # Start interactive assistant"
    echo -e "  meta_model 'What is AI?'      # Ask a question"
    echo -e "  meta_model help               # Show help"
    echo ""
}

# Function to auto-start meta_model (optional)
auto_start_meta_model() {
    # Only auto-start if enabled and we're in an interactive shell
    if [[ "$META_MODEL_AUTO_START" == "true" ]] && [[ -t 0 ]] && [[ -z "$META_MODEL_AUTO_STARTED" ]]; then
        export META_MODEL_AUTO_STARTED=1
        
        # Check if meta_model is available and ready
        if meta_model_available && [[ -d "$META_MODEL_DIR/venv" ]]; then
            echo -e "${META_CYAN}🤖 Auto-starting Meta-Model AI Assistant...${META_NC}"
            echo -e "${META_YELLOW}💡 Press Ctrl+C to exit${META_NC}"
            echo ""
            
            # Change to project directory and start
            cd "$META_MODEL_DIR"
            python3 scripts/startup.py
        fi
    fi
}

# Auto-activate environment if available
if meta_model_available; then
    # Try to activate the environment
    if activate_meta_model; then
        # Set up aliases
        alias meta='meta_model'
        alias meta-help='meta_model_help'
        alias meta-status='meta_model_status'
        alias meta-check='meta_model_check'
        alias meta-setup='meta_model_setup'
        alias meta-download='meta_model_download'
        alias meta-enable-auto='meta_model_enable_auto_start'
        alias meta-disable-auto='meta_model_disable_auto_start'
        alias meta-ready='$META_MODEL_DIR/scripts/check_ready.sh'
        
        # Auto-start meta_model if enabled
        auto_start_meta_model
    fi
fi

# Export functions so they're available in subshells
export -f meta_model
export -f meta_model_help
export -f meta_model_status
export -f meta_model_check
export -f meta_model_setup
export -f meta_model_download
export -f meta_model_available
export -f activate_meta_model
export -f auto_start_meta_model
export -f meta_model_enable_auto_start
export -f meta_model_disable_auto_start 