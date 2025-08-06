#!/usr/bin/env bash
set -e

# ─────────────────────────────────────────────────────────────────────────────
# Meta-Model AI Assistant Shell Integration Installer
# ─────────────────────────────────────────────────────────────────────────────

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_header() {
    echo -e "${MAGENTA}$1${NC}"
}

# Function to detect shell type
detect_shell() {
    if [[ -n "$ZSH_VERSION" ]]; then
        echo "zsh"
    elif [[ -n "$BASH_VERSION" ]]; then
        echo "bash"
    else
        echo "unknown"
    fi
}

# Function to get shell config file
get_shell_config() {
    local shell_type="$1"
    local config_file=""
    
    case "$shell_type" in
        "zsh")
            if [[ -f "$HOME/.zshrc" ]]; then
                config_file="$HOME/.zshrc"
            elif [[ -f "$HOME/.zprofile" ]]; then
                config_file="$HOME/.zprofile"
            else
                config_file="$HOME/.zshrc"
            fi
            ;;
        "bash")
            if [[ -f "$HOME/.bashrc" ]]; then
                config_file="$HOME/.bashrc"
            elif [[ -f "$HOME/.bash_profile" ]]; then
                config_file="$HOME/.bash_profile"
            else
                config_file="$HOME/.bashrc"
            fi
            ;;
        *)
            print_error "Unsupported shell: $shell_type"
            exit 1
            ;;
    esac
    
    echo "$config_file"
}

# Function to check if already installed
is_already_installed() {
    local config_file="$1"
    if [[ -f "$config_file" ]] && grep -q "Meta-Model AI Assistant Shell Profile" "$config_file"; then
        return 0
    else
        return 1
    fi
}

# Function to backup config file
backup_config() {
    local config_file="$1"
    local backup_file="${config_file}.backup.$(date +%Y%m%d_%H%M%S)"
    
    if [[ -f "$config_file" ]]; then
        cp "$config_file" "$backup_file"
        print_info "Backup created: $backup_file"
    fi
}

# Function to install shell integration
install_shell_integration() {
    local shell_type="$1"
    local config_file="$2"
    local project_dir="$(pwd)"
    
    print_info "Installing shell integration for $shell_type..."
    
    # Create config file if it doesn't exist
    if [[ ! -f "$config_file" ]]; then
        touch "$config_file"
        print_info "Created config file: $config_file"
    fi
    
    # Add the profile script to the config file
    cat >> "$config_file" << EOF

# ─────────────────────────────────────────────────────────────────────────────
# Meta-Model AI Assistant Shell Profile
# ─────────────────────────────────────────────────────────────────────────────
source "$project_dir/scripts/meta_profile.sh"
EOF
    
    print_status "Shell integration installed!"
    print_info "Config file: $config_file"
}

# Function to show usage instructions
show_usage_instructions() {
    print_header "🎉 Installation Complete!"
    echo ""
    print_info "To start using Meta-Model AI Assistant:"
    echo ""
    echo "1. Restart your terminal or run:"
    echo "   source ~/.$(detect_shell)rc"
    echo ""
    echo "2. Start the AI assistant:"
    echo "   meta_model"
    echo ""
    echo "3. Available commands:"
    echo "   meta_model help      - Show help"
    echo "   meta_model status    - Check status"
    echo "   meta_model models    - List models"
    echo "   meta_model setup     - Run setup"
    echo "   meta_model download  - Download models"
    echo ""
    print_info "You can now use 'meta_model' from anywhere in your terminal!"
}

# Function to test installation
test_installation() {
    print_info "Testing installation..."
    
    # Source the profile script
    source scripts/meta_profile.sh
    
    # Test if functions are available
    if command -v meta_model >/dev/null 2>&1; then
        print_status "meta_model command is available"
    else
        print_error "meta_model command not found"
        return 1
    fi
    
    if command -v meta_model_status >/dev/null 2>&1; then
        print_status "meta_model_status command is available"
    else
        print_error "meta_model_status command not found"
        return 1
    fi
    
    print_status "Installation test passed!"
    return 0
}

# Main installation function
main() {
    print_header "🚀 Meta-Model AI Assistant Shell Integration Installer"
    echo ""
    
    # Check if we're in the right directory
    if [[ ! -f "scripts/meta_profile.sh" ]]; then
        print_error "Please run this script from the meta_model project root directory"
        exit 1
    fi
    
    # Detect shell type
    local shell_type=$(detect_shell)
    print_info "Detected shell: $shell_type"
    
    if [[ "$shell_type" == "unknown" ]]; then
        print_error "Unsupported shell. Please use bash or zsh."
        exit 1
    fi
    
    # Get shell config file
    local config_file=$(get_shell_config "$shell_type")
    print_info "Config file: $config_file"
    
    # Check if already installed
    if is_already_installed "$config_file"; then
        print_warning "Meta-Model AI Assistant is already installed in $config_file"
        read -p "Do you want to reinstall? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Installation cancelled."
            exit 0
        fi
    fi
    
    # Backup existing config
    backup_config "$config_file"
    
    # Install shell integration
    install_shell_integration "$shell_type" "$config_file"
    
    # Test installation
    if test_installation; then
        show_usage_instructions
    else
        print_error "Installation test failed. Please check the setup."
        exit 1
    fi
}

# Run main function
main "$@" 