#!/bin/bash
# Quark AI Assistant Startup Installation Script
# This script installs the LaunchAgent for automatic startup

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
RED='\033[0;31m'
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

# Quark project directory
META_MODEL_DIR="/Users/camdouglas/quark"
LAUNCH_AGENT_DIR="$HOME/Library/LaunchAgents"
PLIST_FILE="$LAUNCH_AGENT_DIR/com.metamodel.ai.assistant.plist"

# Function to check if LaunchAgent is already installed
is_already_installed() {
    if [[ -f "$PLIST_FILE" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to backup existing LaunchAgent
backup_existing() {
    if [[ -f "$PLIST_FILE" ]]; then
        local backup_file="${PLIST_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
        cp "$PLIST_FILE" "$backup_file"
        print_info "Backup created: $backup_file"
    fi
}

# Function to create LaunchAgent directory
create_launch_agent_dir() {
    if [[ ! -d "$LAUNCH_AGENT_DIR" ]]; then
        mkdir -p "$LAUNCH_AGENT_DIR"
        print_info "Created LaunchAgents directory: $LAUNCH_AGENT_DIR"
    fi
}

# Function to install LaunchAgent
install_launch_agent() {
    print_info "Installing LaunchAgent..."
    
    # Copy the plist file
    cp "$META_MODEL_DIR/scripts/mac_startup_agent.plist" "$PLIST_FILE"
    
    # Set proper permissions
    chmod 644 "$PLIST_FILE"
    
    print_status "LaunchAgent installed: $PLIST_FILE"
}

# Function to load LaunchAgent
load_launch_agent() {
    print_info "Loading LaunchAgent..."
    
    # Unload if already loaded
    if launchctl list | grep -q "com.metamodel.ai.assistant"; then
        launchctl unload "$PLIST_FILE" 2>/dev/null || true
    fi
    
    # Load the LaunchAgent
    launchctl load "$PLIST_FILE"
    
    print_status "LaunchAgent loaded successfully"
}

# Function to test startup
test_startup() {
    print_info "Testing startup sequence..."
    
    # Start the daemon in background for testing
    "$META_MODEL_DIR/scripts/startup_daemon.sh" &
    local test_pid=$!
    
    # Wait a moment for startup
    sleep 15
    
    # Check if ready flag exists
    if [[ -f "$META_MODEL_DIR/logs/model_ready.flag" ]]; then
        print_status "✅ Startup test successful - model is ready"
        
        # Clean up test process
        kill $test_pid 2>/dev/null || true
        rm -f "$META_MODEL_DIR/logs/model_ready.flag"
        rm -f "$META_MODEL_DIR/logs/quark.pid"
        
        return 0
    else
        print_error "❌ Startup test failed"
        
        # Clean up test process
        kill $test_pid 2>/dev/null || true
        
        return 1
    fi
}

# Function to show status
show_status() {
    print_header "🤖 Quark AI Assistant Startup Status"
    echo ""
    
    if is_already_installed; then
        print_status "LaunchAgent is installed"
        echo "  📁 Location: $PLIST_FILE"
        
        if launchctl list | grep -q "com.metamodel.ai.assistant"; then
            print_status "LaunchAgent is loaded and running"
        else
            print_warning "LaunchAgent is not loaded"
        fi
        
        if [[ -f "$META_MODEL_DIR/logs/model_ready.flag" ]]; then
            print_status "Model is ready for input"
        else
            print_warning "Model is not ready"
        fi
    else
        print_warning "LaunchAgent is not installed"
    fi
    
    echo ""
    print_info "Available commands:"
    echo "  ./scripts/install_startup.sh install    - Install startup service"
    echo "  ./scripts/install_startup.sh uninstall  - Remove startup service"
    echo "  ./scripts/install_startup.sh status     - Show status"
    echo "  ./scripts/install_startup.sh test       - Test startup"
    echo "  ./scripts/install_startup.sh start      - Start manually"
    echo "  ./scripts/install_startup.sh stop       - Stop manually"
}

# Function to uninstall
uninstall() {
    print_info "Uninstalling LaunchAgent..."
    
    # Unload if loaded
    if launchctl list | grep -q "com.metamodel.ai.assistant"; then
        launchctl unload "$PLIST_FILE" 2>/dev/null || true
    fi
    
    # Remove plist file
    if [[ -f "$PLIST_FILE" ]]; then
        rm "$PLIST_FILE"
        print_status "LaunchAgent removed"
    fi
    
    # Clean up logs
    rm -f "$META_MODEL_DIR/logs/model_ready.flag"
    rm -f "$META_MODEL_DIR/logs/quark.pid"
    
    print_status "Uninstallation complete"
}

# Function to start manually
start_manual() {
    print_info "Starting Quark AI Assistant manually..."
    "$META_MODEL_DIR/scripts/startup_daemon.sh"
}

# Function to stop manually
stop_manual() {
    print_info "Stopping Quark AI Assistant..."
    
    if [[ -f "$META_MODEL_DIR/logs/quark.pid" ]]; then
        local pid=$(cat "$META_MODEL_DIR/logs/quark.pid")
        if kill -0 $pid 2>/dev/null; then
            kill $pid
            print_status "Model process stopped"
        else
            print_warning "Model process not running"
        fi
        rm -f "$META_MODEL_DIR/logs/quark.pid"
    fi
    
    rm -f "$META_MODEL_DIR/logs/model_ready.flag"
}

# Main function
main() {
    local command="${1:-status}"
    
    case "$command" in
        "install")
            print_header "🚀 Installing Quark AI Assistant Startup Service"
            echo ""
            
            if is_already_installed; then
                print_warning "LaunchAgent is already installed"
                backup_existing
            fi
            
            create_launch_agent_dir
            install_launch_agent
            load_launch_agent
            
            echo ""
            print_status "Installation complete!"
            print_info "The Quark AI Assistant will now start automatically with your system."
            print_info "You can check status with: ./scripts/install_startup.sh status"
            ;;
            
        "uninstall")
            print_header "🗑️  Uninstalling Quark AI Assistant Startup Service"
            echo ""
            uninstall
            ;;
            
        "status")
            show_status
            ;;
            
        "test")
            print_header "🧪 Testing Quark AI Assistant Startup"
            echo ""
            test_startup
            ;;
            
        "start")
            start_manual
            ;;
            
        "stop")
            stop_manual
            ;;
            
        *)
            print_error "Unknown command: $command"
            echo ""
            print_info "Available commands: install, uninstall, status, test, start, stop"
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 