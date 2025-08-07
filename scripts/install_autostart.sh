#!/bin/bash
# Quark Auto-Start Installation Script
# Sets up Quark for automatic startup and creates system-wide shortcuts

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
QUARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHAGENT_DIR="$HOME/Library/LaunchAgents"
LAUNCHAGENT_FILE="$LAUNCHAGENT_DIR/com.camdouglas.quark.plist"
BIN_DIR="/usr/local/bin"

echo -e "${BLUE}🚀 Quark Auto-Start Installation${NC}"
echo "=================================="
echo ""

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running as root for system-wide installation
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root - will install system-wide"
    BIN_DIR="/usr/local/bin"
else
    print_status "Installing for current user"
    BIN_DIR="$HOME/.local/bin"
    mkdir -p "$BIN_DIR"
fi

# Step 1: Verify Quark installation
print_status "Step 1: Verifying Quark installation..."
if [ ! -f "$QUARK_DIR/scripts/quark_optimized_startup.py" ]; then
    print_error "Quark installation not found at $QUARK_DIR"
    exit 1
fi

if [ ! -f "$QUARK_DIR/venv/bin/activate" ]; then
    print_error "Virtual environment not found. Please run: python3 -m venv venv"
    exit 1
fi

print_status "Quark installation verified"

# Step 2: Create system-wide launcher
print_status "Step 2: Creating system-wide launcher..."
LAUNCHER_PATH="$BIN_DIR/quark"

cat > "$LAUNCHER_PATH" << 'EOF'
#!/bin/bash
# Quark System Launcher
# Can be called from anywhere to start Quark

QUARK_DIR="/Users/camdouglas/quark"

if [ ! -d "$QUARK_DIR" ]; then
    echo "❌ Quark installation not found at $QUARK_DIR"
    exit 1
fi

exec "$QUARK_DIR/scripts/quark_shell.sh" "$@"
EOF

chmod +x "$LAUNCHER_PATH"
print_status "System launcher created at $LAUNCHER_PATH"

# Step 3: Install LaunchAgent for auto-start
print_status "Step 3: Installing LaunchAgent for auto-start..."

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$LAUNCHAGENT_DIR"

# Copy the plist file
cp "$QUARK_DIR/scripts/com.camdouglas.quark.plist" "$LAUNCHAGENT_FILE"

# Load the LaunchAgent
if launchctl list | grep -q "com.camdouglas.quark"; then
    print_warning "LaunchAgent already loaded, unloading first..."
    launchctl unload "$LAUNCHAGENT_FILE" 2>/dev/null || true
fi

launchctl load "$LAUNCHAGENT_FILE"
print_status "LaunchAgent installed and loaded"

# Step 4: Create shell profile integration
print_status "Step 4: Setting up shell integration..."

# Detect shell
SHELL_PROFILE=""
if [ -n "$ZSH_VERSION" ]; then
    SHELL_PROFILE="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_PROFILE="$HOME/.bashrc"
    if [ ! -f "$SHELL_PROFILE" ]; then
        SHELL_PROFILE="$HOME/.bash_profile"
    fi
fi

if [ -n "$SHELL_PROFILE" ]; then
    # Add Quark to PATH if not already there
    if ! grep -q "quark" "$SHELL_PROFILE"; then
        echo "" >> "$SHELL_PROFILE"
        echo "# Quark AI System" >> "$SHELL_PROFILE"
        echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$SHELL_PROFILE"
        echo "alias quark='$LAUNCHER_PATH'" >> "$SHELL_PROFILE"
        print_status "Shell profile updated: $SHELL_PROFILE"
    else
        print_warning "Quark already configured in shell profile"
    fi
fi

# Step 5: Create desktop shortcut (optional)
print_status "Step 5: Creating desktop shortcut..."

DESKTOP_SHORTCUT="$HOME/Desktop/Quark.command"
cat > "$DESKTOP_SHORTCUT" << EOF
#!/bin/bash
cd "$QUARK_DIR"
exec "$QUARK_DIR/scripts/quark_shell.sh"
EOF

chmod +x "$DESKTOP_SHORTCUT"
print_status "Desktop shortcut created: $DESKTOP_SHORTCUT"

# Step 6: Test the installation
print_status "Step 6: Testing installation..."

# Test the launcher
if "$LAUNCHER_PATH" status >/dev/null 2>&1; then
    print_status "Launcher test successful"
else
    print_warning "Launcher test failed (this is normal if Quark isn't running)"
fi

# Step 7: Final setup
print_status "Step 7: Final setup..."

# Create necessary directories
mkdir -p "$QUARK_DIR/logs"
mkdir -p "$QUARK_DIR/data"

# Set proper permissions
chmod +x "$QUARK_DIR/scripts/"*.sh
chmod +x "$QUARK_DIR/scripts/"*.py

print_status "Installation completed successfully!"

echo ""
echo -e "${BLUE}🎉 Quark Auto-Start Installation Complete!${NC}"
echo "================================================"
echo ""
echo "📋 What was installed:"
echo "  ✅ System-wide launcher: $LAUNCHER_PATH"
echo "  ✅ Auto-start daemon: $LAUNCHAGENT_FILE"
echo "  ✅ Desktop shortcut: $DESKTOP_SHORTCUT"
echo "  ✅ Shell integration: $SHELL_PROFILE"
echo ""
echo "🚀 Usage:"
echo "  • quark - Start Quark interactive shell"
echo "  • quark status - Check Quark status"
echo "  • Double-click Quark.command on desktop"
echo ""
echo "⚙️ Management:"
echo "  • $QUARK_DIR/scripts/quark_daemon.sh start|stop|restart|status"
echo "  • launchctl unload $LAUNCHAGENT_FILE (to disable auto-start)"
echo "  • launchctl load $LAUNCHAGENT_FILE (to enable auto-start)"
echo ""
echo "📝 Next steps:"
echo "  1. Restart your terminal or run: source $SHELL_PROFILE"
echo "  2. Quark will start automatically on system login"
echo "  3. Use 'quark' command from anywhere to start Quark"
echo "" 