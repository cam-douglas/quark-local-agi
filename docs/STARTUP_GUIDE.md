# Quark AI Assistant Startup Guide

This guide explains how to set up the Quark AI Assistant to start automatically with your laptop and be ready for input.

## 🚀 Quick Setup

### 1. Install Startup Service

```bash
cd /Users/camdouglas/quark
./scripts/install_startup.sh install
```

### 2. Check Status

```bash
./scripts/install_startup.sh status
```

### 3. Quick Status Check

```bash
./scripts/check_ready.sh
# or use the alias
meta-ready
```

## 📋 What Gets Installed

### macOS LaunchAgent
- **File**: `~/Library/LaunchAgents/com.metamodel.ai.assistant.plist`
- **Purpose**: Automatically starts the AI assistant when you log in
- **Behavior**: Runs in background, keeps model loaded and ready

### Startup Daemon
- **File**: `scripts/startup_daemon.sh`
- **Purpose**: Manages the startup sequence and ensures model readiness
- **Features**:
  - Environment setup and validation
  - Model loading and initialization
  - Process monitoring and auto-restart
  - Ready flag management

## 🔧 Startup Sequence

1. **System Boot** → macOS loads LaunchAgent
2. **LaunchAgent** → Starts `startup_daemon.sh`
3. **Daemon** → Sets up environment and loads models
4. **Ready Flag** → Creates `logs/model_ready.flag` when ready
5. **Monitoring** → Continuously monitors and restarts if needed

## 📊 Status Indicators

### Ready Flag
- **File**: `logs/model_ready.flag`
- **Meaning**: Model is loaded and ready for input
- **Check**: `./scripts/check_ready.sh`

### Process ID
- **File**: `logs/quark.pid`
- **Meaning**: Background process is running
- **Check**: `ps aux | grep quark`

### LaunchAgent Status
- **Check**: `launchctl list | grep metamodel`
- **Meaning**: Startup service is loaded

## 🛠️ Management Commands

### Installation
```bash
./scripts/install_startup.sh install    # Install startup service
./scripts/install_startup.sh uninstall  # Remove startup service
```

### Status & Testing
```bash
./scripts/install_startup.sh status     # Show detailed status
./scripts/install_startup.sh test       # Test startup sequence
./scripts/check_ready.sh                # Quick status check
```

### Manual Control
```bash
./scripts/install_startup.sh start      # Start manually
./scripts/install_startup.sh stop       # Stop manually
```

### Shell Aliases
```bash
meta-ready      # Quick status check
meta-status     # Detailed status
meta-check      # Environment check
meta-setup      # Setup environment
meta-download   # Download models
```

## 📁 File Structure

```
quark/
├── scripts/
│   ├── install_startup.sh          # Installation script
│   ├── startup_daemon.sh           # Startup daemon
│   ├── check_ready.sh              # Quick status check
│   └── mac_startup_agent.plist     # LaunchAgent configuration
├── logs/
│   ├── startup.log                 # Startup logs
│   ├── assistant.log               # Assistant logs
│   ├── model_ready.flag           # Ready indicator
│   └── quark.pid             # Process ID
└── ~/Library/LaunchAgents/
    └── com.metamodel.ai.assistant.plist  # macOS LaunchAgent
```

## 🔍 Troubleshooting

### Model Not Ready
```bash
# Check logs
tail -f logs/startup.log
tail -f logs/assistant.log

# Restart manually
./scripts/install_startup.sh stop
./scripts/install_startup.sh start
```

### LaunchAgent Issues
```bash
# Check LaunchAgent status
launchctl list | grep metamodel

# Reload LaunchAgent
launchctl unload ~/Library/LaunchAgents/com.metamodel.ai.assistant.plist
launchctl load ~/Library/LaunchAgents/com.metamodel.ai.assistant.plist
```

### Environment Issues
```bash
# Check environment
./scripts/install_startup.sh test

# Reinstall
./scripts/install_startup.sh uninstall
./scripts/install_startup.sh install
```

## ⚡ Performance Notes

- **Startup Time**: ~30-60 seconds for first run (model loading)
- **Memory Usage**: ~2-4GB RAM (model storage)
- **CPU Usage**: Low when idle, spikes during inference
- **Disk Space**: ~5-10GB for models and logs

## 🔒 Security

- **User Permissions**: Runs under your user account
- **Network Access**: Only for model downloads (first time)
- **File Access**: Limited to project directory
- **System Integration**: Uses standard macOS LaunchAgent system

## 📝 Log Files

- **Startup Log**: `logs/startup.log` - Daemon startup sequence
- **Assistant Log**: `logs/assistant.log` - AI assistant output
- **Error Log**: `logs/startup_error.log` - LaunchAgent errors

## 🎯 Ready State

The model is considered "ready" when:
1. ✅ LaunchAgent is loaded
2. ✅ Background process is running
3. ✅ Ready flag exists
4. ✅ Models are loaded in memory

You can check this with: `./scripts/check_ready.sh`

## 🚀 Next Steps

Once the startup service is installed:

1. **Restart your computer** to test automatic startup
2. **Open a new terminal** and run `meta-ready` to check status
3. **Start using the AI assistant** with `quark` command
4. **Monitor logs** if you encounter any issues

The Quark AI Assistant will now be ready for input every time you start your laptop! 🤖 