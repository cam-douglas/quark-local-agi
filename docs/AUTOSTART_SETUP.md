# Quark AI System - Auto-Start Setup Guide

## 🎯 Overview

This guide explains how Quark is configured to start automatically on system boot and provide immediate terminal access. The system is designed for maximum efficiency with parallel agent loading and immediate interactive shell availability.

## 🚀 Quick Start

### For New Users
1. **Open a new terminal window**
2. **Type `quark`** - This starts the interactive shell immediately
3. **Start chatting!** - Quark is ready while agents load in background

### For Desktop Users
1. **Double-click `Quark.command`** on your desktop
2. **Start chatting immediately** - No waiting required

## 📁 Installation Components

### System-Wide Launcher
- **Location**: `/Users/camdouglas/.local/bin/quark`
- **Purpose**: Allows starting Quark from any directory
- **Usage**: `quark` or `quark status`

### Auto-Start Daemon
- **Location**: `$HOME/Library/LaunchAgents/com.camdouglas.quark.plist`
- **Purpose**: Starts Quark automatically on system login
- **Management**: `launchctl load/unload` commands

### Desktop Shortcut
- **Location**: `$HOME/Desktop/Quark.command`
- **Purpose**: One-click Quark startup
- **Usage**: Double-click to start

### Shell Integration
- **Location**: `$HOME/.bashrc`
- **Purpose**: Adds `quark` command to PATH
- **Activation**: Automatic on new terminal windows

## ⚙️ Technical Architecture

### Optimized Startup Process

1. **Immediate Shell Access**
   - Interactive shell starts instantly
   - No waiting for agent loading
   - User can start chatting immediately

2. **Parallel Agent Loading**
   - All agents load simultaneously in background
   - Uses ThreadPoolExecutor with 8 workers
   - 30-second timeout per agent
   - Graceful handling of missing agents

3. **Background Processing**
   - Agents continue loading while user interacts
   - Progress tracking available via `status` command
   - Ready flag written when complete

### Agent Loading Strategy

```python
# Core agents loaded in parallel
agent_classes = [
    ("MemoryAgent", "memory"),
    ("MetricsAgent", "metrics"),
    ("NLUAgent", "nlu"),
    ("RetrievalAgent", "retrieval"),
    ("PlanningAgent", "planning"),
    ("ReasoningAgent", "reasoning"),
    # ... and more
]

# Dynamic import with error handling
try:
    module = __import__(f"agents.{name}_agent", fromlist=[agent_name])
    agent_class = getattr(module, agent_name)
    agent = agent_class()
except ImportError:
    # Agent not available, continue gracefully
    pass
```

## 🎮 Usage Commands

### Interactive Shell Commands
```
help     - Show available commands
status   - Show system status and loading progress
agents   - List loaded agents with status
memory   - Show memory statistics
metrics  - Show performance metrics
quit     - Exit Quark
```

### System Management Commands
```bash
# Start Quark
quark

# Check status
quark status

# Daemon management
./scripts/quark_daemon.sh start
./scripts/quark_daemon.sh stop
./scripts/quark_daemon.sh restart
./scripts/quark_daemon.sh status

# Auto-start management
launchctl load $HOME/Library/LaunchAgents/com.camdouglas.quark.plist
launchctl unload $HOME/Library/LaunchAgents/com.camdouglas.quark.plist
```

## 🔧 Configuration Files

### LaunchAgent Configuration
```xml
<!-- $HOME/Library/LaunchAgents/com.camdouglas.quark.plist -->
<key>ProgramArguments</key>
<array>
    <string>/Users/camdouglas/quark/scripts/quark_daemon.sh</string>
    <string>start</string>
</array>
<key>RunAtLoad</key>
<true/>
<key>KeepAlive</key>
<true/>
```

### Shell Profile Integration
```bash
# Added to $HOME/.bashrc
export PATH="/Users/camdouglas/.local/bin:$PATH"
alias quark='/Users/camdouglas/.local/bin/quark'
```

## 📊 Performance Features

### Parallel Loading
- **8 concurrent workers** for agent loading
- **30-second timeout** per agent
- **Graceful degradation** for missing agents
- **Progress tracking** with real-time status

### Memory Optimization
- **Lazy loading** of heavy models
- **Shared model instances** where possible
- **Automatic cleanup** on shutdown
- **Memory monitoring** via metrics agent

### Startup Time Optimization
- **Immediate shell access** (0-1 seconds)
- **Background agent loading** (30-60 seconds total)
- **Progressive functionality** as agents become available
- **Status indicators** show loading progress

## 🛠️ Troubleshooting

### Common Issues

**Problem**: `quark` command not found
```bash
# Solution: Reload shell profile
source ~/.bashrc
```

**Problem**: Quark won't start
```bash
# Solution: Restart daemon
./scripts/quark_daemon.sh restart
```

**Problem**: Auto-start not working
```bash
# Solution: Reload LaunchAgent
launchctl unload $HOME/Library/LaunchAgents/com.camdouglas.quark.plist
launchctl load $HOME/Library/LaunchAgents/com.camdouglas.quark.plist
```

**Problem**: Check startup logs
```bash
# View startup logs
tail -f logs/quark_optimized_startup.log
```

### Debug Commands

```bash
# Check if Quark is running
ps aux | grep quark

# Check LaunchAgent status
launchctl list | grep quark

# Check PID file
cat logs/quark.pid

# Check ready flag
cat logs/quark_ready.flag
```

## 🎯 Pillar Integration

### All Pillars Operational
The system ensures all Quark pillars are fully operational:

1. **Foundation (Pillars 1-4)**: Project structure, environment, CLI, testing
2. **Core Framework (Pillars 5-8)**: Architecture, agents, communication, safety
3. **Advanced Features (Pillars 9-12)**: Memory, learning, alignment, orchestration
4. **Intelligence Enhancement (Pillars 13-16)**: Async, UI, safety, meta-learning
5. **AGI Capabilities (Pillars 17-21)**: Memory, reasoning, social, autonomy, governance
6. **Superintelligence (Pillars 24-26)**: Advanced reasoning, meta-cognition, self-improvement
7. **Advanced Intelligence (Pillars 27-33)**: Explainability, negotiation, tools, decisions, creativity, emotions, social understanding

### Pillar Status Monitoring
```bash
# Check pillar status
python3 tests/comprehensive_pillar_check.py

# Test specific pillars
python3 tests/test_pillars_5_6_7_8.py
```

## 🚀 Future Enhancements

### Planned Improvements
- **Model caching** for faster subsequent startups
- **Incremental loading** based on usage patterns
- **Cloud model integration** for reduced local load
- **Advanced monitoring** with web dashboard
- **Plugin system** for custom agent integration

### Performance Targets
- **Target startup time**: < 30 seconds for full functionality
- **Interactive shell**: < 1 second
- **Memory usage**: < 4GB total
- **CPU usage**: < 50% during startup

## 📝 Summary

Quark is now configured for:

✅ **Automatic startup** on system login
✅ **Immediate terminal access** with `quark` command
✅ **Parallel agent loading** for optimal performance
✅ **All pillars operational** by default
✅ **Graceful error handling** for missing components
✅ **Comprehensive monitoring** and status tracking
✅ **Multiple access methods** (terminal, desktop, daemon)

The system provides a seamless, efficient experience where users can start interacting with Quark immediately while the full AI system loads in the background. 