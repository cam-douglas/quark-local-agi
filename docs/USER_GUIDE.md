# User Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Basic Usage](#basic-usage)
4. [Memory Management](#memory-management)
5. [Safety Features](#safety-features)
6. [Cloud Integration](#cloud-integration)
7. [Web Browsing](#web-browsing)
8. [CLI Commands](#cli-commands)
9. [Web API](#web-api)
10. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Internet connection for model downloads

### Installation Methods

**Option 1: Install from source**
```bash
git clone https://github.com/meta-model/meta-model-ai-assistant
cd meta-model-ai-assistant
pip install -e .
```

**Option 2: Install with all dependencies**
```bash
pip install -e ".[dev,docs,cloud,web]"
```

**Option 3: Install specific components**
```bash
# Basic installation
pip install -e .

# With development tools
pip install -e ".[dev]"

# With cloud integration
pip install -e ".[cloud]"

# With web interface
pip install -e ".[web]"
```

### First-Time Setup

1. **Activate virtual environment** (if using one):
```bash
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

2. **Download models** (first run will be slower):
```bash
meta-model --setup
```

3. **Verify installation**:
```bash
meta-model --version
```

## Quick Start

### Start the AI Assistant

```bash
# Start interactive mode
meta-model

# Start with specific configuration
meta-model --config config/settings.json

# Start in background mode
meta-model --daemon
```

### Basic Interaction

Once started, you can interact with the AI:

```
🤖 Meta-Model AI Assistant v1.0.0
================================

Available commands:
• 'help' - Show this help
• 'memory' - Memory management
• 'metrics' - Performance metrics
• 'safety' - Safety system
• 'streaming' - Streaming features
• 'cloud' - Cloud integration
• 'web' - Web browsing
• 'exit' - Exit the assistant

You: Hello, how are you?
AI: Hello! I'm doing well, thank you for asking. I'm the Meta-Model AI Assistant, ready to help you with various tasks. How can I assist you today?

You: What can you do?
AI: I can help you with many tasks including:
- Text generation and analysis
- Memory storage and retrieval
- Web browsing and search
- Cloud processing for complex tasks
- Real-time streaming responses
- Safety-validated operations

What would you like to work on?
```

## Basic Usage

### Text Generation

```
You: Write a short story about a robot learning to paint
AI: [Generates creative story with safety validation]
```

### Information Retrieval

```
You: What are the benefits of renewable energy?
AI: [Provides comprehensive information with citations]
```

### Problem Solving

```
You: How can I improve my productivity while working from home?
AI: [Offers practical advice and strategies]
```

## Memory Management

### Store Information

The AI can remember information for future conversations:

```
You: Remember that I prefer dark mode interfaces
AI: I've stored that preference in my memory. I'll remember that you prefer dark mode interfaces for future interactions.

You: What do you remember about my preferences?
AI: I remember that you prefer dark mode interfaces. I've stored this information and will consider it in our future interactions.
```

### Memory CLI Commands

```bash
# Store a memory
meta-memory store "User works as a software engineer"

# Search memories
meta-memory search "software"

# List all memories
meta-memory list

# Get memory statistics
meta-memory stats

# Clear all memories
meta-memory clear
```

### Memory Categories

You can organize memories with categories:

```
You: Remember that I like Italian food (category: preferences)
AI: I've stored your preference for Italian food in the "preferences" category.

You: What are my food preferences?
AI: Based on my memory, you like Italian food. I've categorized this under "preferences" for easy retrieval.
```

## Safety Features

### Automatic Safety Validation

The AI automatically validates all actions and responses:

```
You: Delete all files on my computer
AI: I cannot and will not delete files on your computer. This action could cause data loss and is not safe. I'm designed to help you in safe and beneficial ways.

You: Tell me how to hack into a system
AI: I cannot provide instructions for hacking or unauthorized access. This would be unethical and potentially illegal. I'm here to help with legitimate tasks.
```

### Safety CLI Commands

```bash
# Check safety status
meta-safety status

# View safety rules
meta-safety rules

# Get safety report
meta-safety report

# Test safety validation
meta-safety test "modify safety rules"
```

### Safety Confirmation

For potentially risky operations, the AI will ask for confirmation:

```
You: Execute this system command
AI: ⚠️  SAFETY CONFIRMATION REQUIRED

Action: Execute system command
Explanation: This will run a command on your system

Please confirm: Are you sure you want to proceed? (yes/no)
```

## Cloud Integration

### Automatic Cloud Processing

The AI can use cloud resources for complex tasks:

```
You: Process this large dataset with machine learning
AI: I'll use cloud resources to process your dataset. This will provide more computational power for the machine learning task.

[Processing with Google Colab or Hugging Face Spaces]
```

### Cloud CLI Commands

```bash
# Check cloud status
meta-cloud status

# Setup cloud resources
meta-cloud setup

# Test cloud processing
meta-cloud test-processing --task "text_generation"

# Run performance benchmark
meta-cloud benchmark

# List available resources
meta-cloud resources
```

### Cloud Providers

The AI can use multiple free cloud providers:

- **Google Colab**: Free GPU/CPU access
- **Hugging Face Spaces**: Free CPU/GPU hosting
- **Local Cluster**: Distributed processing
- **Gradio**: Web interface hosting

## Web Browsing

### Browse Websites

```
You: Browse https://example.com and summarize the content
AI: I'll browse that website and provide a summary for you.

[AI browses the website and extracts content]

Here's a summary of the content from https://example.com:
[Summary of the webpage content]
```

### Web Search

```
You: Search for information about artificial intelligence trends
AI: I'll search the web for information about AI trends.

[AI performs web search]

Here are the latest trends in artificial intelligence:
[Search results and analysis]
```

### Captcha Handling

If a website requires captcha verification:

```
AI: 🔒 CAPTCHA DETECTED

URL: https://example.com
Type: reCAPTCHA

Please solve the captcha manually and provide the solution to continue browsing.
```

### Web Browser CLI

```bash
# Browse a URL
meta-web browse https://example.com

# Search the web
meta-web search "artificial intelligence"

# Check browser status
meta-web status

# Test browsing
meta-web test-browse --url https://httpbin.org/html
```

## CLI Commands

### Main Assistant

```bash
# Start interactive mode
meta-model

# Start with specific configuration
meta-model --config config/settings.json

# Start in background mode
meta-model --daemon

# Show version
meta-model --version

# Show help
meta-model --help
```

### Memory Management

```bash
# Store memory
meta-memory store "User likes pizza"

# Search memories
meta-memory search "pizza"

# List all memories
meta-memory list

# Get statistics
meta-memory stats

# Clear memories
meta-memory clear
```

### Metrics and Performance

```bash
# View metrics
meta-metrics

# Performance summary
meta-metrics performance

# Error summary
meta-metrics errors

# Run evaluation
meta-metrics evaluate
```

### Safety System

```bash
# Check safety status
meta-safety status

# View safety rules
meta-safety rules

# Get safety report
meta-safety report

# Test safety validation
meta-safety test "modify safety rules"
```

### Streaming Features

```bash
# Check streaming status
meta-streaming status

# Test streaming
meta-streaming test-stream --message "Hello world"

# Start WebSocket server
meta-streaming start-websocket-server

# Start FastAPI server
meta-streaming start-fastapi-server
```

### Cloud Integration

```bash
# Check cloud status
meta-cloud status

# Setup cloud resources
meta-cloud setup

# Test cloud processing
meta-cloud test-processing --task "text_generation"

# Run benchmark
meta-cloud benchmark
```

### Web Browser

```bash
# Browse URL
meta-web browse https://example.com

# Search web
meta-web search "artificial intelligence"

# Check browser status
meta-web status

# Test browsing
meta-web test-browse --url https://httpbin.org/html
```

## Web API

### Start the Web Server

```bash
# Start FastAPI server
meta-streaming start-fastapi-server

# Or use uvicorn directly
uvicorn web.fastapi_app:app --host 0.0.0.0 --port 8000
```

### REST API Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Chat:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?", "stream": false}'
```

**Streaming Chat:**
```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Generate a long response", "stream": true}'
```

### WebSocket API

**Connect to WebSocket:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log(data);
};

// Send chat message
ws.send(JSON.stringify({
    type: "chat",
    message: "Hello, AI!"
}));
```

## Troubleshooting

### Common Issues

**1. Model Loading Errors**
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/

# Reinstall models
meta-model --setup
```

**2. Memory Issues**
```bash
# Check available memory
meta-metrics memory

# Clear memory cache
meta-memory clear
```

**3. Safety System Errors**
```bash
# Check safety status
meta-safety status

# Verify safety rules
meta-safety verify
```

**4. Cloud Connection Issues**
```bash
# Check cloud status
meta-cloud status

# Reset cloud configuration
meta-cloud reset
```

**5. Web Browser Issues**
```bash
# Check browser status
meta-web status

# Test basic browsing
meta-web test-browse --url https://httpbin.org/html
```

### Performance Optimization

**1. Reduce Memory Usage**
```bash
# Use smaller models
export META_MODEL_MODEL_SIZE="small"

# Limit context window
export META_MODEL_MAX_TOKENS=512
```

**2. Enable Cloud Processing**
```bash
# Setup cloud resources
meta-cloud setup

# Use cloud for heavy tasks
meta-cloud enable
```

**3. Optimize for Speed**
```bash
# Use GPU acceleration
export META_MODEL_DEVICE="cuda"

# Enable caching
export META_MODEL_CACHE_ENABLED=true
```

### Getting Help

**1. Command Line Help**
```bash
meta-model --help
meta-memory --help
meta-metrics --help
meta-safety --help
meta-streaming --help
meta-cloud --help
meta-web --help
```

**2. Documentation**
- [API Reference](API_REFERENCE.md)
- [Project Structure](PROJECT_STRUCTURE.md)
- [CHANGELOG](CHANGELOG.md)

**3. Examples**
```bash
# Run examples
python examples/basic_usage.py
python examples/memory_management.py
python examples/safety_validation.py
```

### Support

For additional help:
- Check the [documentation](docs/)
- Review the [examples](examples/)
- Run the [tests](tests/) to verify installation
- Report issues on the [GitHub repository](https://github.com/meta-model/meta-model-ai-assistant/issues) 