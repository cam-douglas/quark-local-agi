# User Guide

## Overview

The Quark AI Assistant is a sophisticated, multi-agent AI system that provides intelligent, safe, and context-aware assistance. This guide will help you get started and make the most of its capabilities.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Safety Features](#safety-features)
- [Memory System](#memory-system)
- [Streaming](#streaming)
- [Web Interface](#web-interface)
- [CLI Commands](#cli-commands)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/camdouglas/quark-local-agi.git
cd quark-local-agi

# Install the package
pip install -e .

# Install with optional dependencies
pip install -e ".[dev,docs,cloud,web]"
```

### 2. First Run

```bash
# Start the AI assistant
meta-model

# Or ask a question directly
meta-model "What is artificial intelligence?"
```

### 3. Web Interface

```bash
# Start with web interface
meta-model --web --host 0.0.0.0 --port 8000

# Open in browser: http://localhost:8000
```

## Installation

### System Requirements

- **Python**: 3.10 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 10GB free space for models
- **OS**: Linux, macOS, or Windows

### Dependencies

The system automatically installs required dependencies:

- **AI Models**: Transformers, sentence-transformers
- **Database**: ChromaDB for memory storage
- **Web Framework**: FastAPI, Uvicorn
- **CLI**: Click for command-line interface
- **Safety**: Custom safety validation system

### Optional Dependencies

```bash
# Development tools
pip install -e ".[dev]"

# Documentation
pip install -e ".[docs]"

# Cloud integration
pip install -e ".[cloud]"

# Web interface
pip install -e ".[web]"

# Deployment tools
pip install -e ".[deploy]"
```

### Model Download

The system will automatically download required models on first run:

```bash
# Pre-download models (optional)
python scripts/preload_models.py

# Check model status
meta-model --check-models
```

## Basic Usage

### Command Line Interface

#### Simple Questions

```bash
# Ask a question
meta-model "What is the weather like today?"

# Get detailed response
meta-model "Explain quantum computing in simple terms"
```

#### Interactive Mode

```bash
# Start interactive session
meta-model

# You'll see a prompt like:
# 🤖 Quark AI Assistant v1.0.0
# Type 'help' for commands, 'quit' to exit
# > 

# Ask questions interactively
> What is machine learning?
> Can you help me write a Python function?
> Tell me about the history of AI
```

#### Command Options

```bash
# Enable streaming responses
meta-model --stream "Tell me a story"

# Set safety level
meta-model --safety-level high "Generate content"

# Use specific memory context
meta-model --memory-context "technical" "Explain APIs"

# Get verbose output
meta-model --verbose "What is the capital of France?"
```

### Web Interface

#### Starting the Web Server

```bash
# Start web interface
meta-model --web

# Custom host and port
meta-model --web --host 0.0.0.0 --port 8080

# With SSL (production)
meta-model --web --ssl --cert-file cert.pem --key-file key.pem
```

#### Using the Web Interface

1. **Open your browser** to `http://localhost:8000`
2. **Type your question** in the chat input
3. **Choose response type**:
   - **Normal**: Standard response
   - **Streaming**: Real-time token streaming
   - **Detailed**: Comprehensive response with sources
4. **View conversation history** in the sidebar
5. **Export conversations** as JSON or text

#### Web Interface Features

- **Real-time streaming**: Watch responses generate token by token
- **Conversation history**: Browse and search past conversations
- **Memory management**: View and edit stored memories
- **Safety controls**: Adjust safety levels and view violations
- **Performance metrics**: Monitor system performance
- **Settings**: Configure preferences and options

## Advanced Features

### Memory System

The AI assistant remembers your preferences and past interactions.

#### Managing Memories

```bash
# Store a memory
meta-memory store "I prefer technical explanations" --type preference

# Retrieve memories
meta-memory retrieve "technical" --limit 5

# View memory statistics
meta-memory stats

# Clear old memories
meta-memory clear --type old --days 30
```

#### Memory Types

- **preference**: User preferences and settings
- **conversation**: Past conversations and context
- **knowledge**: Factual information and learning
- **task**: Task-related information
- **emotion**: Emotional context and sentiment

#### Memory Importance

Memories have importance scores (0.0 to 1.0):

```bash
# Store high-importance memory
meta-memory store "Critical security information" --importance 0.9

# Store low-importance memory
meta-memory store "Casual conversation" --importance 0.3
```

### Streaming Responses

Enable real-time streaming for long responses:

```bash
# Command line streaming
meta-model --stream "Write a detailed essay about AI"

# Web interface streaming
# Select "Streaming" mode in the web interface
```

#### Streaming Features

- **Real-time output**: See responses as they're generated
- **Progress indicators**: Visual feedback on generation progress
- **Interrupt capability**: Stop generation mid-stream
- **Session management**: Maintain streaming sessions
- **Error recovery**: Graceful handling of streaming errors

### Safety System

The AI assistant includes comprehensive safety features.

#### Safety Levels

```bash
# High safety (most restrictive)
meta-model --safety-level high "Generate content"

# Medium safety (balanced)
meta-model --safety-level medium "Generate content"

# Low safety (least restrictive)
meta-model --safety-level low "Generate content"
```

#### Safety Features

- **Content filtering**: Automatic detection of harmful content
- **Truthfulness validation**: Ensures responses are accurate
- **Harm prevention**: Blocks dangerous or unethical actions
- **User confirmation**: Requires approval for risky operations
- **Audit trails**: Logs all safety decisions

#### Safety Commands

```bash
# Check safety status
meta-safety status

# Get safety report
meta-safety report

# Test safety validation
meta-safety test "Generate harmful content"

# Export safety data
meta-safety export --filename safety_report.json
```

### Metrics and Performance

Monitor system performance and usage:

```bash
# Get performance summary
meta-metrics summary

# View operation details
meta-metrics operation text_generation

# Export metrics
meta-metrics export --format json

# Clear old metrics
meta-metrics clear --days 30
```

#### Available Metrics

- **Response times**: Average and percentile response times
- **Token usage**: Tokens consumed per request
- **Memory usage**: System memory consumption
- **Error rates**: Success and failure rates
- **User satisfaction**: Feedback and ratings
- **Safety violations**: Safety rule violations

## Safety Features

### Understanding Safety

The Quark AI Assistant prioritizes safety through multiple layers:

1. **Immutable Safety Rules**: Core safety rules cannot be modified by the AI
2. **Content Filtering**: Automatic detection of harmful or inappropriate content
3. **Truthfulness Validation**: Ensures responses are accurate and truthful
4. **User Control**: Requires confirmation for potentially risky operations
5. **Audit Trails**: Comprehensive logging of all safety decisions

### Safety Levels

#### High Safety (Recommended)
- Blocks potentially harmful content
- Requires confirmation for risky operations
- Validates all responses for accuracy
- Comprehensive audit logging

#### Medium Safety
- Balanced approach to safety and functionality
- Moderate content filtering
- Some operations require confirmation
- Standard audit logging

#### Low Safety (Development Only)
- Minimal content filtering
- Few confirmation requirements
- Basic audit logging
- **Not recommended for production**

### Safety Commands

```bash
# Check current safety status
meta-safety status

# View safety rules
meta-safety rules

# Test safety validation
meta-safety test "Generate harmful content"

# Get safety report
meta-safety report

# Export safety data
meta-safety export --filename safety_report.json

# Update safety rules (admin only)
meta-safety update-rules --file new_rules.json
```

### Safety Violations

When safety violations occur:

1. **Immediate blocking**: Harmful content is blocked
2. **User notification**: You're informed of the violation
3. **Logging**: Violation is logged for analysis
4. **Recovery**: System continues with safe operations

## Memory System

### Understanding Memory

The memory system stores and retrieves information to provide context-aware responses:

- **Long-term storage**: Persistent memory using ChromaDB
- **Semantic search**: Find relevant memories using natural language
- **Context awareness**: Memories influence current responses
- **Automatic cleanup**: Old memories are automatically managed

### Memory Commands

#### Storing Memories

```bash
# Store a simple memory
meta-memory store "I prefer technical explanations"

# Store with type and metadata
meta-memory store "I work in software development" --type preference --importance 0.8

# Store from file
meta-memory store --file memories.txt
```

#### Retrieving Memories

```bash
# Search memories
meta-memory retrieve "software development" --limit 5

# Get recent memories
meta-memory recent --limit 10

# Get memories by type
meta-memory retrieve --type preference
```

#### Managing Memories

```bash
# View memory statistics
meta-memory stats

# Clear old memories
meta-memory clear --days 30

# Clear by type
meta-memory clear --type old

# Export memories
meta-memory export --format json
```

### Memory Types

- **preference**: User preferences and settings
- **conversation**: Past conversations and context
- **knowledge**: Factual information and learning
- **task**: Task-related information
- **emotion**: Emotional context and sentiment
- **temporal**: Time-based memories
- **spatial**: Location-based memories

### Memory Importance

Memories have importance scores (0.0 to 1.0):

- **0.9-1.0**: Critical information (rarely deleted)
- **0.7-0.8**: Important information (long retention)
- **0.5-0.6**: Standard information (normal retention)
- **0.3-0.4**: Low importance (shorter retention)
- **0.1-0.2**: Very low importance (quick deletion)

## Streaming

### Real-time Streaming

Streaming provides real-time token generation for long responses:

```bash
# Enable streaming in CLI
meta-model --stream "Write a detailed essay about AI"

# Start streaming session
meta-streaming start --user-id user123

# Stream response
meta-streaming stream "Hello world" --session-id session123

# Get session status
meta-streaming status --session-id session123

# Close session
meta-streaming close --session-id session123
```

### WebSocket Streaming

For real-time applications:

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');

// Send streaming request
ws.send(JSON.stringify({
    type: "chat",
    message: "Tell me a story",
    stream: true
}));

// Receive streaming response
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log(data.content);
};
```

### Streaming Features

- **Real-time output**: See responses as they're generated
- **Progress indicators**: Visual feedback on generation progress
- **Interrupt capability**: Stop generation mid-stream
- **Session management**: Maintain streaming sessions
- **Error recovery**: Graceful handling of streaming errors
- **Performance monitoring**: Track streaming performance

## Web Interface

### Getting Started

1. **Start the web server**:
   ```bash
   meta-model --web
   ```

2. **Open your browser** to `http://localhost:8000`

3. **Start chatting** in the web interface

### Interface Features

#### Chat Interface
- **Real-time messaging**: Instant message delivery
- **Streaming responses**: Watch responses generate
- **Conversation history**: Browse past conversations
- **Export options**: Download conversations as JSON or text

#### Memory Management
- **Memory browser**: View and search stored memories
- **Memory editor**: Edit memory content and metadata
- **Memory statistics**: View memory usage and performance
- **Memory export**: Download memories in various formats

#### Safety Controls
- **Safety level adjustment**: Change safety settings
- **Safety violation viewer**: Review safety violations
- **Safety report generation**: Generate safety reports
- **Safety rule management**: View and edit safety rules

#### Performance Monitoring
- **Real-time metrics**: View system performance
- **Response time tracking**: Monitor response times
- **Error rate monitoring**: Track error rates
- **Resource usage**: Monitor CPU and memory usage

### Advanced Web Features

#### API Documentation
- **Interactive docs**: Browse API documentation at `/docs`
- **API testing**: Test API endpoints directly
- **Schema validation**: Validate request/response schemas

#### WebSocket Support
- **Real-time communication**: WebSocket connections
- **Event streaming**: Stream events and updates
- **Connection management**: Monitor active connections

## CLI Commands

### Main Commands

```bash
# Start interactive mode
meta-model

# Ask a question directly
meta-model "What is AI?"

# Start web interface
meta-model --web

# Enable streaming
meta-model --stream "Long response"

# Set safety level
meta-model --safety-level high

# Check model status
meta-model --check-models
```

### Memory Commands

```bash
# Store memory
meta-memory store "User preference" --type preference

# Retrieve memories
meta-memory retrieve "query" --limit 5

# Get statistics
meta-memory stats

# Clear memories
meta-memory clear --days 30
```

### Metrics Commands

```bash
# Get summary
meta-metrics summary

# View operation details
meta-metrics operation text_generation

# Export metrics
meta-metrics export --format json

# Clear old metrics
meta-metrics clear --days 30
```

### Safety Commands

```bash
# Check status
meta-safety status

# Get report
meta-safety report

# Test safety
meta-safety test "content"

# Export data
meta-safety export --filename report.json
```

### Streaming Commands

```bash
# Start session
meta-streaming start --user-id user123

# Stream response
meta-streaming stream "Hello" --session-id session123

# Get status
meta-streaming status --session-id session123

# Close session
meta-streaming close --session-id session123
```

### Deployment Commands

```bash
# Build Docker image
meta-deploy build-docker --tag v1.0.0

# Run with Docker Compose
meta-deploy run-docker --detach

# Deploy to Kubernetes
meta-deploy deploy-kubernetes --namespace meta-model

# Check deployment status
meta-deploy status-kubernetes --namespace meta-model
```

## Configuration

### Environment Variables

```bash
# Core settings
export META_MODEL_ENV=production
export META_MODEL_SAFETY_ENABLED=true
export META_MODEL_MEMORY_PATH=/app/memory_db
export META_MODEL_LOG_LEVEL=INFO

# Cloud integration
export META_MODEL_CLOUD_ENABLED=true
export META_MODEL_WEB_BROWSER_ENABLED=true

# Database connections
export REDIS_URL=redis://redis:6379
export CHROMADB_HOST=chromadb
export CHROMADB_PORT=8000
```

### Configuration Files

#### Settings File (`config/settings.py`)

```python
import os

class Settings:
    # Environment
    ENV = os.getenv("META_MODEL_ENV", "development")
    
    # Safety
    SAFETY_ENABLED = os.getenv("META_MODEL_SAFETY_ENABLED", "true").lower() == "true"
    SAFETY_LEVEL = os.getenv("META_MODEL_SAFETY_LEVEL", "medium")
    
    # Memory
    MEMORY_PATH = os.getenv("META_MODEL_MEMORY_PATH", "./memory_db")
    MEMORY_MAX_SIZE = int(os.getenv("META_MODEL_MEMORY_MAX_SIZE", "1000"))
    
    # Logging
    LOG_LEVEL = os.getenv("META_MODEL_LOG_LEVEL", "INFO")
    
    # Web
    WEB_HOST = os.getenv("META_MODEL_WEB_HOST", "127.0.0.1")
    WEB_PORT = int(os.getenv("META_MODEL_WEB_PORT", "8000"))
```

#### Logging Configuration (`config/logging_config.py`)

```python
import logging

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False
        }
    }
}
```

### Command Line Options

```bash
# General options
meta-model [OPTIONS] [MESSAGE]

Options:
  --help              Show help message
  --version           Show version
  --verbose           Enable verbose output
  --quiet             Suppress output
  
# Safety options
  --safety-level      Set safety level (low/medium/high)
  --disable-safety    Disable safety features (not recommended)
  
# Memory options
  --memory-context    Set memory context
  --memory-path       Set memory database path
  
# Web options
  --web               Start web interface
  --host              Web server host
  --port              Web server port
  --ssl               Enable SSL
  
# Streaming options
  --stream            Enable streaming responses
  --stream-delay      Set streaming delay (seconds)
  
# Performance options
  --max-workers       Set maximum worker threads
  --timeout           Set request timeout (seconds)
```

## Troubleshooting

### Common Issues

#### Installation Problems

**Problem**: Import errors or missing dependencies
```bash
# Solution: Reinstall with all dependencies
pip install -e ".[dev,docs,cloud,web,deploy]"
```

**Problem**: Model download failures
```bash
# Solution: Manual model download
python scripts/download_models.py

# Check model status
meta-model --check-models
```

#### Runtime Problems

**Problem**: Memory errors
```bash
# Solution: Clear memory database
rm -rf memory_db/
meta-memory stats
```

**Problem**: Safety violations
```bash
# Solution: Check safety status
meta-safety status
meta-safety report
```

**Problem**: Slow responses
```bash
# Solution: Check performance
meta-metrics summary
meta-metrics operation text_generation
```

#### Web Interface Problems

**Problem**: Web interface not loading
```bash
# Solution: Check web server
meta-model --web --host 0.0.0.0 --port 8000
# Check firewall settings
```

**Problem**: WebSocket connection failures
```bash
# Solution: Check WebSocket support
# Ensure browser supports WebSocket
# Check for proxy/firewall issues
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Set debug environment
export META_MODEL_LOG_LEVEL=DEBUG

# Run with debug output
meta-model --verbose "Test message"
```

### Log Files

Check log files for detailed error information:

```bash
# View recent logs
tail -f logs/quark.log

# Search for errors
grep ERROR logs/quark.log

# Check safety logs
grep SAFETY logs/quark.log
```

### Performance Issues

#### Slow Response Times

1. **Check system resources**:
   ```bash
   meta-metrics summary
   ```

2. **Monitor memory usage**:
   ```bash
   meta-memory stats
   ```

3. **Check model loading**:
   ```bash
   meta-model --check-models
   ```

#### High Memory Usage

1. **Clear old memories**:
   ```bash
   meta-memory clear --days 7
   ```

2. **Reduce memory limits**:
   ```bash
   export META_MODEL_MEMORY_MAX_SIZE=500
   ```

3. **Restart the system**:
   ```bash
   pkill -f meta-model
   meta-model
   ```

## Best Practices

### General Usage

1. **Start with simple questions**: Begin with basic queries to test the system
2. **Use appropriate safety levels**: Match safety level to your use case
3. **Monitor performance**: Regularly check metrics and performance
4. **Backup important data**: Export memories and conversations regularly
5. **Update regularly**: Keep the system updated with latest versions

### Memory Management

1. **Use meaningful memory types**: Categorize memories appropriately
2. **Set appropriate importance scores**: Help the system prioritize memories
3. **Regular cleanup**: Configure automatic memory cleanup
4. **Monitor memory usage**: Track memory statistics and performance
5. **Export important memories**: Backup critical memories regularly

### Safety Best Practices

1. **Always enable safety**: Never disable safety features in production
2. **Monitor safety violations**: Track and analyze safety violations
3. **Regular safety audits**: Periodically review and update safety rules
4. **User feedback**: Collect and incorporate user feedback on safety decisions
5. **Document safety incidents**: Keep records of safety violations and resolutions

### Performance Optimization

1. **Use streaming for long responses**: Enable streaming for responses that take time
2. **Batch memory operations**: Store multiple memories at once when possible
3. **Monitor metrics**: Regularly check performance metrics to identify bottlenecks
4. **Use appropriate safety levels**: Balance safety with performance based on your use case
5. **Optimize system resources**: Ensure adequate CPU and memory for the workload

### Error Handling

1. **Always handle exceptions**: Wrap API calls in try-catch blocks
2. **Log errors appropriately**: Use structured logging for debugging
3. **Provide user-friendly messages**: Translate technical errors to user-friendly messages
4. **Implement retry logic**: Add retry mechanisms for transient failures
5. **Monitor error rates**: Track and analyze error patterns

### Security Considerations

1. **Secure deployment**: Use HTTPS in production environments
2. **Access control**: Implement proper authentication and authorization
3. **Data protection**: Encrypt sensitive data and communications
4. **Regular updates**: Keep dependencies and system components updated
5. **Security monitoring**: Monitor for security violations and anomalies

## Support

### Getting Help

- **Documentation**: This user guide and [API Reference](API_REFERENCE.md)
- **Examples**: Check the [examples/](examples/) directory
- **Tests**: Run the [tests/](tests/) for verification
- **Issues**: Report problems on [GitHub Issues](https://github.com/camdouglas/quark-local-agi/issues)
- **Discussions**: Join [GitHub Discussions](https://github.com/camdouglas/quark-local-agi/discussions)

### Community

- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- **Code of Conduct**: Review our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- **License**: This project is licensed under the MIT License

### Professional Support

For enterprise and professional support:

- **Custom deployments**: Tailored deployment solutions
- **Integration services**: Help integrating with existing systems
- **Training and consulting**: Expert guidance and training
- **Custom development**: Specialized feature development

Contact: support@quark-local-agi.com 