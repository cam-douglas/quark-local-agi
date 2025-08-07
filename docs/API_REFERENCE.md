# API Reference

## Overview

The Quark AI Assistant provides a comprehensive API for interacting with the multi-agent AI system. This document covers all available endpoints, classes, and methods.

## Table of Contents

- [Core Components](#core-components)
- [Agent API](#agent-api)
- [CLI Commands](#cli-commands)
- [Web API](#web-api)
- [Safety API](#safety-api)
- [Memory API](#memory-api)
- [Metrics API](#metrics-api)
- [Streaming API](#streaming-api)

## Core Components

### Orchestrator

The main orchestrator coordinates all agents and handles request routing.

```python
from core.orchestrator import Orchestrator

# Initialize orchestrator
orchestrator = Orchestrator(max_workers=4)

# Handle a request
response = orchestrator.handle("What is the weather like today?")

# Get performance stats
stats = orchestrator.get_performance_stats()
```

#### Methods

- `handle(prompt: str) -> Dict[str, Any]`: Process a user request
- `get_performance_stats() -> Dict[str, Any]`: Get execution statistics
- `shutdown()`: Clean shutdown of all agents

### Router

The router determines which agents should handle a request.

```python
from core.router import Router

router = Router()
category = router.classify_intent("What is the capital of France?")
```

#### Methods

- `classify_intent(text: str) -> str`: Classify the intent of user input
- `get_pipeline(category: str) -> List[str]`: Get agent pipeline for category

## Agent API

### Base Agent

All agents inherit from the base Agent class.

```python
from agents.base import Agent

class CustomAgent(Agent):
    def __init__(self, model_name: str):
        super().__init__(model_name)
    
    def load_model(self):
        # Load your model here
        pass
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        # Generate response
        pass
```

### NLU Agent

Natural Language Understanding agent for intent classification.

```python
from agents.nlu_agent import NLUAgent

nlu_agent = NLUAgent(model_name="facebook/bart-large-mnli")
result = nlu_agent.generate("What is the weather like?", operation="classify_intent")
```

### Memory Agent

Long-term memory storage and retrieval.

```python
from agents.memory_agent import MemoryAgent

memory_agent = MemoryAgent()
memory_id = memory_agent.store_memory(
    content="User likes pizza",
    memory_type="preference",
    metadata={"importance": 0.8}
)

memories = memory_agent.retrieve_memories("pizza", n_results=5)
```

#### Methods

- `store_memory(content: str, memory_type: str, metadata: Dict) -> str`: Store a memory
- `retrieve_memories(query: str, n_results: int) -> List[Dict]`: Retrieve relevant memories
- `get_memory_stats() -> Dict[str, Any]`: Get memory statistics

### Metrics Agent

Performance monitoring and evaluation.

```python
from agents.metrics_agent import MetricsAgent

metrics_agent = MetricsAgent()
op_id = metrics_agent.start_operation("test_operation", "test input")
metrics_agent.end_operation(op_id, success=True, tokens_used=100)
```

#### Methods

- `start_operation(operation: str, input_data: str) -> str`: Start tracking an operation
- `end_operation(operation_id: str, success: bool, **kwargs)`: End operation tracking
- `get_performance_summary() -> Dict[str, Any]`: Get performance summary

### Safety Agent

Safety validation and content filtering.

```python
from agents.safety_agent import SafetyAgent

safety_agent = SafetyAgent()
result = safety_agent.generate("Generate harmful content", operation="assess_safety")
```

#### Methods

- `assess_safety(content: str) -> Dict[str, Any]`: Assess safety of content
- `filter_content(content: str) -> Dict[str, Any]`: Filter inappropriate content

### Streaming Agent

Real-time streaming capabilities.

```python
from agents.streaming_agent import StreamingAgent

streaming_agent = StreamingAgent()
session_id = streaming_agent.create_session(user_id="user123")

async for event in streaming_agent.stream_response("Hello", response_generator, session_id):
    print(f"Event: {event.content}")
```

#### Methods

- `create_session(user_id: str) -> str`: Create a streaming session
- `stream_response(prompt: str, generator: Callable, session_id: str)`: Stream response
- `close_session(session_id: str)`: Close a session

## CLI Commands

### Main CLI

```bash
# Start the AI assistant
meta-model

# Ask a question directly
meta-model "What is the weather like?"

# Start with web interface
meta-model --web --host 0.0.0.0 --port 8000

# Enable streaming
meta-model --stream

# Set safety level
meta-model --safety-level high
```

### Memory CLI

```bash
# Store a memory
meta-memory store "User likes pizza" --type preference

# Retrieve memories
meta-memory retrieve "pizza" --limit 5

# Get memory stats
meta-memory stats

# Clear memories
meta-memory clear --type old
```

### Metrics CLI

```bash
# Get performance summary
meta-metrics summary

# Get operation details
meta-metrics operation test_operation

# Export metrics
meta-metrics export --format json

# Clear old metrics
meta-metrics clear --days 30
```

### Safety CLI

```bash
# Assess content safety
meta-safety assess "Generate harmful content"

# Get safety report
meta-safety report

# Export safety data
meta-safety export --filename safety_report.json

# Update safety rules
meta-safety update-rules --file rules.json
```

### Streaming CLI

```bash
# Start streaming session
meta-streaming start --user-id user123

# Stream response
meta-streaming stream "Hello world" --session-id session123

# Get session status
meta-streaming status --session-id session123

# Close session
meta-streaming close --session-id session123
```

## Web API

### FastAPI Endpoints

The web API provides REST endpoints and WebSocket support.

#### REST Endpoints

```python
# Health check
GET /health

# Chat endpoint
POST /chat
{
    "message": "What is the weather like?",
    "stream": false,
    "context": {"user_id": "user123"}
}

# Streaming chat
POST /chat/stream
{
    "message": "Tell me a story",
    "stream": true
}

# Create streaming session
POST /stream/session
{
    "user_id": "user123",
    "context": {"preferences": "technical"}
}

# Get stream status
GET /stream/{stream_id}

# Close stream
DELETE /stream/{stream_id}

# Get all streams
GET /streams

# Get system status
GET /status
```

#### WebSocket Endpoints

```python
# WebSocket connection
WS /ws

# Send message
{
    "type": "chat",
    "message": "Hello",
    "stream": true
}

# Get status
{
    "type": "status"
}
```

### Response Formats

#### Chat Response

```json
{
    "response": "The weather is sunny today.",
    "category": "knowledge_retrieval",
    "stream_id": "stream_123",
    "timestamp": "2024-01-15T10:30:00Z",
    "safety_validated": true
}
```

#### Status Response

```json
{
    "status": "healthy",
    "active_connections": 5,
    "streaming_stats": {
        "total_sessions": 10,
        "active_sessions": 3
    },
    "safety_status": {
        "enabled": true,
        "violations": 0
    }
}
```

## Safety API

### Safety Enforcement

```python
from core.safety_enforcement import get_safety_enforcement

safety = get_safety_enforcement()
result = safety.validate_action("Generate helpful content", {})
```

#### Methods

- `validate_action(action: str, context: Dict) -> Dict[str, Any]`: Validate an action
- `check_content(content: str) -> Dict[str, Any]`: Check content safety
- `get_safety_rules() -> List[str]`: Get current safety rules

### Safety Guardrails

```python
from core.safety_guardrails import SafetyGuardrails

guardrails = SafetyGuardrails()
result = guardrails.validate_change(change_type="rule_modification", content="...")
```

## Memory API

### Memory Management

```python
from core.memory_eviction import MemoryEvictionManager
from core.context_window_manager import ContextWindowManager

# Memory eviction
eviction_manager = MemoryEvictionManager(memory_agent)
eviction_manager.cleanup_old_memories()

# Context window management
context_manager = ContextWindowManager()
context_manager.add_to_context("User said: Hello")
context = context_manager.get_current_context()
```

## Metrics API

### System Monitor

```python
from core.metrics import SystemMonitor

monitor = SystemMonitor()
memory_usage = monitor.get_memory_usage()
cpu_usage = monitor.get_cpu_usage()
gpu_usage = monitor.get_gpu_usage()
```

### Performance Tracking

```python
from core.metrics import PerformanceTracker

tracker = PerformanceTracker()
tracker.start_operation("test_op")
# ... perform operation ...
tracker.end_operation("test_op", success=True)
```

## Streaming API

### WebSocket Manager

```python
from core.websocket_manager import get_websocket_manager

ws_manager = get_websocket_manager()
await ws_manager.connect(websocket)
await ws_manager.send_message(user_id, message)
```

### Streaming Manager

```python
from core.streaming_manager import get_streaming_manager

streaming_manager = get_streaming_manager()
stream_id = streaming_manager.create_stream("test_stream")
streaming_manager.add_chunk(stream_id, "Hello")
```

## Error Handling

### Common Exceptions

```python
class MetaModelError(Exception):
    """Base exception for Quark AI Assistant."""
    pass

class SafetyViolationError(MetaModelError):
    """Raised when safety rules are violated."""
    pass

class MemoryError(MetaModelError):
    """Raised when memory operations fail."""
    pass

class StreamingError(MetaModelError):
    """Raised when streaming operations fail."""
    pass
```

### Error Response Format

```json
{
    "error": "Safety violation detected",
    "error_type": "SafetyViolationError",
    "details": "Content contains harmful language",
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123"
}
```

## Configuration

### Environment Variables

```bash
# Core settings
META_MODEL_ENV=production
META_MODEL_SAFETY_ENABLED=true
META_MODEL_MEMORY_PATH=/app/memory_db
META_MODEL_LOG_LEVEL=INFO

# Cloud integration
META_MODEL_CLOUD_ENABLED=true
META_MODEL_WEB_BROWSER_ENABLED=true

# Database connections
REDIS_URL=redis://redis:6379
CHROMADB_HOST=chromadb
CHROMADB_PORT=8000
```

### Configuration Files

```python
# config/settings.py
import os

class Settings:
    ENV = os.getenv("META_MODEL_ENV", "development")
    SAFETY_ENABLED = os.getenv("META_MODEL_SAFETY_ENABLED", "true").lower() == "true"
    MEMORY_PATH = os.getenv("META_MODEL_MEMORY_PATH", "./memory_db")
    LOG_LEVEL = os.getenv("META_MODEL_LOG_LEVEL", "INFO")
```

## Examples

### Basic Usage

```python
from core.orchestrator import Orchestrator

# Initialize the system
orchestrator = Orchestrator()

# Ask a question
response = orchestrator.handle("What is the capital of France?")
print(response["final_response"])

# Get performance stats
stats = orchestrator.get_performance_stats()
print(f"Total requests: {stats['total_requests']}")
```

### Streaming Example

```python
import asyncio
from agents.streaming_agent import StreamingAgent

async def main():
    streaming_agent = StreamingAgent()
    session_id = streaming_agent.create_session("user123")
    
    async def response_generator(prompt: str):
        return "This is a streaming response with multiple tokens."
    
    async for event in streaming_agent.stream_response("Hello", response_generator, session_id):
        print(f"Token: {event.content}")
    
    await streaming_agent.close_session(session_id)

asyncio.run(main())
```

### Memory Example

```python
from agents.memory_agent import MemoryAgent

memory_agent = MemoryAgent()

# Store a memory
memory_id = memory_agent.store_memory(
    content="User prefers technical explanations",
    memory_type="preference",
    metadata={"importance": 0.9}
)

# Retrieve relevant memories
memories = memory_agent.retrieve_memories("technical", n_results=3)
for memory in memories:
    print(f"Memory: {memory['content']} (similarity: {memory['similarity']})")
```

### Safety Example

```python
from agents.safety_agent import SafetyAgent

safety_agent = SafetyAgent()
result = safety_agent.generate("Generate harmful content", operation="assess_safety")

if result["is_safe"]:
    print("Content is safe")
else:
    print(f"Content blocked: {result['reason']}")
```

## Best Practices

### Performance

1. **Use streaming for long responses**: Enable streaming for responses that take time to generate
2. **Batch memory operations**: Store multiple memories at once when possible
3. **Monitor metrics**: Regularly check performance metrics to identify bottlenecks
4. **Use appropriate safety levels**: Balance safety with performance based on your use case

### Safety

1. **Always enable safety**: Never disable safety features in production
2. **Monitor safety violations**: Track and analyze safety violations
3. **Regular safety audits**: Periodically review and update safety rules
4. **User feedback**: Collect and incorporate user feedback on safety decisions

### Memory

1. **Use meaningful memory types**: Categorize memories appropriately
2. **Set appropriate importance scores**: Help the system prioritize memories
3. **Regular cleanup**: Configure automatic memory cleanup
4. **Monitor memory usage**: Track memory statistics and performance

### Error Handling

1. **Always handle exceptions**: Wrap API calls in try-catch blocks
2. **Log errors appropriately**: Use structured logging for debugging
3. **Provide user-friendly messages**: Translate technical errors to user-friendly messages
4. **Implement retry logic**: Add retry mechanisms for transient failures

## Support

For additional support:

- **Documentation**: [User Guide](USER_GUIDE.md)
- **Issues**: [GitHub Issues](https://github.com/camdouglas/quark-local-agi/issues)
- **Discussions**: [GitHub Discussions](https://github.com/camdouglas/quark-local-agi/discussions)
- **Examples**: [examples/](examples/) directory 