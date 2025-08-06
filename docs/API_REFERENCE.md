# API Reference

## Table of Contents

1. [Core Components](#core-components)
2. [Agents](#agents)
3. [CLI Interfaces](#cli-interfaces)
4. [Web API](#web-api)
5. [Safety System](#safety-system)
6. [Memory System](#memory-system)
7. [Cloud Integration](#cloud-integration)
8. [Web Browser](#web-browser)

## Core Components

### Orchestrator

The main orchestrator that coordinates all AI agents and handles user requests.

```python
from core.orchestrator import Orchestrator

# Initialize orchestrator
orchestrator = Orchestrator()

# Handle user request
result = orchestrator.handle("Hello, how are you?")
```

**Methods:**
- `handle(prompt: str) -> Dict[str, Any]`: Process user input and return response
- `get_status() -> Dict[str, Any]`: Get orchestrator status

### Router

Routes user prompts to appropriate agents based on intent classification.

```python
from core.router import Router

router = Router()
category = router.route("What is the weather today?")
```

**Methods:**
- `route(prompt: str) -> str`: Route prompt to appropriate category

### Context Window Manager

Manages short-term conversational context and token limits.

```python
from core.context_window_manager import ContextWindowManager

context_manager = ContextWindowManager()
context_manager.add_message("user", "Hello")
context = context_manager.get_context()
```

**Methods:**
- `add_message(role: str, content: str) -> Dict[str, Any]`: Add message to context
- `get_context() -> Dict[str, Any]`: Get current context
- `clear_context() -> Dict[str, Any]`: Clear all context

## Agents

### NLU Agent

Natural Language Understanding agent for intent classification and entity recognition.

```python
from agents.nlu_agent import NLUAgent

nlu_agent = NLUAgent()
intent = nlu_agent.classify_intent("What's the weather like?")
```

### Memory Agent

Manages long-term memory storage and retrieval using ChromaDB.

```python
from agents.memory_agent import MemoryAgent

memory_agent = MemoryAgent()
result = memory_agent.store_memory("User likes pizza", {"category": "preferences"})
memories = memory_agent.retrieve_memories("food preferences")
```

**Methods:**
- `store_memory(content: str, metadata: Dict) -> Dict[str, Any]`: Store memory
- `retrieve_memories(query: str) -> Dict[str, Any]`: Retrieve relevant memories
- `get_memory_stats() -> Dict[str, Any]`: Get memory statistics
- `delete_memory(memory_id: str) -> Dict[str, Any]`: Delete specific memory

### Metrics Agent

Collects and provides performance metrics and analytics.

```python
from agents.metrics_agent import MetricsAgent

metrics_agent = MetricsAgent()
metrics_agent.record_operation("text_generation", 0.5, 100)
summary = metrics_agent.get_performance_summary()
```

**Methods:**
- `record_operation(operation_type: str, duration: float, tokens: int) -> None`: Record operation
- `get_performance_summary() -> Dict[str, Any]`: Get performance summary
- `get_error_summary() -> Dict[str, Any]`: Get error summary

### Self-Improvement Agent

Manages automated learning and capability bootstrapping.

```python
from agents.self_improvement_agent import SelfImprovementAgent

improvement_agent = SelfImprovementAgent()
result = improvement_agent.learn_from_feedback("positive", "Great response!")
```

**Methods:**
- `learn_from_feedback(feedback_type: str, feedback_text: str) -> Dict[str, Any]`: Learn from feedback
- `bootstrap_capability(capability: str) -> Dict[str, Any]`: Learn new capability
- `get_learning_stats() -> Dict[str, Any]`: Get learning statistics

## CLI Interfaces

### Main CLI

```bash
# Start the AI assistant
meta-model

# Available commands
meta-model --help
```

### Memory CLI

```bash
# Manage memories
meta-memory

# Store memory
meta-memory store "User likes pizza"

# Search memories
meta-memory search "pizza"

# Get statistics
meta-memory stats
```

### Metrics CLI

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

### Safety CLI

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

### Streaming CLI

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

### Cloud CLI

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

### Web Browser CLI

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

### FastAPI Application

The web API provides REST endpoints and WebSocket support.

```python
from web.fastapi_app import app
import uvicorn

# Run the server
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### REST Endpoints

**Health Check:**
```bash
GET /health
```

**Chat Endpoint:**
```bash
POST /chat
{
    "message": "Hello, how are you?",
    "stream": false,
    "context": {}
}
```

**Streaming Chat:**
```bash
POST /chat/stream
{
    "message": "Generate a long response",
    "stream": true
}
```

**Status:**
```bash
GET /status
```

### WebSocket Endpoints

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

## Safety System

### Immutable Safety Rules

```python
from core.immutable_safety_rules import ImmutableSafetyRules

safety_rules = ImmutableSafetyRules()
result = safety_rules.check_action_safety("Generate helpful content", {})
```

**Methods:**
- `check_action_safety(action: str, context: Dict) -> Dict[str, Any]`: Check if action is safe
- `validate_truthfulness(response: str, capabilities: List[str]) -> bool`: Validate response truthfulness
- `verify_integrity() -> bool`: Verify safety rules integrity

### Safety Enforcement

```python
from core.safety_enforcement import SafetyEnforcement

safety_enforcement = SafetyEnforcement()
result = safety_enforcement.validate_action("Generate content", {})
```

**Methods:**
- `validate_action(action: str, context: Dict) -> Dict[str, Any]`: Validate action
- `validate_response(response: str, capabilities: List[str]) -> Dict[str, Any]`: Validate response
- `require_confirmation(action: str, explanation: str, user_input: str) -> bool`: Check if confirmation needed

## Memory System

### Memory Eviction Manager

```python
from core.memory_eviction import MemoryEvictionManager

eviction_manager = MemoryEvictionManager(memory_agent)
result = eviction_manager.apply_time_based_eviction(memories, days_threshold=30)
```

**Methods:**
- `apply_time_based_eviction(memories: List, days_threshold: int) -> List`: Time-based eviction
- `apply_size_based_eviction(memories: List, max_size: int) -> List`: Size-based eviction
- `apply_relevance_based_eviction(memories: List, threshold: float) -> List`: Relevance-based eviction

## Cloud Integration

### Cloud Integration Manager

```python
from core.cloud_integration import CloudIntegration

cloud_integration = CloudIntegration()
result = await cloud_integration.process_with_cloud("text_generation", {"prompt": "Hello"})
```

**Methods:**
- `setup_google_colab(api_key: str = None, notebook_url: str = None) -> bool`: Setup Google Colab
- `setup_huggingface_spaces(space_name: str, api_token: str = None) -> bool`: Setup HF Spaces
- `process_with_cloud(task: str, data: Dict, cloud_provider: str = "auto") -> Dict[str, Any]`: Process with cloud

## Web Browser

### Web Browser Manager

```python
from core.web_browser import WebBrowser

browser = WebBrowser()
result = browser.browse("https://example.com", "Extract main content")
```

**Methods:**
- `browse(url: str, user_query: str = None) -> Dict[str, Any]`: Browse URL
- `search_web(query: str, max_results: int = 5) -> Dict[str, Any]`: Search web
- `resolve_captcha(captcha_id: str, solution: str) -> Dict[str, Any]`: Resolve captcha

## Error Handling

All API methods return dictionaries with the following structure:

```python
{
    "success": bool,           # Whether operation succeeded
    "error": str,             # Error message if failed
    "data": Any,              # Result data if successful
    "metadata": Dict[str, Any] # Additional metadata
}
```

## Configuration

Configuration is handled through environment variables and config files:

```bash
# Environment variables
export META_MODEL_SAFETY_ENABLED=true
export META_MODEL_MEMORY_PATH=./memory_db
export META_MODEL_CLOUD_ENABLED=true
export META_MODEL_WEB_BROWSER_ENABLED=true
```

## Examples

### Basic Usage

```python
from core.orchestrator import Orchestrator

# Initialize and use
orchestrator = Orchestrator()
result = orchestrator.handle("What's the weather like today?")

print(f"Category: {result['category']}")
print(f"Response: {result['results']['Reasoning']}")
```

### Memory Management

```python
from agents.memory_agent import MemoryAgent

# Store and retrieve memories
memory_agent = MemoryAgent()
memory_agent.store_memory("User prefers dark mode", {"category": "preferences"})
memories = memory_agent.retrieve_memories("user preferences")
```

### Safety Validation

```python
from core.safety_enforcement import SafetyEnforcement

# Validate actions and responses
safety = SafetyEnforcement()
action_result = safety.validate_action("Generate helpful content", {})
response_result = safety.validate_response("I can help you", ["text_generation"])
```

### Cloud Processing

```python
from core.cloud_integration import CloudIntegration
import asyncio

async def process_with_cloud():
    cloud = CloudIntegration()
    result = await cloud.process_with_cloud("text_generation", {"prompt": "Hello"})
    return result

# Run async function
result = asyncio.run(process_with_cloud())
```

For more detailed examples, see the [examples/](examples/) directory and [tests/](tests/) directory. 