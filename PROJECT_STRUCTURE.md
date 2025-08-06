# Meta-Model AI Assistant - Project Structure

## Directory Organization

### `/agents/` - AI Agent Implementations
- **nlu_agent.py** - Natural Language Understanding agent
- **planning_agent.py** - Task planning and decomposition agent
- **reasoning_agent.py** - Logical reasoning agent
- **interpretation_agent.py** - Model interpretation agent
- **leader_agent.py** - Coordination and leadership agent
- **retrieval_agent.py** - Information retrieval agent
- **memory_agent.py** - Memory management agent
- **metrics_agent.py** - Performance metrics agent
- **action_agent.py** - Action execution agent
- **generation_agent.py** - Content generation agent
- **execution_agent.py** - Task execution agent
- **intent_classifier.py** - Intent classification
- **base.py** - Base agent class

### `/core/` - Core Application Logic
- **orchestrator.py** - Main orchestrator for agent coordination
- **orchestrator_async.py** - Asynchronous orchestrator
- **router.py** - Request routing logic
- **base.py** - Core base classes
- **use_cases_tasks.py** - Task definitions and use cases
- **memory_eviction.py** - Memory management
- **context_window_manager.py** - Context window management
- **metrics.py** - Performance metrics
- **metrics.json** - Metrics data
- **orchestrator.pt** - Orchestrator model file
- **meta_model** - Main entry point

### `/cli/` - Command Line Interfaces
- **cli.py** - Main CLI interface
- **metrics_cli.py** - Metrics command line tool
- **memory_cli.py** - Memory management CLI

### `/training/` - Training and Evaluation
- **train.py** - Model training scripts
- **evaluate.py** - Model evaluation
- **datasets.py** - Dataset handling

### `/scripts/` - Utility Scripts
- **ai_assistant.sh** - AI assistant launcher
- **setup_env.sh** - Environment setup
- **preload_models.py** - Model preloading
- **startup.py** - Startup scripts
- **download_models.py** - Model downloader
- **run_train.sh** - Training runner
- **run_startup.sh** - Startup runner
- **cleanup_models.sh** - Model cleanup
- **run.sh** - Main runner script

### `/config/` - Configuration Files
- **setup.py** - Package setup
- **requirements.txt** - Python dependencies
- **pyproject.toml** - Project configuration
- **ci.yml** - CI/CD configuration
- **Dockerfile** - Docker configuration
- **.gitignore** - Git ignore rules
- **logging_config.py** - Logging configuration

### `/docs/` - Documentation
- **README.md** - Main documentation
- **CHANGELOG.md** - Change log
- **README_update.md** - Updated documentation

### `/tests/` - Test Files
- **test_agent.py** - Agent tests
- **test_memory_agent.py** - Memory agent tests
- **meta_model.py** - Meta model tests

### `/web/` - Web Interface
- **app.py** - Web application
- **web.py** - Web interface
- **metrics_server.py** - Metrics server

### Other Directories
- **/logs/** - Application logs
- **/memory_db/** - Memory database files
- **/models/** - ML model files
- **/venv/** - Python virtual environment

## File Organization Principles

1. **Separation of Concerns**: Each directory has a specific purpose
2. **Logical Grouping**: Related files are grouped together
3. **Easy Navigation**: Clear directory structure for developers
4. **Scalability**: Structure supports adding new components
5. **Maintainability**: Easy to find and modify specific functionality

## Development Workflow

- **Core Development**: Work in `/core/` for main application logic
- **Agent Development**: Add new agents in `/agents/`
- **CLI Tools**: Create new CLI tools in `/cli/`
- **Training**: Use `/training/` for model training
- **Scripts**: Add utility scripts in `/scripts/`
- **Configuration**: Modify config files in `/config/` 