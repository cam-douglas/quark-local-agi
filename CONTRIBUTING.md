# Contributing to Meta-Model AI Assistant

Thank you for your interest in contributing to the Meta-Model AI Assistant! This document provides guidelines and information for contributors.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Testing](#testing)
6. [Code Quality](#code-quality)
7. [Pull Request Process](#pull-request-process)
8. [Release Process](#release-process)

## Code of Conduct

This project and its participants are governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of AI/ML concepts
- Familiarity with Python development

### Quick Start

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/meta-model-ai-assistant.git
   cd meta-model-ai-assistant
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install in development mode
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Development Setup

### Environment Variables

Create a `.env` file for local development:

```bash
# Development settings
META_MODEL_DEBUG=true
META_MODEL_LOG_LEVEL=DEBUG
META_MODEL_SAFETY_ENABLED=true
META_MODEL_MEMORY_PATH=./memory_db
META_MODEL_CLOUD_ENABLED=true
META_MODEL_WEB_BROWSER_ENABLED=true

# Model settings
META_MODEL_DEVICE=cpu  # or cuda for GPU
META_MODEL_MAX_TOKENS=512
META_MODEL_CACHE_ENABLED=true
```

### Project Structure

```
meta-model-ai-assistant/
├── agents/              # AI agent implementations
├── cli/                 # Command-line interfaces
├── core/                # Core application logic
├── web/                 # Web API components
├── tests/               # Test suite
├── docs/                # Documentation
├── scripts/             # Utility scripts
├── examples/            # Example usage
└── config/              # Configuration files
```

## Contributing Guidelines

### Types of Contributions

We welcome contributions in the following areas:

1. **Bug Fixes**: Fix issues and improve stability
2. **Feature Development**: Add new capabilities and agents
3. **Documentation**: Improve docs and add examples
4. **Testing**: Add tests and improve coverage
5. **Performance**: Optimize speed and memory usage
6. **Safety**: Enhance safety mechanisms
7. **UI/UX**: Improve user interfaces

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards
   - Add tests for new functionality
   - Update documentation

3. **Test your changes**
   ```bash
   # Run all tests
   python tests/run_tests.py
   
   # Run specific test categories
   pytest tests/ -m unit
   pytest tests/ -m integration
   pytest tests/ -m performance
   ```

4. **Check code quality**
   ```bash
   # Format code
   black .
   isort .
   
   # Lint code
   flake8 .
   mypy .
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Build/tooling changes

**Examples:**
```bash
git commit -m "feat: add new memory agent"
git commit -m "fix: resolve safety validation issue"
git commit -m "docs: update API reference"
git commit -m "test: add unit tests for orchestrator"
```

## Testing

### Running Tests

```bash
# Run all tests
python tests/run_tests.py

# Run specific test categories
pytest tests/ -m unit
pytest tests/ -m integration
pytest tests/ -m performance
pytest tests/ -m security
pytest tests/ -m adversarial

# Run with coverage
pytest tests/ --cov=agents --cov=core --cov=cli --cov=web
```

### Test Categories

- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **Performance Tests**: Test speed and resource usage
- **Security Tests**: Test safety mechanisms
- **Adversarial Tests**: Test against attack vectors

### Writing Tests

```python
import unittest
from core.orchestrator import Orchestrator

class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.orchestrator = Orchestrator()
    
    def test_basic_functionality(self):
        """Test basic orchestrator functionality."""
        result = self.orchestrator.handle("Hello")
        self.assertIsInstance(result, dict)
        self.assertIn("category", result)
```

## Code Quality

### Code Style

We use:
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

### Pre-commit Hooks

Pre-commit hooks automatically check code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Type Hints

All new code should include type hints:

```python
from typing import Dict, Any, List, Optional

def process_request(
    user_input: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process a user request."""
    # Implementation
    return {"result": "success"}
```

### Documentation

- **Docstrings**: Use Google-style docstrings
- **Comments**: Explain complex logic
- **README**: Keep updated with new features
- **API Docs**: Document all public APIs

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**
   ```bash
   python tests/run_tests.py
   ```

2. **Check code quality**
   ```bash
   black --check .
   isort --check-only .
   flake8 .
   mypy .
   ```

3. **Update documentation**
   - Update relevant docs
   - Add examples if needed
   - Update changelog

4. **Create pull request**
   - Use descriptive title
   - Fill out PR template
   - Link related issues

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Safety enhancement

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Performance tests pass
- [ ] Security tests pass

## Documentation
- [ ] API documentation updated
- [ ] User guide updated
- [ ] Examples added/updated

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. **Update version**
   ```bash
   # Update pyproject.toml version
   # Update CHANGELOG.md
   ```

2. **Create release branch**
   ```bash
   git checkout -b release/v1.0.0
   ```

3. **Update changelog**
   - Add release notes
   - List all changes
   - Credit contributors

4. **Create GitHub release**
   - Tag the release
   - Upload build artifacts
   - Publish to PyPI

### Release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Release notes written
- [ ] GitHub release created
- [ ] PyPI package published

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Requests**: Code reviews and feedback

### Resources

- [API Reference](docs/API_REFERENCE.md)
- [User Guide](docs/USER_GUIDE.md)
- [Project Structure](PROJECT_STRUCTURE.md)
- [Development Roadmap](docs/DEVELOPMENT_ROADMAP.md)

### Mentorship

New contributors are welcome! We provide:
- **Issue labels**: `good first issue`, `help wanted`
- **Documentation**: Comprehensive guides
- **Code reviews**: Detailed feedback
- **Community support**: Active discussions

## Recognition

Contributors will be recognized in:
- **CHANGELOG.md**: For significant contributions
- **README.md**: For major contributors
- **GitHub releases**: For release contributions
- **Documentation**: For documentation improvements

Thank you for contributing to the Meta-Model AI Assistant! 🚀 