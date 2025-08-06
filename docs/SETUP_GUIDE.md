# Meta-Model AI Assistant - Complete Setup Guide

## 🚀 Quick Start (Recommended)

### 1. Install Shell Integration

```bash
# From the meta_model project directory
./scripts/install_shell.sh
```

This will:
- ✅ Detect your shell (bash/zsh)
- ✅ Install the meta_model command globally
- ✅ Set up automatic environment activation
- ✅ Create helpful aliases

### 2. Restart Your Terminal

```bash
# Or reload your shell configuration
source ~/.bashrc  # for bash
source ~/.zshrc   # for zsh
```

### 3. Start Using Meta-Model

```bash
# Start the AI assistant
meta_model

# Or ask a question directly
meta_model "What is artificial intelligence?"

# Check status
meta_model status

# Get help
meta_model help
```

## 🔧 Manual Setup (Alternative)

If you prefer manual setup:

### 1. Add to Your Shell Profile

Add this line to your `~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`:

```bash
source "/Users/camdouglas/meta_model/scripts/meta_profile.sh"
```

### 2. Reload Your Shell

```bash
source ~/.bashrc  # or ~/.zshrc
```

## 📋 Available Commands

Once installed, you'll have access to these commands:

### Core Commands
- `meta_model` - Start the AI assistant
- `meta_model help` - Show help
- `meta_model status` - Check environment status
- `meta_model models` - List available models
- `meta_model setup` - Run environment setup
- `meta_model download` - Download models
- `meta_model test` - Run a quick test

### Utility Commands
- `meta_model_check` - Check environment
- `meta_model_status` - Show status
- `meta_model_setup` - Run setup
- `meta_model_download` - Download models
- `meta_model_help` - Show help

### Aliases (Shortcuts)
- `meta` - Alias for `meta_model`
- `meta-help` - Alias for `meta_model_help`
- `meta-status` - Alias for `meta_model_status`
- `meta-check` - Alias for `meta_model_check`
- `meta-setup` - Alias for `meta_model_setup`
- `meta-download` - Alias for `meta_model_download`

## 🤖 Using the AI Assistant

### Interactive Mode

```bash
meta_model
```

This starts an interactive session where you can:
- Ask questions
- Request text processing
- Get translations
- Analyze sentiment
- Extract entities
- Generate text

### Direct Commands

```bash
# Ask a question
meta_model "What is machine learning?"

# Request translation
meta_model "Translate to German: Hello world"

# Analyze sentiment
meta_model "Analyze sentiment: I love this product"

# Extract entities
meta_model "Extract entities: Apple Inc. is headquartered in Cupertino"

# Generate text
meta_model "Generate text about: artificial intelligence"
```

## 🔍 Environment Management

### Check Environment Status

```bash
meta_model status
```

This will show:
- ✅ Python version compatibility
- ✅ Virtual environment status
- ✅ Model availability
- ✅ Project structure

### Setup Environment

```bash
meta_model setup
```

This will:
- ✅ Install Python dependencies
- ✅ Create virtual environment
- ✅ Download SpaCy model
- ✅ Preload all models

### Download Models

```bash
meta_model download
```

This downloads all required models:
- Intent classification models
- Named entity recognition models
- Sentiment analysis models
- Text generation models
- Translation models
- Embedding models

## 🛠️ Troubleshooting

### Python Version Issues

If you get "Python version too old" errors:

```bash
# Install Python 3.11
brew install python@3.11

# Update PATH
export PATH="/usr/local/bin:$PATH"

# Verify version
python3 --version
```

### Missing Dependencies

If dependencies are missing:

```bash
# Reinstall in fresh environment
rm -rf venv
meta_model setup
```

### Model Download Issues

If models fail to download:

```bash
# Manual model download
meta_model download

# Or download specific models
python3 -c "from transformers import pipeline; pipeline('text-classification', model='typeform/distilbert-base-uncased-mnli')"
```

### Shell Integration Issues

If the `meta_model` command is not found:

```bash
# Reinstall shell integration
./scripts/install_shell.sh

# Or manually source the profile
source /Users/camdouglas/meta_model/scripts/meta_profile.sh
```

### Memory Issues

If you run out of memory:

```bash
# Use smaller models
export TRANSFORMERS_CACHE="/tmp/transformers"
export HF_HOME="/tmp/huggingface"

# Or increase swap space
sudo sysctl vm.swappiness=10
```

## 📁 Project Structure

```
meta_model/
├── scripts/
│   ├── meta_shell.sh          # Main shell launcher
│   ├── meta_profile.sh        # Shell profile functions
│   ├── install_shell.sh       # Installation script
│   ├── setup_env.sh           # Environment setup
│   ├── download_models.py     # Model downloader
│   ├── check_env.py           # Environment checker
│   └── preload_models.py      # Model preloader
├── cli/
│   └── cli.py                 # Enhanced CLI interface
├── core/
│   └── orchestrator.py        # Main orchestrator
└── agents/                    # AI agent implementations
```

## 🎯 Examples

### Text Processing

```bash
# Summarize text
meta_model "Summarize this text: Artificial intelligence is a field of computer science that aims to create intelligent machines."

# Analyze sentiment
meta_model "Analyze sentiment: I absolutely love this new AI assistant!"

# Extract entities
meta_model "Extract entities: Tim Cook is the CEO of Apple Inc. in Cupertino, California."
```

### Translation

```bash
# Translate to German
meta_model "Translate to German: Hello, how are you today?"

# Translate to French
meta_model "Translate to French: The weather is beautiful today."
```

### Generation

```bash
# Generate text
meta_model "Generate text about: the future of artificial intelligence"

# Creative writing
meta_model "Write a short story about: a robot learning to paint"
```

### Questions

```bash
# General questions
meta_model "What is the difference between machine learning and deep learning?"

# Technical questions
meta_model "How does a neural network work?"

# Practical questions
meta_model "What are the best practices for data preprocessing?"
```

## 🔄 Updates and Maintenance

### Update Models

```bash
meta_model download
```

### Update Environment

```bash
meta_model setup
```

### Check for Issues

```bash
meta_model status
```

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Run `meta_model status` to diagnose issues
3. Check the logs in the `logs/` directory
4. Ensure Python 3.10+ is installed
5. Verify all models are downloaded

## 🎉 Success!

Once everything is set up, you'll have a powerful AI assistant that can:

- ✅ Understand and classify user intent
- ✅ Process and analyze text
- ✅ Generate creative content
- ✅ Translate between languages
- ✅ Extract named entities
- ✅ Perform sentiment analysis
- ✅ Answer questions intelligently
- ✅ Plan and reason about tasks

The assistant will automatically use the most appropriate models for each task, providing intelligent responses to your queries! 