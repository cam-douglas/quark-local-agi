#!/usr/bin/env bash
# Meta-Model AI Assistant Shell Script
# This script provides the main entry point for the meta_model command

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Meta-Model project directory
META_MODEL_DIR="/Users/camdouglas/meta_model"

# Function to print status messages
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

print_header() {
    echo -e "${MAGENTA}$1${NC}"
}

# Function to check if we're in the project root
check_project_root() {
    if [[ ! -d "$META_MODEL_DIR" ]]; then
        print_error "Meta-Model project not found at $META_MODEL_DIR"
        return 1
    fi
    return 0
}

# Function to ensure environment is ready
ensure_environment() {
    print_info "Checking environment..."
    
    # Check Python version
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    if [[ "$python_version" < "3.10" ]]; then
        print_error "Python 3.10+ required, found $python_version"
        print_info "Please install Python 3.10+ and try again"
        return 1
    fi
    
    # Check virtual environment
    if [[ ! -d "$META_MODEL_DIR/venv" ]]; then
        print_warning "Virtual environment not found"
        print_info "Running environment setup..."
        cd "$META_MODEL_DIR"
        ./scripts/setup_env.sh
        return $?
    fi
    
    # Check models
    if [[ ! -d "$META_MODEL_DIR/models" ]] || [[ -z "$(ls -A "$META_MODEL_DIR/models" 2>/dev/null)" ]]; then
        print_warning "Models not found"
        print_info "Downloading models..."
        cd "$META_MODEL_DIR"
        python3 scripts/download_models.py
        return $?
    fi
    
    print_status "Environment is ready"
    return 0
}

# Function to start the AI assistant
start_ai_assistant() {
    print_header "🚀 Starting Meta-Model AI Assistant"
    print_info "Type 'exit' or 'quit' to leave"
    print_info "Type 'help' for available commands"
    echo ""
    
    cd "$META_MODEL_DIR"
    python3 main.py
}

# Function to show help
show_help() {
    print_header "🤖 Meta-Model AI Assistant Help"
    echo ""
    print_info "Available commands:"
    echo "  meta_model                    # Start interactive assistant"
    echo "  meta_model 'What is AI?'      # Ask a question directly"
    echo "  meta_model help               # Show this help"
    echo "  meta_model status             # Check environment status"
    echo "  meta_model models             # List available models"
    echo "  meta_model setup              # Run environment setup"
    echo "  meta_model download           # Download models"
    echo "  meta_model test               # Run a quick test"
    echo ""
    print_info "Examples:"
    echo "  meta_model 'What is machine learning?'"
    echo "  meta_model 'Translate to German: Hello world'"
    echo "  meta_model 'Analyze sentiment: I love this product'"
    echo ""
}

# Function to check status
check_status() {
    print_header "🔍 Meta-Model AI Assistant Status"
    echo ""
    
    if ! check_project_root; then
        return 1
    fi
    
    cd "$META_MODEL_DIR"
    python3 scripts/check_env.py
}

# Function to list models
list_models() {
    print_header "🤖 Available Models"
    echo ""
    
    if ! check_project_root; then
        return 1
    fi
    
    cd "$META_MODEL_DIR"
    echo "Intent Classification: facebook/bart-large-mnli"
    echo "Named Entity Recognition: elastic/distilbert-base-cased-finetuned-conll03-english"
    echo "Sentiment Analysis: distilbert-base-uncased-finetuned-sst-2-english"
    echo "Semantic Embeddings: sentence-transformers/all-distilroberta-v1"
    echo "Text Generation: google/flan-t5-small"
    echo "Summarization: sshleifer/distilbart-cnn-12-6"
    echo "Translation: Helsinki-NLP/opus-mt-en-de"
    echo ""
}

# Function to run setup
run_setup() {
    print_header "🔧 Meta-Model AI Assistant Setup"
    echo ""
    
    if ! check_project_root; then
        return 1
    fi
    
    cd "$META_MODEL_DIR"
    ./scripts/setup_env.sh
}

# Function to download models
download_models() {
    print_header "📥 Downloading Models"
    echo ""
    
    if ! check_project_root; then
        return 1
    fi
    
    cd "$META_MODEL_DIR"
    python3 scripts/download_models.py
}

# Function to run test
run_test() {
    print_header "🧪 Running Quick Test"
    echo ""
    
    if ! check_project_root; then
        return 1
    fi
    
    cd "$META_MODEL_DIR"
    echo "Testing basic functionality..."
    python3 -c "
import sys
sys.path.insert(0, '.')
from main import cli
print('✅ Basic import test passed')
"
}

# Main function
main() {
    # Check project root
    if ! check_project_root; then
        exit 1
    fi
    
    # Parse command line arguments
    case "${1:-}" in
        "help"|"--help"|"-h")
            show_help
            ;;
        "status"|"--status")
            check_status
            ;;
        "models"|"--models")
            list_models
            ;;
        "setup"|"--setup")
            run_setup
            ;;
        "download"|"--download")
            download_models
            ;;
        "test"|"--test")
            run_test
            ;;
        "")
            # No arguments - start interactive mode
            if ensure_environment; then
                start_ai_assistant
            else
                print_error "Environment setup failed"
                exit 1
            fi
            ;;
        *)
            # Direct question mode
            if ensure_environment; then
                cd "$META_MODEL_DIR"
                echo "$1" | python3 main.py
            else
                print_error "Environment setup failed"
                exit 1
            fi
            ;;
    esac
}

# Run main function
main "$@" 