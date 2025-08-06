#!/usr/bin/env python3
"""
Environment check script for Meta-Model AI Assistant.
Verifies all dependencies, models, and configurations are ready.
"""
import os
import sys
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        print(f"❌ Python {sys.version} is too old. Need Python 3.10+")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = {
        'transformers': 'Hugging Face Transformers',
        'torch': 'PyTorch',
        'sentence_transformers': 'Sentence Transformers',
        'spacy': 'SpaCy',
        'chromadb': 'ChromaDB',
        'click': 'Click CLI framework',
        'requests': 'Requests HTTP library',
        'prometheus_client': 'Prometheus client',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn'
    }
    
    missing_packages = []
    for package, description in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"✅ {description}")
        except ImportError:
            print(f"❌ {description} - Missing")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_spacy_model():
    """Check if SpaCy English model is installed."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ SpaCy English model (en_core_web_sm)")
        return True
    except OSError:
        print("❌ SpaCy English model (en_core_web_sm) - Missing")
        return False

def check_models_directory():
    """Check if models directory exists and has content."""
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory - Missing")
        return False
    
    model_count = len(list(models_dir.iterdir()))
    if model_count == 0:
        print("❌ Models directory - Empty")
        return False
    
    print(f"✅ Models directory ({model_count} items)")
    return True

def check_virtual_environment():
    """Check if we're in a virtual environment."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment active")
        return True
    else:
        print("⚠️  Not in virtual environment (this might be okay)")
        return True

def check_project_structure():
    """Check if project structure is correct."""
    required_dirs = ['agents', 'core', 'cli', 'scripts', 'config', 'docs', 'tests']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/ directory")
        else:
            print(f"❌ {dir_name}/ directory - Missing")
            missing_dirs.append(dir_name)
    
    return len(missing_dirs) == 0

def check_config_files():
    """Check if essential config files exist."""
    required_files = [
        'config/requirements.txt',
        'config/pyproject.toml',
        'config/setup.py',
        'scripts/setup_env.sh',
        'scripts/ai_assistant.sh'
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def main():
    """Main environment check function."""
    print("🔍 Meta-Model AI Assistant - Environment Check")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Check Python version
    print("\n🐍 Python Environment:")
    if not check_python_version():
        all_checks_passed = False
    
    # Check virtual environment
    print("\n🔧 Virtual Environment:")
    check_virtual_environment()
    
    # Check dependencies
    print("\n📦 Dependencies:")
    deps_ok, missing_deps = check_dependencies()
    if not deps_ok:
        all_checks_passed = False
    
    # Check SpaCy model
    print("\n📚 SpaCy Model:")
    if not check_spacy_model():
        all_checks_passed = False
    
    # Check project structure
    print("\n📁 Project Structure:")
    if not check_project_structure():
        all_checks_passed = False
    
    # Check config files
    print("\n⚙️  Configuration Files:")
    if not check_config_files():
        all_checks_passed = False
    
    # Check models directory
    print("\n🤖 Models:")
    if not check_models_directory():
        all_checks_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("✅ All checks passed! Environment is ready.")
        print("🎯 You can run the AI assistant with: python scripts/ai_assistant.sh")
    else:
        print("❌ Some checks failed. Please run setup:")
        print("   ./scripts/setup_env.sh")
        if missing_deps:
            print(f"   Missing packages: {', '.join(missing_deps)}")
    
    return all_checks_passed

if __name__ == "__main__":
    main() 