#!/usr/bin/env python3
"""
Startup script that ensures the environment is ready before running the AI assistant.
"""
import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'transformers',
        'torch', 
        'sentence_transformers',
        'spacy',
        'chromadb',
        'click',
        'requests',
        'prometheus_client'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def check_spacy_model():
    """Check if SpaCy English model is installed."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        return True
    except OSError:
        return False

def check_models_directory():
    """Check if models directory exists and has content."""
    models_dir = Path("models")
    return models_dir.exists() and any(models_dir.iterdir())

def setup_environment():
    """Set up the environment if needed."""
    print("🔍 Checking environment...")
    
    # Check dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("🔧 Running environment setup...")
        subprocess.run(["./scripts/setup_env.sh"], check=True)
        return
    
    # Check SpaCy model
    if not check_spacy_model():
        print("📚 Installing SpaCy English model...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    
    # Check models directory
    if not check_models_directory():
        print("📥 Downloading models...")
        subprocess.run([sys.executable, "scripts/download_models.py"], check=True)
    
    print("✅ Environment is ready!")

def main():
    """Main startup function."""
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🚀 Meta-Model AI Assistant Startup")
    print("=" * 40)
    
    setup_environment()
    
    # Start the AI assistant
    print("\n🎯 Starting AI Assistant...")
    
    # Run the main entry point
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to start CLI. Please run setup: ./scripts/setup_env.sh")
    except FileNotFoundError:
        print("❌ Main entry point not found. Please check the project structure.")

if __name__ == "__main__":
    main()

