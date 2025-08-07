#!/usr/bin/env python3
"""
Quick installer for missing dependencies
========================================
Installs missing packages needed for the training system.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data."""
    try:
        import nltk
        import ssl
        
        # Handle SSL certificate issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download required NLTK data
        data_packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for package in data_packages:
            try:
                nltk.download(package, quiet=True)
                print(f"✅ Downloaded NLTK {package}")
            except Exception as e:
                print(f"⚠️  Failed to download NLTK {package}: {e}")
        
        return True
    except ImportError:
        print("❌ NLTK not installed")
        return False

def install_spacy_model():
    """Install spaCy English model."""
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ Successfully installed spaCy English model")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install spaCy model: {e}")
        return False

def main():
    print("🔧 Installing Missing Dependencies for Quark Training")
    print("=" * 60)
    
    # Critical packages that are commonly missing
    critical_packages = [
        "nltk>=3.8.1",
        "networkx>=3.1", 
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "psutil>=5.9.0",
        "spacy>=3.6.0",
        "faiss-cpu>=1.7.4",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0"
    ]
    
    print("📦 Installing critical packages...")
    for package in critical_packages:
        print(f"  Installing {package}...")
        install_package(package)
    
    print("\n📚 Setting up NLTK data...")
    download_nltk_data()
    
    print("\n🌐 Installing spaCy language model...")
    install_spacy_model()
    
    print("\n✅ Dependency installation complete!")
    print("🚀 You can now run the training system.")

if __name__ == "__main__":
    main()