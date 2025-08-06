#!/usr/bin/env python3
"""
Main entry point for Meta-Model AI Assistant
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the CLI
from cli.cli import cli

if __name__ == "__main__":
    cli() 