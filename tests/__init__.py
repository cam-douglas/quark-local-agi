"""
TESTING FRAMEWORK
================

Comprehensive testing suite for the Meta-Model AI Assistant.
Includes unit tests, integration tests, end-to-end tests, and performance benchmarks.
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_CONFIG = {
    "unit_tests": True,
    "integration_tests": True,
    "end_to_end_tests": True,
    "performance_tests": True,
    "security_tests": True,
    "adversarial_tests": True,
    "coverage_threshold": 80.0,
    "timeout_seconds": 30,
    "max_memory_mb": 1024
} 