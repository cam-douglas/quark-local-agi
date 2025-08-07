#!/usr/bin/env python3
"""
Test Fast Startup Performance
============================

Measures the startup time of Quark's fast startup system.
"""

import time
import subprocess
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_fast_startup():
    """Test the fast startup performance"""
    print("🧪 Testing Quark Fast Startup Performance...")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Run the fast startup script
        result = subprocess.run([
            'python3', 'scripts/quark_fast_startup.py'
        ], 
        input=b'how are you?\nquit\n',
        capture_output=True,
        timeout=30,
        cwd=str(project_root)
        )
        
        end_time = time.time()
        startup_time = end_time - start_time
        
        print(f"✅ Fast startup completed in {startup_time:.2f} seconds")
        
        # Check if it responded correctly
        output = result.stdout.decode('utf-8')
        if "Hello! I'm doing well" in output:
            print("✅ Response generation working correctly")
        else:
            print("❌ Response generation not working")
            
        print(f"📊 Startup time: {startup_time:.2f}s")
        
        if startup_time < 10:
            print("🚀 EXCELLENT: Startup under 10 seconds!")
        elif startup_time < 20:
            print("✅ GOOD: Startup under 20 seconds")
        else:
            print("⚠️ SLOW: Startup over 20 seconds")
            
    except subprocess.TimeoutExpired:
        print("❌ Timeout: Startup took too long")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_fast_startup() 