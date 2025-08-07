#!/usr/bin/env python3
"""
Quick Launch Script for Comprehensive Training
==============================================

Simple launcher for the overnight training session.
"""

import os
import sys
import subprocess
from datetime import datetime

def main():
    print("🚀 Quark AI - Training Session Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("agents") or not os.path.exists("core"):
        print("❌ Please run this script from the quark root directory")
        print("   Current directory:", os.getcwd())
        return
    
    print("✅ Quark directory detected")
    print(f"📅 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Training options
    print("\n🎯 Training Options:")
    print("1. Full overnight training (12 hours, all pillars)")
    print("2. Quick training test (1 hour, core pillars only)")
    print("3. Custom training session")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        print("\n🌙 Starting full overnight training session...")
        cmd = [sys.executable, "scripts/overnight_comprehensive_training.py"]
        
    elif choice == "2":
        print("\n⚡ Starting quick training test...")
        # Modify the script for quick test
        cmd = [sys.executable, "-c", """
import asyncio
import sys
import os
sys.path.insert(0, '.')

from scripts.overnight_comprehensive_training import ComprehensiveTrainingSession

async def quick_test():
    trainer = ComprehensiveTrainingSession('quick_test')
    trainer.config['max_training_hours'] = 1
    trainer.config['pillar_training_cycles'] = 1
    
    # Test with just a few core pillars
    core_pillars = ['Natural Language Understanding', 'Knowledge Retrieval', 'Programming & Code Generation']
    
    await trainer.initialize_agents()
    trainer.generate_training_datasets()
    
    for pillar in core_pillars:
        print(f'Testing {pillar}...')
        results = await trainer.train_pillar(pillar)
        print(f'Result: {results["status"]}')
    
    print('Quick test completed!')

asyncio.run(quick_test())
"""]
        
    elif choice == "3":
        print("\n⚙️  Custom training - edit the script manually")
        print("📝 File location: scripts/overnight_comprehensive_training.py")
        return
        
    else:
        print("❌ Invalid choice")
        return
    
    print(f"\n🔄 Executing command: {' '.join(cmd)}")
    print("📝 Training logs will be saved to training_sessions/")
    print("🛑 Press Ctrl+C to stop training gracefully")
    print("-" * 50)
    
    try:
        # Execute the training
        result = subprocess.run(cmd, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("\n🎉 Training completed successfully!")
        else:
            print(f"\n⚠️  Training exited with code: {result.returncode}")
            
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n💥 Error running training: {e}")

if __name__ == "__main__":
    main()