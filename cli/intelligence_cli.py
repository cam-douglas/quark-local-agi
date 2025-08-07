#!/usr/bin/env python3
"""
Intelligence CLI for Quark AI System
Monitor and control background intelligence improvement
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.background_intelligence_improver import BackgroundIntelligenceImprover

class IntelligenceCLI:
    """CLI for intelligence improvement monitoring and control"""
    
    def __init__(self):
        self.quark_dir = Path("/Users/camdouglas/quark")
        self.data_dir = self.quark_dir / "data" / "intelligence_improvement"
        self.improver = BackgroundIntelligenceImprover()
    
    def show_stats(self):
        """Show intelligence improvement statistics"""
        try:
            stats = self.improver.get_improvement_stats()
            
            print("🧠 Quark Intelligence Improvement Stats")
            print("=" * 50)
            print(f"Total Improvements: {stats['total_improvements']}")
            print(f"Optimization Cycles: {stats['optimization_cycles']}")
            print(f"Learning Patterns: {stats['learning_patterns']}")
            print(f"Efficiency Gains: {stats['efficiency_gains']}")
            print(f"Current Memory: {stats['current_memory_mb']:.1f}MB")
            print(f"Current CPU: {stats['current_cpu_percent']:.1f}%")
            
            if stats['last_optimization']:
                print(f"Last Optimization: {stats['last_optimization']}")
            else:
                print("Last Optimization: Never")
            
            print()
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
    
    def show_learning_patterns(self):
        """Show recent learning patterns"""
        try:
            data_file = self.data_dir / "improvement_data.json"
            if data_file.exists():
                with open(data_file, 'r') as f:
                    data = json.load(f)
                
                patterns = data.get('learning_patterns', [])
                
                print("📚 Recent Learning Patterns")
                print("=" * 40)
                
                for i, pattern in enumerate(patterns[-10:], 1):  # Last 10 patterns
                    print(f"{i}. Type: {pattern.get('type', 'unknown')}")
                    if pattern.get('type') == 'response_time':
                        print(f"   Average: {pattern.get('average', 0):.3f}s")
                        print(f"   Count: {pattern.get('count', 0)}")
                    elif pattern.get('type') == 'errors':
                        print(f"   Error Count: {pattern.get('count', 0)}")
                        print(f"   Recent Errors: {len(pattern.get('patterns', []))}")
                    print()
            else:
                print("📚 No learning patterns available yet")
                
        except Exception as e:
            print(f"❌ Error showing learning patterns: {e}")
    
    def show_efficiency_gains(self):
        """Show recent efficiency gains"""
        try:
            data_file = self.data_dir / "improvement_data.json"
            if data_file.exists():
                with open(data_file, 'r') as f:
                    data = json.load(f)
                
                gains = data.get('efficiency_gains', [])
                
                print("⚡ Recent Efficiency Gains")
                print("=" * 40)
                
                for i, gain in enumerate(gains[-10:], 1):  # Last 10 gains
                    print(f"{i}. Timestamp: {gain.get('timestamp', 'unknown')}")
                    print(f"   Memory: {gain.get('memory_usage', 0):.1f}MB")
                    print(f"   CPU: {gain.get('cpu_usage', 0):.1f}%")
                    optimizations = gain.get('optimizations_applied', [])
                    if optimizations:
                        print(f"   Optimizations: {', '.join(optimizations)}")
                    print()
            else:
                print("⚡ No efficiency gains available yet")
                
        except Exception as e:
            print(f"❌ Error showing efficiency gains: {e}")
    
    def show_agent_optimizations(self):
        """Show current agent optimizations"""
        try:
            optimization_file = self.data_dir / "agent_optimizations.json"
            if optimization_file.exists():
                with open(optimization_file, 'r') as f:
                    optimizations = json.load(f)
                
                print("🤖 Current Agent Optimizations")
                print("=" * 40)
                
                for agent, config in optimizations.items():
                    print(f"Agent: {agent}")
                    for key, value in config.items():
                        print(f"  {key}: {value}")
                    print()
            else:
                print("🤖 No agent optimizations available yet")
                
        except Exception as e:
            print(f"❌ Error showing agent optimizations: {e}")
    
    def start_improver(self):
        """Start the background intelligence improver"""
        try:
            print("🚀 Starting background intelligence improver...")
            thread = self.improver.start()
            print("✅ Background intelligence improver started")
            print("💡 Use 'quark intelligence stats' to monitor progress")
            
        except Exception as e:
            print(f"❌ Error starting improver: {e}")
    
    def stop_improver(self):
        """Stop the background intelligence improver"""
        try:
            print("🛑 Stopping background intelligence improver...")
            self.improver.stop()
            print("✅ Background intelligence improver stopped")
            
        except Exception as e:
            print(f"❌ Error stopping improver: {e}")
    
    def run_optimization(self):
        """Run a single optimization cycle"""
        try:
            print("🔄 Running optimization cycle...")
            
            # Run optimizations
            self.improver.optimize_memory_usage()
            self.improver.optimize_intelligence_algorithms()
            self.improver.improve_efficiency()
            self.improver.adaptive_learning()
            
            print("✅ Optimization cycle completed")
            
        except Exception as e:
            print(f"❌ Error running optimization: {e}")
    
    def show_help(self):
        """Show help information"""
        print("🧠 Quark Intelligence CLI")
        print("=" * 30)
        print("Commands:")
        print("  stats                    - Show intelligence improvement stats")
        print("  patterns                 - Show recent learning patterns")
        print("  gains                    - Show recent efficiency gains")
        print("  agents                   - Show current agent optimizations")
        print("  start                    - Start background intelligence improver")
        print("  stop                     - Stop background intelligence improver")
        print("  optimize                 - Run a single optimization cycle")
        print("  help                     - Show this help")
        print()

def main():
    """Main CLI entry point"""
    cli = IntelligenceCLI()
    
    if len(sys.argv) < 2:
        cli.show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "stats":
        cli.show_stats()
    elif command == "patterns":
        cli.show_learning_patterns()
    elif command == "gains":
        cli.show_efficiency_gains()
    elif command == "agents":
        cli.show_agent_optimizations()
    elif command == "start":
        cli.start_improver()
    elif command == "stop":
        cli.stop_improver()
    elif command == "optimize":
        cli.run_optimization()
    elif command == "help":
        cli.show_help()
    else:
        print(f"❌ Unknown command: {command}")
        cli.show_help()

if __name__ == "__main__":
    main() 