#!/usr/bin/env python3
"""
CLI for Autonomous Decision Agent
Provides command-line interface for autonomous decision making
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.autonomous_decision_agent import AutonomousDecisionAgent

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Autonomous Decision Agent CLI")
    parser.add_argument("command", choices=[
        "decide", "history", "metrics", "patterns", "stats", "help"
    ], help="Command to execute")
    parser.add_argument("--prompt", "-p", help="Decision prompt for decide command")
    parser.add_argument("--limit", "-l", type=int, default=5, help="Limit for history/patterns")
    parser.add_argument("--format", "-f", choices=["text", "json"], default="text", help="Output format")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = AutonomousDecisionAgent()
    agent.load_model()
    
    if args.command == "decide":
        if not args.prompt:
            print("❌ Error: Prompt required for decide command")
            print("Usage: python cli/autonomous_decision_cli.py decide --prompt 'Your decision prompt'")
            return
        
        print("🤖 Making autonomous decision...")
        result = agent.generate(args.prompt)
        print(result)
        
    elif args.command == "history":
        history = agent.get_decision_history()
        if args.format == "json":
            print(json.dumps(history, indent=2, default=str))
        else:
            print("📋 Decision History:")
            print("=" * 50)
            for decision in history[-args.limit:]:
                print(f"ID: {decision['id']}")
                print(f"Type: {decision['type']}")
                print(f"Priority: {decision['priority']}")
                print(f"Confidence: {decision['confidence']:.2%}")
                print(f"Timestamp: {decision['timestamp']}")
                print("-" * 30)
    
    elif args.command == "metrics":
        metrics = agent.get_metrics()
        if args.format == "json":
            print(json.dumps(metrics, indent=2))
        else:
            print("📊 Decision Metrics:")
            print("=" * 30)
            print(f"Total Decisions: {metrics['total_decisions']}")
            print(f"Successful Decisions: {metrics['successful_decisions']}")
            print(f"Average Confidence: {metrics['average_confidence']:.2%}")
            print(f"Decision Accuracy: {metrics['decision_accuracy']:.2%}")
            print(f"Learning Rate: {metrics['learning_rate']:.2%}")
    
    elif args.command == "patterns":
        patterns = agent.analyze_decision_patterns()
        if args.format == "json":
            print(json.dumps(patterns, indent=2))
        else:
            print("🔍 Decision Patterns:")
            print("=" * 30)
            print(f"Total Decisions: {patterns['total_decisions']}")
            print(f"Average Confidence: {patterns['average_confidence']:.2%}")
            print("\nDecision Types:")
            for decision_type, count in patterns['decision_types'].items():
                print(f"  {decision_type}: {count}")
            print(f"\nConfidence Trends: {len(patterns['confidence_trends'])} data points")
    
    elif args.command == "stats":
        metrics = agent.get_metrics()
        patterns = agent.analyze_decision_patterns()
        recent_decisions = agent.get_recent_decisions(args.limit)
        
        print("📈 Autonomous Decision Agent Statistics")
        print("=" * 50)
        print(f"🤖 Total Decisions Made: {metrics['total_decisions']}")
        print(f"✅ Successful Decisions: {metrics['successful_decisions']}")
        print(f"📊 Success Rate: {metrics['successful_decisions']/max(metrics['total_decisions'], 1)*100:.1f}%")
        print(f"🎯 Average Confidence: {metrics['average_confidence']:.2%}")
        print(f"📈 Learning Rate: {metrics['learning_rate']:.2%}")
        print(f"🔍 Decision Types: {len(patterns.get('decision_types', {}))}")
        print(f"📅 Recent Decisions: {len(recent_decisions)}")
        
        if recent_decisions:
            print("\n🕒 Recent Decision Types:")
            for decision in recent_decisions:
                print(f"  • {decision.type.value} ({decision.priority.value}) - {decision.confidence:.1%} confidence")
    
    elif args.command == "help":
        print("🤖 Autonomous Decision Agent CLI Help")
        print("=" * 40)
        print("Commands:")
        print("  decide --prompt 'Your decision prompt'")
        print("    Make an autonomous decision based on the prompt")
        print()
        print("  history [--limit 5] [--format text|json]")
        print("    Show decision history")
        print()
        print("  metrics [--format text|json]")
        print("    Show decision performance metrics")
        print()
        print("  patterns [--limit 5] [--format text|json]")
        print("    Analyze decision patterns")
        print()
        print("  stats [--limit 5]")
        print("    Show comprehensive statistics")
        print()
        print("Examples:")
        print("  python cli/autonomous_decision_cli.py decide --prompt 'Optimize system performance'")
        print("  python cli/autonomous_decision_cli.py history --limit 10")
        print("  python cli/autonomous_decision_cli.py metrics --format json")

if __name__ == "__main__":
    main() 