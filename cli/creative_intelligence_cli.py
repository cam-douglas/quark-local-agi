#!/usr/bin/env python3
"""
CLI for Creative Intelligence Agent
Provides command-line interface for creative content generation
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.creative_intelligence_agent import CreativeIntelligenceAgent

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Creative Intelligence Agent CLI")
    parser.add_argument("command", choices=[
        "create", "history", "metrics", "patterns", "stats", "help"
    ], help="Command to execute")
    parser.add_argument("--prompt", "-p", help="Creative prompt for create command")
    parser.add_argument("--style", "-s", help="Creative style (abstract, realistic, surreal, etc.)")
    parser.add_argument("--mood", "-m", help="Creative mood (joyful, calm, energetic, etc.)")
    parser.add_argument("--medium", "-d", help="Creative medium (visual_art, literature, music, etc.)")
    parser.add_argument("--audience", "-a", help="Target audience")
    parser.add_argument("--complexity", "-c", help="Complexity level (low, medium, high)")
    parser.add_argument("--collaboration", action="store_true", help="Enable collaboration mode")
    parser.add_argument("--limit", "-l", type=int, default=5, help="Limit for history/patterns")
    parser.add_argument("--format", "-f", choices=["text", "json"], default="text", help="Output format")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = CreativeIntelligenceAgent()
    agent.load_model()
    
    if args.command == "create":
        if not args.prompt:
            print("❌ Error: Prompt required for create command")
            print("Usage: python cli/creative_intelligence_cli.py create --prompt 'Your creative prompt'")
            return
        
        # Build kwargs for creative generation
        kwargs = {}
        if args.style:
            kwargs['style'] = args.style
        if args.mood:
            kwargs['mood'] = args.mood
        if args.medium:
            kwargs['medium'] = args.medium
        if args.audience:
            kwargs['audience'] = args.audience
        if args.complexity:
            kwargs['complexity'] = args.complexity
        if args.collaboration:
            kwargs['collaboration'] = True
        
        print("🎨 Creating creative work...")
        result = agent.generate(args.prompt, **kwargs)
        print(result)
        
    elif args.command == "history":
        history = agent.get_creative_history()
        if args.format == "json":
            print(json.dumps(history, indent=2, default=str))
        else:
            print("📋 Creative Work History:")
            print("=" * 50)
            for work in history[-args.limit:]:
                print(f"Title: {work['title']}")
                print(f"Type: {work['type']}")
                print(f"Style: {work['style']}")
                print(f"Mood: {work['mood']}")
                print(f"Medium: {work['context']['medium']}")
                print(f"Elements: {len(work['elements'])}")
                print(f"Timestamp: {work['timestamp']}")
                print("-" * 30)
    
    elif args.command == "metrics":
        metrics = agent.get_metrics()
        if args.format == "json":
            print(json.dumps(metrics, indent=2))
        else:
            print("📊 Creative Metrics:")
            print("=" * 30)
            print(f"Total Works: {metrics['total_works']}")
            print(f"Successful Works: {metrics['successful_works']}")
            print(f"Average Complexity: {metrics['average_complexity']:.2f}")
            print(f"Innovation Score: {metrics['innovation_score']:.2f}")
            print(f"Collaboration Rate: {metrics['collaboration_rate']:.2f}")
            print(f"Audience Engagement: {metrics['audience_engagement']:.2f}")
            print(f"Creative Evolution: {metrics['creative_evolution']:.2f}")
    
    elif args.command == "patterns":
        patterns = agent.analyze_creative_patterns()
        if args.format == "json":
            print(json.dumps(patterns, indent=2))
        else:
            print("🔍 Creative Patterns:")
            print("=" * 30)
            print(f"Total Works: {len(patterns.get('complexity_trends', []))}")
            print("\nStyle Distribution:")
            for style, count in patterns.get('style_distribution', {}).items():
                print(f"  {style}: {count}")
            print("\nMood Distribution:")
            for mood, count in patterns.get('mood_distribution', {}).items():
                print(f"  {mood}: {count}")
            print("\nType Distribution:")
            for work_type, count in patterns.get('type_distribution', {}).items():
                print(f"  {work_type}: {count}")
            print(f"\nComplexity Trends: {len(patterns.get('complexity_trends', []))} data points")
    
    elif args.command == "stats":
        metrics = agent.get_metrics()
        patterns = agent.analyze_creative_patterns()
        recent_works = agent.get_recent_works(args.limit)
        
        print("📈 Creative Intelligence Statistics")
        print("=" * 50)
        print(f"🎨 Total Works Created: {metrics['total_works']}")
        print(f"✅ Successful Works: {metrics['successful_works']}")
        print(f"📊 Success Rate: {metrics['successful_works']/max(metrics['total_works'], 1)*100:.1f}%")
        print(f"🎯 Average Complexity: {metrics['average_complexity']:.2f}")
        print(f"💡 Innovation Score: {metrics['innovation_score']:.2f}")
        print(f"🤝 Collaboration Rate: {metrics['collaboration_rate']:.2f}")
        print(f"👥 Audience Engagement: {metrics['audience_engagement']:.2f}")
        print(f"🔄 Creative Evolution: {metrics['creative_evolution']:.2f}")
        print(f"🎭 Style Types: {len(patterns.get('style_distribution', {}))}")
        print(f"😊 Mood Types: {len(patterns.get('mood_distribution', {}))}")
        print(f"📅 Recent Works: {len(recent_works)}")
        
        if recent_works:
            print("\n🕒 Recent Creative Works:")
            for work in recent_works:
                print(f"  • {work.title} ({work.type.value}) - {work.style.value} style, {work.mood.value} mood")
    
    elif args.command == "help":
        print("🎨 Creative Intelligence Agent CLI Help")
        print("=" * 40)
        print("Commands:")
        print("  create --prompt 'Your creative prompt'")
        print("    Create a new creative work based on the prompt")
        print()
        print("  history [--limit 5] [--format text|json]")
        print("    Show creative work history")
        print()
        print("  metrics [--format text|json]")
        print("    Show creative performance metrics")
        print()
        print("  patterns [--limit 5] [--format text|json]")
        print("    Analyze creative patterns")
        print()
        print("  stats [--limit 5]")
        print("    Show comprehensive statistics")
        print()
        print("Options for create command:")
        print("  --style: abstract, realistic, surreal, minimalist, expressive, technical, experimental")
        print("  --mood: joyful, melancholic, energetic, calm, mysterious, dramatic, playful")
        print("  --medium: visual_art, literature, music, design, mixed_media")
        print("  --audience: general, artists, children, professionals, etc.")
        print("  --complexity: low, medium, high")
        print("  --collaboration: Enable collaboration mode")
        print()
        print("Examples:")
        print("  python cli/creative_intelligence_cli.py create --prompt 'Abstract painting inspired by nature'")
        print("  python cli/creative_intelligence_cli.py create --prompt 'Joyful music' --style expressive --mood joyful")
        print("  python cli/creative_intelligence_cli.py history --limit 10")
        print("  python cli/creative_intelligence_cli.py metrics --format json")

if __name__ == "__main__":
    main() 