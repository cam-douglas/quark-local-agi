#!/usr/bin/env python3
"""
Self-Improvement CLI for Quark AI Assistant
Provides commands for managing self-improvement and capability bootstrapping
"""

import click
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.self_improvement_agent import SelfImprovementAgent
from core.capability_bootstrapping import CapabilityBootstrapping

@click.group()
def self_improvement_cli():
    """Self-improvement and capability bootstrapping commands."""
    pass

@self_improvement_cli.command()
def reflection():
    """Run self-reflection analysis."""
    try:
        agent = SelfImprovementAgent()
        
        click.echo("🤔 Running Self-Reflection Analysis")
        click.echo("=" * 50)
        
        # Run self-reflection
        reflection_result = agent.run_self_reflection()
        
        if reflection_result.get('status') == 'disabled':
            click.echo("❌ Self-reflection is disabled")
            return
        
        # Display performance analysis
        performance = reflection_result.get('performance_analysis', {})
        click.echo(f"📊 Performance Analysis:")
        click.echo(f"  Total Examples: {performance.get('total_examples', 0)}")
        click.echo(f"  Average Feedback: {performance.get('average_feedback', 0):.2%}")
        
        # Display improvement opportunities
        opportunities = performance.get('improvement_opportunities', [])
        if opportunities:
            click.echo(f"\n🎯 Improvement Opportunities:")
            for i, opp in enumerate(opportunities, 1):
                click.echo(f"  {i}. {opp['category']}: {opp['gap']:.2%} improvement needed")
        else:
            click.echo(f"\n✅ No improvement opportunities identified")
        
        # Display recommendations
        recommendations = reflection_result.get('recommendations', [])
        if recommendations:
            click.echo(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                priority = rec.get('priority', 'medium')
                action = rec.get('action', '')
                click.echo(f"  {i}. [{priority.upper()}] {action}")
        
        # Display insights
        insights = reflection_result.get('reflection_insights', [])
        if insights:
            click.echo(f"\n💭 Insights:")
            for insight in insights:
                click.echo(f"  • {insight}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@self_improvement_cli.command()
@click.option('--input-text', '-i', required=True, help='Input text')
@click.option('--expected-output', '-e', required=True, help='Expected output')
@click.option('--actual-output', '-a', required=True, help='Actual output')
@click.option('--category', '-c', default='general', help='Category')
@click.option('--feedback-score', '-f', type=float, help='Feedback score (0-1)')
def add_example(input_text, expected_output, actual_output, category, feedback_score):
    """Add a learning example."""
    try:
        agent = SelfImprovementAgent()
        
        example_id = agent.add_learning_example(
            input_text=input_text,
            expected_output=expected_output,
            actual_output=actual_output,
            feedback_score=feedback_score,
            category=category
        )
        
        if example_id:
            click.echo(f"✅ Learning example added with ID: {example_id}")
        else:
            click.echo("❌ Failed to add learning example")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@self_improvement_cli.command()
@click.option('--target-improvement', '-t', default=0.05, help='Target improvement (0-1)')
def fine_tuning(target_improvement):
    """Run automated fine-tuning."""
    try:
        agent = SelfImprovementAgent()
        
        click.echo("🔧 Running Automated Fine-Tuning")
        click.echo("=" * 50)
        
        result = agent.run_automated_fine_tuning(target_improvement)
        
        if result.get('status') == 'disabled':
            click.echo("❌ Automated fine-tuning is disabled")
            return
        elif result.get('status') == 'insufficient_data':
            click.echo(f"❌ Insufficient data: {result.get('available_examples', 0)} examples available")
            return
        elif result.get('status') == 'no_improvements_needed':
            click.echo("✅ No improvements needed")
            return
        
        click.echo(f"✅ Fine-tuning completed")
        click.echo(f"  Session ID: {result.get('session_id')}")
        click.echo(f"  Improvements Made: {result.get('improvements_made', 0)}")
        click.echo(f"  Performance Gain: {result.get('performance_gain', 0):.2%}")
        click.echo(f"  Duration: {result.get('duration', 0):.2f}s")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@self_improvement_cli.command()
def statistics():
    """Show learning statistics."""
    try:
        agent = SelfImprovementAgent()
        
        stats = agent.get_learning_statistics()
        
        click.echo("📊 Learning Statistics")
        click.echo("=" * 50)
        click.echo(f"Total Examples: {stats.get('total_examples', 0)}")
        click.echo(f"Total Sessions: {stats.get('total_sessions', 0)}")
        click.echo(f"Average Feedback: {stats.get('average_feedback', 0):.2%}")
        click.echo(f"Improvement Trend: {stats.get('improvement_trend', 'no_data')}")
        
        # Show recent performance
        recent_performance = stats.get('recent_performance', {})
        if recent_performance.get('total_examples', 0) > 0:
            click.echo(f"\n📈 Recent Performance (24h):")
            click.echo(f"  Examples: {recent_performance.get('total_examples', 0)}")
            click.echo(f"  Average Feedback: {recent_performance.get('average_feedback', 0):.2%}")
            
            opportunities = recent_performance.get('improvement_opportunities', [])
            if opportunities:
                click.echo(f"  Improvement Opportunities: {len(opportunities)}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@self_improvement_cli.command()
def capabilities():
    """Show capability status."""
    try:
        bootstrapping = CapabilityBootstrapping()
        
        status = bootstrapping.get_capability_status()
        
        click.echo("🎯 Capability Status")
        click.echo("=" * 50)
        click.echo(f"Total Capabilities: {status.get('total_capabilities', 0)}")
        click.echo(f"Not Started: {status.get('not_started', 0)}")
        click.echo(f"Learning: {status.get('learning', 0)}")
        click.echo(f"Mastered: {status.get('mastered', 0)}")
        click.echo(f"Failed: {status.get('failed', 0)}")
        
        # Show individual capabilities
        capabilities = status.get('capabilities', {})
        if capabilities:
            click.echo(f"\n📋 Individual Capabilities:")
            for name, cap in capabilities.items():
                status_emoji = {
                    'not_started': '⏳',
                    'learning': '📚',
                    'mastered': '✅',
                    'failed': '❌'
                }.get(cap['status'], '❓')
                
                click.echo(f"  {status_emoji} {name} ({cap['category']}, {cap['complexity']})")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@self_improvement_cli.command()
@click.option('--capability', '-c', required=True, help='Capability name')
def start_learning(capability):
    """Start learning a capability."""
    try:
        bootstrapping = CapabilityBootstrapping()
        
        click.echo(f"🎯 Starting Learning: {capability}")
        click.echo("=" * 50)
        
        result = bootstrapping.start_capability_learning(capability)
        
        if result.get('status') == 'error':
            click.echo(f"❌ Error: {result.get('message')}")
        elif result.get('status') == 'prerequisites_not_met':
            click.echo(f"❌ Prerequisites not met: {result.get('missing_prerequisites')}")
        elif result.get('status') == 'too_many_active':
            click.echo(f"❌ Too many active capabilities: {result.get('active_capabilities')}/{result.get('max_allowed')}")
        elif result.get('status') == 'started':
            click.echo(f"✅ Learning started")
            click.echo(f"  Learning Tasks: {result.get('learning_tasks', 0)}")
            click.echo(f"  Estimated Duration: {result.get('estimated_duration', 0)} minutes")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@self_improvement_cli.command()
@click.option('--capability', '-c', required=True, help='Capability name')
def evaluate_progress(capability):
    """Evaluate progress on a capability."""
    try:
        bootstrapping = CapabilityBootstrapping()
        
        click.echo(f"📊 Evaluating Progress: {capability}")
        click.echo("=" * 50)
        
        result = bootstrapping.evaluate_capability_progress(capability)
        
        if result.get('status') == 'error':
            click.echo(f"❌ Error: {result.get('message')}")
        elif result.get('status') == 'no_progress':
            click.echo(f"❌ No progress: {result.get('message')}")
        else:
            click.echo(f"Status: {result.get('status')}")
            click.echo(f"Total Tasks: {result.get('total_tasks', 0)}")
            click.echo(f"Completed Tasks: {result.get('completed_tasks', 0)}")
            click.echo(f"Completion Rate: {result.get('completion_rate', 0):.2%}")
            click.echo(f"Average Score: {result.get('average_score', 0):.2%}")
            click.echo(f"Progress: {result.get('progress_percentage', 0):.1f}%")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@self_improvement_cli.command()
@click.option('--filename', '-f', help='Output filename')
def export(filename):
    """Export learning data."""
    try:
        agent = SelfImprovementAgent()
        bootstrapping = CapabilityBootstrapping()
        
        # Export learning data
        learning_filepath = agent.export_learning_data(filename)
        click.echo(f"✅ Learning data exported to: {learning_filepath}")
        
        # Export capability data
        capability_filepath = bootstrapping.export_capability_data()
        click.echo(f"✅ Capability data exported to: {capability_filepath}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@self_improvement_cli.command()
def opportunities():
    """Show learning opportunities."""
    try:
        bootstrapping = CapabilityBootstrapping()
        
        # Simulate some user interactions
        user_interactions = [
            {'category': 'text_processing', 'input': 'Summarize this text'},
            {'category': 'programming', 'input': 'Write a function'},
            {'category': 'reasoning', 'input': 'Solve this problem'}
        ]
        
        opportunities = bootstrapping.identify_learning_opportunities(user_interactions)
        
        click.echo("🎯 Learning Opportunities")
        click.echo("=" * 50)
        
        if opportunities:
            for i, opp in enumerate(opportunities, 1):
                capability = opp['capability']
                click.echo(f"{i}. {capability.name}")
                click.echo(f"   Description: {capability.description}")
                click.echo(f"   Category: {capability.category}")
                click.echo(f"   Complexity: {capability.complexity}")
                click.echo(f"   Usefulness Score: {opp['usefulness_score']:.2%}")
                click.echo(f"   Priority: {opp['priority']}")
                click.echo(f"   Reason: {opp['reason']}")
                click.echo()
        else:
            click.echo("No learning opportunities identified")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

if __name__ == '__main__':
    self_improvement_cli() 