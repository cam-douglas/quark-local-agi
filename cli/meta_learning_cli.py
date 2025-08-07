#!/usr/bin/env python3
"""
Meta-Learning CLI for Quark AI Assistant
Command-line interface for meta-learning and self-reflection capabilities

Part of Pillar 16: Meta-Learning & Self-Reflection
"""

import click
import json
import time
from datetime import datetime
from typing import Dict, Any

from meta_learning.meta_learning_orchestrator import MetaLearningOrchestrator
from meta_learning.performance_monitor import PerformanceMonitor
from meta_learning.pipeline_reconfigurator import PipelineReconfigurator
from meta_learning.self_reflection_agent import SelfReflectionAgent

@click.group()
def meta_learning_cli():
    """Meta-Learning CLI for Quark AI Assistant"""
    pass

@meta_learning_cli.command()
def status():
    """Show meta-learning system status."""
    try:
        orchestrator = MetaLearningOrchestrator()
        
        click.echo("🤖 Meta-Learning System Status")
        click.echo("=" * 50)
        
        # Get component status
        component_status = orchestrator.get_component_status()
        
        # Display orchestrator status
        orchestrator_status = component_status['orchestrator']
        click.echo(f"🔄 Orchestrator: {'✅ Enabled' if orchestrator_status['enabled'] else '❌ Disabled'}")
        click.echo(f"   Auto-orchestration: {'✅ On' if orchestrator_status['auto_orchestration'] else '❌ Off'}")
        click.echo(f"   Total sessions: {orchestrator_status['total_sessions']}")
        
        # Display component status
        click.echo(f"\n📊 Performance Monitor: {'✅ Enabled' if component_status['performance_monitor']['enabled'] else '❌ Disabled'}")
        click.echo(f"🔄 Pipeline Reconfigurator: {'✅ Enabled' if component_status['pipeline_reconfigurator']['enabled'] else '❌ Disabled'}")
        click.echo(f"🤔 Self-Reflection Agent: {'✅ Enabled' if component_status['self_reflection_agent']['enabled'] else '❌ Disabled'}")
        
        # Display recent statistics
        stats = orchestrator.get_meta_learning_statistics()
        if stats['status'] == 'success':
            click.echo(f"\n📈 Recent Performance:")
            click.echo(f"   Average improvement score: {stats['average_improvement_score']:.2%}")
            click.echo(f"   Recent sessions: {stats['recent_sessions']}")
            click.echo(f"   Improvement trend: {stats['recent_improvement_trend']:+.2%}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@meta_learning_cli.command()
@click.option('--type', 'session_type', default='comprehensive', 
              help='Type of meta-learning session')
def run_session(session_type):
    """Run a meta-learning session."""
    try:
        orchestrator = MetaLearningOrchestrator()
        
        click.echo(f"🚀 Starting {session_type} meta-learning session...")
        click.echo("=" * 50)
        
        # Run the session
        result = orchestrator.run_meta_learning_session(session_type)
        
        if result['status'] == 'completed':
            click.echo(f"✅ Session completed: {result['session_id']}")
            click.echo(f"📊 Components used: {', '.join(result['components_used'])}")
            click.echo(f"💡 Insights generated: {result['insights_generated']}")
            click.echo(f"🔧 Optimizations applied: {result['optimizations_applied']}")
            click.echo(f"📈 Improvement score: {result['improvement_score']:.2%}")
            
            # Display detailed results
            if result.get('performance_results'):
                perf = result['performance_results']
                click.echo(f"\n📊 Performance Results:")
                click.echo(f"   Status: {perf['status']}")
                if perf.get('insights_count'):
                    click.echo(f"   Insights: {perf['insights_count']}")
            
            if result.get('reflection_results'):
                refl = result['reflection_results']
                click.echo(f"\n🤔 Reflection Results:")
                click.echo(f"   Type: {refl.get('reflection_type', 'unknown')}")
                click.echo(f"   Insights: {refl.get('insights_count', 0)}")
                click.echo(f"   Improvement score: {refl.get('improvement_score', 0):.2%}")
            
            if result.get('reconfiguration_results'):
                reconfig = result['reconfiguration_results']
                click.echo(f"\n🔧 Reconfiguration Results:")
                click.echo(f"   Status: {reconfig['status']}")
                click.echo(f"   Applied actions: {reconfig.get('applied_actions', 0)}")
        
        elif result['status'] == 'too_soon':
            click.echo(f"⏰ {result['message']}")
        elif result['status'] == 'disabled':
            click.echo(f"❌ {result['message']}")
        else:
            click.echo(f"❌ Session failed: {result.get('message', 'Unknown error')}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@meta_learning_cli.command()
@click.option('--component', required=True, 
              type=click.Choice(['performance_monitor', 'pipeline_reconfigurator', 'self_reflection_agent']),
              help='Component to optimize')
@click.option('--type', 'optimization_type', default='performance',
              help='Type of optimization to apply')
def optimize(component, optimization_type):
    """Run targeted optimization for a specific component."""
    try:
        orchestrator = MetaLearningOrchestrator()
        
        click.echo(f"🔧 Optimizing {component}...")
        click.echo("=" * 50)
        
        result = orchestrator.run_targeted_optimization(component, optimization_type)
        
        if result['status'] == 'completed':
            click.echo(f"✅ Optimization completed for {component}")
            click.echo(f"🔧 Optimizations applied: {result.get('optimizations_applied', 0)}")
            
            # Display component-specific results
            if component == 'performance_monitor' and 'new_thresholds' in result:
                click.echo(f"\n📊 New thresholds:")
                for metric, value in result['new_thresholds'].items():
                    click.echo(f"   {metric}: {value}")
            
            elif component == 'pipeline_reconfigurator' and 'new_threshold' in result:
                click.echo(f"\n🔧 New performance threshold: {result['new_threshold']:.2f}")
            
            elif component == 'self_reflection_agent' and 'new_interval' in result:
                click.echo(f"\n🤔 New reflection interval: {result['new_interval']:.0f} seconds")
        
        else:
            click.echo(f"❌ Optimization failed: {result.get('message', 'Unknown error')}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@meta_learning_cli.command()
def performance():
    """Show performance monitoring information."""
    try:
        monitor = PerformanceMonitor()
        
        click.echo("📊 Performance Monitoring")
        click.echo("=" * 50)
        
        # Get performance summary
        summary = monitor.get_performance_summary()
        
        if summary['status'] == 'success':
            click.echo(f"📈 System Health: {summary['system_health']['status']}")
            click.echo(f"🏥 Health Score: {summary['system_health']['score']:.2%}")
            click.echo(f"📊 Total snapshots: {summary['total_snapshots']}")
            click.echo(f"⚠️  Total alerts: {summary['total_alerts']}")
            
            # Display agent performance
            if summary.get('performance_summary'):
                click.echo(f"\n🤖 Agent Performance:")
                for agent, metrics in summary['performance_summary'].items():
                    click.echo(f"   {agent}:")
                    for metric, data in metrics.items():
                        click.echo(f"     {metric}: {data['average']:.2f} (avg)")
        
        else:
            click.echo(f"📊 {summary['message']}")
        
        # Show monitor statistics
        stats = monitor.get_monitor_statistics()
        click.echo(f"\n⚙️  Monitor Settings:")
        click.echo(f"   Monitoring: {'✅ Enabled' if stats['monitoring_enabled'] else '❌ Disabled'}")
        click.echo(f"   Alerts: {'✅ Enabled' if stats['alert_enabled'] else '❌ Disabled'}")
        click.echo(f"   Auto-recovery: {'✅ Enabled' if stats['auto_recovery'] else '❌ Disabled'}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@meta_learning_cli.command()
def reflection():
    """Run self-reflection analysis."""
    try:
        reflection_agent = SelfReflectionAgent()
        
        click.echo("🤔 Running Self-Reflection Analysis")
        click.echo("=" * 50)
        
        # Run reflection session
        result = reflection_agent.run_reflection_session(reflection_type="general")
        
        if result['status'] == 'completed':
            click.echo(f"✅ Reflection completed: {result['session_id']}")
            click.echo(f"🤔 Reflection type: {result['reflection_type']}")
            click.echo(f"💡 Insights generated: {result['insights_count']}")
            click.echo(f"🎯 Actionable insights: {result['actionable_insights']}")
            click.echo(f"📈 Improvement score: {result['improvement_score']:.2%}")
            
            # Display insights
            if result.get('insights'):
                click.echo(f"\n💭 Key Insights:")
                for i, insight in enumerate(result['insights'][:5], 1):  # Show top 5
                    priority = insight.get('priority', 'medium')
                    insight_type = insight.get('insight_type', 'unknown')
                    description = insight.get('description', 'No description')
                    
                    priority_icon = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}.get(priority, '⚪')
                    type_icon = {'strength': '💪', 'weakness': '⚠️', 'opportunity': '🎯', 'pattern': '📊'}.get(insight_type, '💭')
                    
                    click.echo(f"   {i}. {priority_icon} {type_icon} {description}")
        
        elif result['status'] == 'too_soon':
            click.echo(f"⏰ {result['message']}")
        elif result['status'] == 'disabled':
            click.echo(f"❌ {result['message']}")
        else:
            click.echo(f"❌ Reflection failed: {result.get('message', 'Unknown error')}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@meta_learning_cli.command()
def pipeline():
    """Show pipeline reconfiguration information."""
    try:
        reconfigurator = PipelineReconfigurator()
        
        click.echo("🔧 Pipeline Reconfiguration")
        click.echo("=" * 50)
        
        # Get reconfiguration statistics
        stats = reconfigurator.get_reconfiguration_statistics()
        
        click.echo(f"📊 Total configurations: {stats['total_configurations']}")
        click.echo(f"🔄 Auto-reconfiguration: {'✅ Enabled' if stats['auto_reconfiguration'] else '❌ Disabled'}")
        click.echo(f"🧪 Testing mode: {'✅ Enabled' if stats['testing_mode'] else '❌ Disabled'}")
        click.echo(f"🔄 Total reconfigurations: {stats['total_reconfigurations']}")
        click.echo(f"📈 Performance threshold: {stats['performance_threshold']:.2f}")
        
        if stats['active_configuration']:
            click.echo(f"✅ Active configuration: {stats['active_configuration']}")
        else:
            click.echo("❌ No active configuration")
        
        # Show available configurations
        if reconfigurator.pipeline_configurations:
            click.echo(f"\n📋 Available Configurations:")
            for config in reconfigurator.pipeline_configurations[:5]:  # Show top 5
                status_icon = '✅' if config.status == 'active' else '⏸️' if config.status == 'inactive' else '🧪'
                click.echo(f"   {status_icon} {config.config_id}: {config.agent_sequence}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@meta_learning_cli.command()
def statistics():
    """Show comprehensive meta-learning statistics."""
    try:
        orchestrator = MetaLearningOrchestrator()
        
        click.echo("📊 Meta-Learning Statistics")
        click.echo("=" * 50)
        
        stats = orchestrator.get_meta_learning_statistics()
        
        if stats['status'] == 'success':
            click.echo(f"📈 Total sessions: {stats['total_sessions']}")
            click.echo(f"💡 Total insights: {stats['total_insights']}")
            click.echo(f"🔧 Total optimizations: {stats['total_optimizations']}")
            click.echo(f"📊 Average improvement score: {stats['average_improvement_score']:.2%}")
            click.echo(f"🕒 Recent sessions (24h): {stats['recent_sessions']}")
            click.echo(f"📈 Recent improvement trend: {stats['recent_improvement_trend']:+.2%}")
            
            # Display component status
            component_status = stats['component_status']
            click.echo(f"\n🤖 Component Status:")
            for component, status in component_status.items():
                if component != 'orchestrator':  # Already shown in status command
                    enabled = status['enabled'] if isinstance(status, dict) else status
                    click.echo(f"   {component}: {'✅ Enabled' if enabled else '❌ Disabled'}")
        
        else:
            click.echo(f"📊 {stats['message']}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

if __name__ == '__main__':
    meta_learning_cli() 