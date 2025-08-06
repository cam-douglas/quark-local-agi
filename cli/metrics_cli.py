#!/usr/bin/env python3
"""
Metrics CLI for Meta-Model AI Assistant
Provides commands for managing metrics and evaluation
"""

import click
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.metrics_agent import MetricsAgent
from core.evaluation_framework import EvaluationFramework
from core.orchestrator import Orchestrator

@click.group()
def metrics_cli():
    """Metrics and evaluation commands for Meta-Model AI Assistant."""
    pass

@metrics_cli.command()
@click.option('--time-window', '-t', default=3600, help='Time window in seconds')
def summary(time_window):
    """Show metrics summary."""
    try:
        metrics_agent = MetricsAgent()
        
        # Get performance summary
        performance = metrics_agent.get_performance_summary(time_window=time_window)
        error_summary = metrics_agent.get_error_summary(time_window=time_window)
        latency_analysis = metrics_agent.get_latency_analysis(time_window=time_window)
        
        click.echo("📊 Metrics Summary")
        click.echo("=" * 50)
        
        # Performance metrics
        click.echo(f"Total Operations: {performance['total_operations']}")
        click.echo(f"Success Rate: {performance['success_rate']:.2%}")
        click.echo(f"Error Rate: {performance['error_rate']:.2%}")
        click.echo(f"Average Latency: {performance['average_latency']:.3f}s")
        if performance['total_operations'] > 0:
            click.echo(f"Min Latency: {performance['min_latency']:.3f}s")
            click.echo(f"Max Latency: {performance['max_latency']:.3f}s")
        
        # Error summary
        if error_summary['total_errors'] > 0:
            click.echo(f"\n❌ Error Summary:")
            click.echo(f"Total Errors: {error_summary['total_errors']}")
            click.echo("Most Common Errors:")
            for error_type, count in error_summary['most_common_errors']:
                click.echo(f"  {error_type}: {count}")
        
        # Latency percentiles
        if latency_analysis.get('percentiles'):
            click.echo(f"\n⏱️  Latency Percentiles:")
            percentiles = latency_analysis['percentiles']
            click.echo(f"P50: {percentiles['p50']:.3f}s")
            click.echo(f"P90: {percentiles['p90']:.3f}s")
            click.echo(f"P95: {percentiles['p95']:.3f}s")
            click.echo(f"P99: {percentiles['p99']:.3f}s")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@metrics_cli.command()
@click.option('--operation', '-o', help='Filter by operation name')
@click.option('--time-window', '-t', default=3600, help='Time window in seconds')
def performance(operation, time_window):
    """Show detailed performance metrics."""
    try:
        metrics_agent = MetricsAgent()
        performance = metrics_agent.get_performance_summary(operation, time_window)
        
        click.echo("🚀 Performance Metrics")
        click.echo("=" * 50)
        click.echo(f"Operation: {operation or 'All'}")
        click.echo(f"Time Window: {time_window}s")
        click.echo(f"Total Operations: {performance['total_operations']}")
        if performance['total_operations'] > 0:
            click.echo(f"Successful: {performance['successful_operations']}")
            click.echo(f"Failed: {performance['failed_operations']}")
            click.echo(f"Success Rate: {performance['success_rate']:.2%}")
            click.echo(f"Error Rate: {performance['error_rate']:.2%}")
            click.echo(f"Average Latency: {performance['average_latency']:.3f}s")
            click.echo(f"Average Tokens: {performance['average_tokens_used']:.1f}")
            click.echo(f"Average Memory: {performance['average_memory_usage']:.2f}MB")
        else:
            click.echo("No operations found in the specified time window.")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@metrics_cli.command()
@click.option('--time-window', '-t', default=3600, help='Time window in seconds')
def errors(time_window):
    """Show error analysis."""
    try:
        metrics_agent = MetricsAgent()
        error_summary = metrics_agent.get_error_summary(time_window)
        
        click.echo("❌ Error Analysis")
        click.echo("=" * 50)
        click.echo(f"Time Window: {time_window}s")
        click.echo(f"Total Errors: {error_summary['total_errors']}")
        
        if error_summary['error_types']:
            click.echo("\nError Types:")
            for error_type, count in error_summary['error_types'].items():
                click.echo(f"  {error_type}: {count}")
        
        if error_summary['recent_errors']:
            click.echo("\nRecent Errors:")
            for error in error_summary['recent_errors'][-5:]:
                click.echo(f"  [{error['timestamp']}] {error['operation']}: {error['error']}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@metrics_cli.command()
@click.option('--operation', '-o', help='Filter by operation name')
@click.option('--time-window', '-t', default=3600, help='Time window in seconds')
def latency(operation, time_window):
    """Show latency analysis."""
    try:
        metrics_agent = MetricsAgent()
        latency_analysis = metrics_agent.get_latency_analysis(operation, time_window)
        
        click.echo("⏱️  Latency Analysis")
        click.echo("=" * 50)
        click.echo(f"Operation: {operation or 'All'}")
        click.echo(f"Time Window: {time_window}s")
        
        if latency_analysis.get('percentiles'):
            click.echo("\nPercentiles:")
            percentiles = latency_analysis['percentiles']
            click.echo(f"P50: {percentiles['p50']:.3f}s")
            click.echo(f"P90: {percentiles['p90']:.3f}s")
            click.echo(f"P95: {percentiles['p95']:.3f}s")
            click.echo(f"P99: {percentiles['p99']:.3f}s")
        
        if latency_analysis.get('distribution'):
            click.echo("\nDistribution:")
            dist = latency_analysis['distribution']
            click.echo(f"Mean: {dist['mean']:.3f}s")
            click.echo(f"Median: {dist['median']:.3f}s")
            click.echo(f"Std Dev: {dist['std_dev']:.3f}s")
            click.echo(f"Min: {dist['min']:.3f}s")
            click.echo(f"Max: {dist['max']:.3f}s")
        
        click.echo(f"\nTotal Operations: {latency_analysis.get('total_operations', 0)}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@metrics_cli.command()
@click.option('--filename', '-f', help='Output filename')
def export(filename):
    """Export metrics to JSON."""
    try:
        metrics_agent = MetricsAgent()
        filepath = metrics_agent.export_metrics(filename)
        click.echo(f"✅ Metrics exported to: {filepath}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@metrics_cli.command()
@click.option('--days', '-d', default=30, help='Days to retain')
def clear(days):
    """Clear old metrics."""
    try:
        metrics_agent = MetricsAgent()
        metrics_agent.clear_old_metrics(days)
        click.echo(f"✅ Cleared metrics older than {days} days")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@metrics_cli.command()
@click.option('--test-suite', '-s', help='Test suite to run')
@click.option('--max-tests', '-m', default=10, help='Maximum number of tests')
def evaluate(test_suite, max_tests):
    """Run evaluation tests."""
    try:
        # Initialize evaluation framework
        evaluation_framework = EvaluationFramework()
        
        # Create a simple model handler for testing
        def model_handler(input_text):
            orchestrator = Orchestrator()
            result = orchestrator.handle(input_text)
            return result.get('results', {}).get('Reasoning', '')
        
        click.echo("🧪 Running Evaluation Tests")
        click.echo("=" * 50)
        click.echo(f"Test Suite: {test_suite or 'All'}")
        click.echo(f"Max Tests: {max_tests}")
        
        # Run evaluation
        results = evaluation_framework.run_evaluation(
            model_handler, 
            test_suite=test_suite, 
            max_tests=max_tests
        )
        
        # Display results
        click.echo(f"\n📊 Evaluation Results:")
        click.echo(f"Total Tests: {results['total_tests']}")
        click.echo(f"Successful: {results['successful_tests']}")
        click.echo(f"Failed: {results['failed_tests']}")
        click.echo(f"Accuracy: {results['accuracy']:.2%}")
        click.echo(f"Average Latency: {results['average_latency']:.3f}s")
        click.echo(f"Error Rate: {results['error_rate']:.2%}")
        
        # Generate report
        report = evaluation_framework.generate_evaluation_report(results)
        click.echo(f"\n📋 Report:\n{report}")
        
        # Export results
        export_path = evaluation_framework.export_evaluation_results(results)
        click.echo(f"✅ Results exported to: {export_path}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@metrics_cli.command()
@click.option('--iterations', '-i', default=10, help='Number of iterations')
def benchmark(iterations):
    """Run performance benchmark."""
    try:
        evaluation_framework = EvaluationFramework()
        
        def model_handler(input_text):
            orchestrator = Orchestrator()
            result = orchestrator.handle(input_text)
            return result.get('results', {}).get('Reasoning', '')
        
        click.echo("🏃 Running Performance Benchmark")
        click.echo("=" * 50)
        click.echo(f"Iterations: {iterations}")
        
        # Run benchmark
        results = evaluation_framework.run_benchmark(model_handler, iterations)
        
        # Display results
        click.echo(f"\n📊 Benchmark Results:")
        click.echo(f"Average Latency: {results['average_latency']:.3f}s")
        click.echo(f"Min Latency: {results['min_latency']:.3f}s")
        click.echo(f"Max Latency: {results['max_latency']:.3f}s")
        click.echo(f"Latency Std Dev: {results['latency_std']:.3f}s")
        click.echo(f"Average Tokens: {results['average_tokens']:.1f}")
        click.echo(f"Error Rate: {results['error_rate']:.2%}")
        click.echo(f"Throughput: {results['throughput']:.2f} requests/second")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@metrics_cli.command()
@click.option('--file1', '-f1', required=True, help='First evaluation results file')
@click.option('--file2', '-f2', required=True, help='Second evaluation results file')
def compare(file1, file2):
    """Compare two evaluation results."""
    try:
        evaluation_framework = EvaluationFramework()
        
        # Load results
        results1 = evaluation_framework.load_evaluation_results(file1)
        results2 = evaluation_framework.load_evaluation_results(file2)
        
        # Compare
        comparison = evaluation_framework.compare_evaluations(results1, results2)
        
        click.echo("📈 Evaluation Comparison")
        click.echo("=" * 50)
        click.echo(f"Accuracy Improvement: {comparison['accuracy_improvement']:.2%}")
        click.echo(f"Latency Improvement: {comparison['latency_improvement']:.3f}s")
        click.echo(f"Error Rate Improvement: {comparison['error_rate_improvement']:.2%}")
        
        click.echo(f"\nBaseline Results:")
        click.echo(f"  Accuracy: {results1['accuracy']:.2%}")
        click.echo(f"  Latency: {results1['average_latency']:.3f}s")
        click.echo(f"  Error Rate: {results1['error_rate']:.2%}")
        
        click.echo(f"\nImproved Results:")
        click.echo(f"  Accuracy: {results2['accuracy']:.2%}")
        click.echo(f"  Latency: {results2['average_latency']:.3f}s")
        click.echo(f"  Error Rate: {results2['error_rate']:.2%}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

if __name__ == '__main__':
    metrics_cli()

