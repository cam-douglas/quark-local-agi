#!/usr/bin/env python3
"""
ASYNC CLI
=========

Command-line interface for testing and managing the async orchestrator.
"""

import asyncio
import click
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.async_orchestrator import AsyncOrchestrator, get_async_orchestrator


@click.group()
def async_cli():
    """Async orchestrator management and testing."""
    pass


@async_cli.command()
@click.option('--max-workers', default=4, help='Maximum number of concurrent workers')
def start(max_workers: int):
    """Start the async orchestrator."""
    try:
        orchestrator = AsyncOrchestrator(max_workers=max_workers)
        click.echo(f"✅ Async orchestrator started with {max_workers} workers")
        
        # Keep running
        asyncio.run(_run_orchestrator(orchestrator))
        
    except Exception as e:
        click.echo(f"❌ Failed to start async orchestrator: {e}")


async def _run_orchestrator(orchestrator: AsyncOrchestrator):
    """Run the orchestrator in interactive mode."""
    click.echo("🤖 Async Orchestrator is ready!")
    click.echo("Type 'quit' to exit, 'stats' for performance stats")
    
    while True:
        try:
            user_input = click.prompt("You")
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'stats':
                stats = await orchestrator.get_performance_stats()
                click.echo(f"📊 Performance Stats: {stats}")
                continue
            
            # Process with async orchestrator
            result = await orchestrator.handle(user_input)
            
            if "error" in result:
                click.echo(f"❌ Error: {result['error']}")
            else:
                click.echo(f"🤖 Response: {result.get('results', {}).get('Reasoning', 'No response')}")
                click.echo(f"⏱️  Execution time: {result.get('execution_time', 0):.2f}s")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            click.echo(f"❌ Error: {e}")
    
    await orchestrator.shutdown()
    click.echo("👋 Async orchestrator shutdown complete")


@async_cli.command()
@click.option('--prompt', required=True, help='Test prompt')
@click.option('--max-workers', default=4, help='Maximum number of concurrent workers')
def test(prompt: str, max_workers: int):
    """Test the async orchestrator with a single prompt."""
    async def _test():
        orchestrator = AsyncOrchestrator(max_workers=max_workers)
        
        click.echo(f"🧪 Testing async orchestrator with prompt: {prompt}")
        
        start_time = asyncio.get_event_loop().time()
        result = await orchestrator.handle(prompt)
        end_time = asyncio.get_event_loop().time()
        
        execution_time = end_time - start_time
        
        if "error" in result:
            click.echo(f"❌ Error: {result['error']}")
        else:
            click.echo(f"✅ Success!")
            click.echo(f"📊 Category: {result.get('category', 'Unknown')}")
            click.echo(f"⏱️  Total execution time: {execution_time:.2f}s")
            click.echo(f"🔄 Parallel execution: {result.get('parallel_execution', False)}")
            
            # Show agent results
            results = result.get('results', {})
            for agent_name, agent_result in results.items():
                if isinstance(agent_result, dict) and agent_result.get('success'):
                    click.echo(f"  ✅ {agent_name}: {agent_result.get('execution_time', 0):.2f}s")
                else:
                    click.echo(f"  ❌ {agent_name}: Failed")
        
        await orchestrator.shutdown()
    
    asyncio.run(_test())


@async_cli.command()
@click.option('--prompts', required=True, help='Comma-separated list of test prompts')
@click.option('--max-workers', default=4, help='Maximum number of concurrent workers')
def benchmark(prompts: str, max_workers: int):
    """Benchmark the async orchestrator with multiple prompts."""
    async def _benchmark():
        orchestrator = AsyncOrchestrator(max_workers=max_workers)
        prompt_list = [p.strip() for p in prompts.split(',')]
        
        click.echo(f"🏁 Benchmarking async orchestrator with {len(prompt_list)} prompts")
        click.echo(f"🔧 Max workers: {max_workers}")
        
        results = []
        total_start_time = asyncio.get_event_loop().time()
        
        # Process all prompts concurrently
        tasks = [orchestrator.handle(prompt) for prompt in prompt_list]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_end_time = asyncio.get_event_loop().time()
        total_time = total_end_time - total_start_time
        
        successful = 0
        failed = 0
        total_execution_time = 0
        
        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                click.echo(f"❌ Prompt {i+1}: Error - {result}")
                failed += 1
            else:
                if "error" in result:
                    click.echo(f"❌ Prompt {i+1}: {result['error']}")
                    failed += 1
                else:
                    execution_time = result.get('execution_time', 0)
                    total_execution_time += execution_time
                    click.echo(f"✅ Prompt {i+1}: {execution_time:.2f}s")
                    successful += 1
        
        # Performance stats
        stats = await orchestrator.get_performance_stats()
        
        click.echo(f"\n📊 Benchmark Results:")
        click.echo(f"  Total time: {total_time:.2f}s")
        click.echo(f"  Successful: {successful}/{len(prompt_list)}")
        click.echo(f"  Failed: {failed}/{len(prompt_list)}")
        click.echo(f"  Average execution time: {total_execution_time/len(prompt_list):.2f}s")
        click.echo(f"  Throughput: {len(prompt_list)/total_time:.2f} prompts/second")
        click.echo(f"  Parallel execution count: {stats['performance_metrics']['parallel_execution_count']}")
        click.echo(f"  Concurrent agents: {stats['performance_metrics']['concurrent_agents']}")
        
        await orchestrator.shutdown()
    
    asyncio.run(_benchmark())


@async_cli.command()
@click.option('--max-workers', default=4, help='Maximum number of concurrent workers')
def performance(max_workers: int):
    """Show performance statistics."""
    async def _performance():
        orchestrator = AsyncOrchestrator(max_workers=max_workers)
        
        # Run a few test prompts to generate stats
        test_prompts = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Explain neural networks",
            "What are the benefits of AI?"
        ]
        
        click.echo("📈 Generating performance statistics...")
        
        for prompt in test_prompts:
            await orchestrator.handle(prompt)
        
        stats = await orchestrator.get_performance_stats()
        
        click.echo(f"📊 Performance Statistics:")
        click.echo(f"  Total tasks: {stats['execution_stats']['total_tasks']}")
        click.echo(f"  Completed tasks: {stats['execution_stats']['completed_tasks']}")
        click.echo(f"  Failed tasks: {stats['execution_stats']['failed_tasks']}")
        click.echo(f"  Average execution time: {stats['execution_stats']['average_execution_time']:.2f}s")
        click.echo(f"  Parallel execution count: {stats['performance_metrics']['parallel_execution_count']}")
        click.echo(f"  Concurrent agents: {stats['performance_metrics']['concurrent_agents']}")
        click.echo(f"  Throughput: {stats['performance_metrics']['throughput']:.2f} ops/second")
        click.echo(f"  Thread pool size: {stats['thread_pool_size']}")
        
        await orchestrator.shutdown()
    
    asyncio.run(_performance())


if __name__ == "__main__":
    async_cli() 