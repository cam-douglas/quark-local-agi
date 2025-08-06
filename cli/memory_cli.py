#!/usr/bin/env python3
"""
Memory CLI for Meta-Model AI Assistant
Provides commands for managing memories and memory operations
"""

import click
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.memory_agent import MemoryAgent
from core.memory_eviction import MemoryEvictionManager

@click.group()
def memory_cli():
    """Memory management commands for Meta-Model AI Assistant."""
    pass

@memory_cli.command()
@click.option('--content', '-c', required=True, help='Content to store')
@click.option('--type', '-t', default='conversation', help='Memory type')
@click.option('--metadata', '-m', help='Additional metadata (JSON)')
def store(content, type, metadata):
    """Store a new memory."""
    try:
        memory_agent = MemoryAgent()
        memory_agent._ensure_model()
        
        # Parse metadata if provided
        metadata_dict = {}
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                click.echo("❌ Invalid JSON metadata")
                return
        
        # Store memory
        memory_id = memory_agent.store_memory(content, type, metadata_dict)
        
        if memory_id:
            click.echo(f"✅ Memory stored with ID: {memory_id}")
        else:
            click.echo("❌ Failed to store memory")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@memory_cli.command()
@click.option('--query', '-q', required=True, help='Search query')
@click.option('--count', '-n', default=5, help='Number of results')
@click.option('--type', '-t', help='Filter by memory type')
def search(query, count, type):
    """Search for relevant memories."""
    try:
        memory_agent = MemoryAgent()
        memory_agent._ensure_model()
        
        memories = memory_agent.retrieve_memories(query, count, type)
        
        if not memories:
            click.echo("📭 No relevant memories found")
            return
        
        click.echo(f"🔍 Found {len(memories)} relevant memories:")
        click.echo("=" * 50)
        
        for i, memory in enumerate(memories, 1):
            click.echo(f"\n{i}. ID: {memory['id']}")
            click.echo(f"   Type: {memory['metadata'].get('type', 'unknown')}")
            click.echo(f"   Timestamp: {memory['metadata'].get('timestamp', 'unknown')}")
            click.echo(f"   Content: {memory['content'][:200]}...")
            if memory.get('distance'):
                click.echo(f"   Relevance: {1 - memory['distance']:.3f}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@memory_cli.command()
@click.option('--count', '-n', default=10, help='Number of recent memories')
def recent(count):
    """Show recent memories."""
    try:
        memory_agent = MemoryAgent()
        memory_agent._ensure_model()
        
        memories = memory_agent.get_recent_memories(count)
        
        if not memories:
            click.echo("📭 No memories found")
            return
        
        click.echo(f"📅 Recent memories ({len(memories)}):")
        click.echo("=" * 50)
        
        for i, memory in enumerate(memories, 1):
            click.echo(f"\n{i}. ID: {memory['id']}")
            click.echo(f"   Type: {memory['metadata'].get('type', 'unknown')}")
            click.echo(f"   Timestamp: {memory['metadata'].get('timestamp', 'unknown')}")
            click.echo(f"   Content: {memory['content'][:200]}...")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@memory_cli.command()
def stats():
    """Show memory statistics."""
    try:
        memory_agent = MemoryAgent()
        memory_agent._ensure_model()
        
        stats = memory_agent.get_memory_stats()
        
        click.echo("📊 Memory Statistics:")
        click.echo("=" * 30)
        click.echo(f"Total memories: {stats.get('total_memories', 0)}")
        click.echo(f"Oldest memory: {stats.get('oldest_memory', 'N/A')}")
        click.echo(f"Newest memory: {stats.get('newest_memory', 'N/A')}")
        
        if stats.get('memory_types'):
            click.echo("\nMemory types:")
            for memory_type, count in stats['memory_types'].items():
                click.echo(f"  {memory_type}: {count}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@memory_cli.command()
@click.option('--id', required=True, help='Memory ID to delete')
def delete(id):
    """Delete a specific memory."""
    try:
        memory_agent = MemoryAgent()
        memory_agent._ensure_model()
        
        if memory_agent.delete_memory(id):
            click.echo(f"✅ Memory {id} deleted")
        else:
            click.echo(f"❌ Failed to delete memory {id}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@memory_cli.command()
@click.option('--type', '-t', help='Memory type to clear (leave empty for all)')
def clear(type):
    """Clear memories."""
    try:
        memory_agent = MemoryAgent()
        memory_agent._ensure_model()
        
        if type:
            if memory_agent.clear_memories(type):
                click.echo(f"✅ Cleared all memories of type '{type}'")
            else:
                click.echo(f"❌ Failed to clear memories of type '{type}'")
        else:
            if memory_agent.clear_memories():
                click.echo("✅ Cleared all memories")
            else:
                click.echo("❌ Failed to clear memories")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@memory_cli.command()
def cleanup():
    """Run memory cleanup."""
    try:
        memory_agent = MemoryAgent()
        memory_agent._ensure_model()
        
        eviction_manager = MemoryEvictionManager(memory_agent)
        results = eviction_manager.run_cleanup()
        
        click.echo("🧹 Memory Cleanup Results:")
        click.echo("=" * 30)
        click.echo(f"Status: {results['status']}")
        click.echo(f"Timestamp: {results['timestamp']}")
        click.echo(f"Policies run: {', '.join(results['policies_run'])}")
        click.echo(f"Memories removed: {results['memories_removed']}")
        
        if results.get('errors'):
            click.echo(f"Errors: {', '.join(results['errors'])}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@memory_cli.command()
def health():
    """Show memory health report."""
    try:
        memory_agent = MemoryAgent()
        memory_agent._ensure_model()
        
        eviction_manager = MemoryEvictionManager(memory_agent)
        health_report = eviction_manager.get_memory_health_report()
        
        click.echo("🏥 Memory Health Report:")
        click.echo("=" * 30)
        click.echo(f"Status: {health_report['health_status']}")
        click.echo(f"Usage: {health_report['memory_usage_percent']:.1f}%")
        click.echo(f"Total memories: {health_report['total_memories']}")
        click.echo(f"Max memories: {health_report['max_memories']}")
        
        if health_report.get('recommendations'):
            click.echo("\nRecommendations:")
            for rec in health_report['recommendations']:
                click.echo(f"  • {rec}")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@memory_cli.command()
@click.option('--output', '-o', help='Output file for export')
def export(output):
    """Export all memories to JSON."""
    try:
        memory_agent = MemoryAgent()
        memory_agent._ensure_model()
        
        # Get all memories
        memories = memory_agent.get_recent_memories(10000)  # Get all
        
        if not memories:
            click.echo("📭 No memories to export")
            return
        
        # Prepare export data
        export_data = {
            'export_timestamp': memory_agent.get_memory_stats().get('newest_memory'),
            'total_memories': len(memories),
            'memories': memories
        }
        
        # Output
        if output:
            with open(output, 'w') as f:
                json.dump(export_data, f, indent=2)
            click.echo(f"✅ Exported {len(memories)} memories to {output}")
        else:
            click.echo(json.dumps(export_data, indent=2))
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")

if __name__ == '__main__':
    memory_cli()

