#!/usr/bin/env python3
"""
Streaming CLI for Meta-Model AI Assistant
=========================================

Provides commands for managing streaming, WebSocket connections, and real-time features.
"""

import click
import json
import asyncio
import time
from datetime import datetime
from typing import Dict, Any

from core.streaming_manager import get_streaming_manager, create_stream_session
from core.websocket_manager import get_websocket_manager
from core.safety_enforcement import get_safety_enforcement


@click.group()
def streaming():
    """Streaming and real-time management commands."""
    pass


@streaming.command()
def status():
    """Check streaming system status."""
    click.secho("🚀 STREAMING SYSTEM STATUS", fg="blue", bold=True)
    click.echo()
    
    streaming_manager = get_streaming_manager()
    websocket_manager = get_websocket_manager()
    
    # Get streaming stats
    streaming_stats = streaming_manager.get_streaming_stats()
    websocket_stats = websocket_manager.get_connection_stats()
    
    click.secho("📊 STREAMING STATISTICS:", fg="yellow", bold=True)
    click.echo(f"Active streams: {streaming_stats['active_streams']}")
    click.echo(f"Total streams: {streaming_stats['total_streams']}")
    click.echo(f"Total chunks: {streaming_stats['total_chunks']}")
    click.echo(f"Total tokens: {streaming_stats['total_tokens']}")
    click.echo()
    
    click.secho("🔌 WEBSOCKET STATISTICS:", fg="yellow", bold=True)
    click.echo(f"Active connections: {websocket_stats['active_connections']}")
    click.echo(f"Total connections: {websocket_stats['total_connections']}")
    click.echo(f"Max connections: {websocket_stats['max_connections']}")
    click.echo()
    
    click.secho("⚙️  CONFIGURATION:", fg="cyan", bold=True)
    config = streaming_stats['config']
    click.echo(f"Chunk size: {config['chunk_size']} tokens")
    click.echo(f"Stream delay: {config['stream_delay']} seconds")
    click.echo(f"Max stream time: {config['max_stream_time']} seconds")
    click.echo(f"Progressive output: {'✅' if config['enable_progressive'] else '❌'}")
    click.echo(f"Metadata enabled: {'✅' if config['enable_metadata'] else '❌'}")
    click.echo()
    
    click.secho("✅ Streaming system is active and ready", fg="green")


@streaming.command()
def streams():
    """List all active streams."""
    click.secho("📺 ACTIVE STREAMS", fg="blue", bold=True)
    click.echo()
    
    streaming_manager = get_streaming_manager()
    active_streams = streaming_manager.get_active_streams()
    
    if not active_streams:
        click.secho("No active streams", fg="yellow")
        return
    
    for i, stream_id in enumerate(active_streams, 1):
        status = streaming_manager.get_stream_status(stream_id)
        if status:
            click.secho(f"{i}. Stream: {stream_id}", fg="green")
            click.echo(f"   Status: {status['status']}")
            click.echo(f"   Chunks: {status['total_chunks']}")
            click.echo(f"   Tokens: {status['total_tokens']}")
            click.echo(f"   Duration: {status['duration']:.1f}s")
            click.echo()


@streaming.command()
@click.argument('stream_id')
def stream_info(stream_id):
    """Get detailed information about a specific stream."""
    click.secho(f"📺 STREAM INFO: {stream_id}", fg="blue", bold=True)
    click.echo()
    
    streaming_manager = get_streaming_manager()
    status = streaming_manager.get_stream_status(stream_id)
    
    if not status:
        click.secho(f"❌ Stream not found: {stream_id}", fg="red")
        return
    
    click.secho("STATUS:", fg="yellow", bold=True)
    click.echo(f"Stream ID: {status['stream_id']}")
    click.echo(f"Status: {status['status']}")
    click.echo(f"Total chunks: {status['total_chunks']}")
    click.echo(f"Total tokens: {status['total_tokens']}")
    click.echo(f"Created: {datetime.fromtimestamp(status['created_at'])}")
    click.echo(f"Last chunk: {datetime.fromtimestamp(status['last_chunk_time'])}")
    click.echo(f"Duration: {status['duration']:.1f} seconds")
    click.echo()
    
    # Show stream history
    history = streaming_manager.get_stream_history(stream_id)
    if history:
        click.secho("HISTORY:", fg="yellow", bold=True)
        for i, chunk in enumerate(history[-10:], 1):  # Last 10 chunks
            click.echo(f"{i}. [{chunk.chunk_type}] {chunk.content[:50]}...")
            if chunk.metadata:
                click.echo(f"   Metadata: {chunk.metadata}")


@streaming.command()
def connections():
    """List all active WebSocket connections."""
    click.secho("🔌 ACTIVE WEBSOCKET CONNECTIONS", fg="blue", bold=True)
    click.echo()
    
    websocket_manager = get_websocket_manager()
    active_connections = websocket_manager.active_connections
    
    if not active_connections:
        click.secho("No active WebSocket connections", fg="yellow")
        return
    
    for i, (connection_id, connection) in enumerate(active_connections.items(), 1):
        click.secho(f"{i}. Connection: {connection_id}", fg="green")
        click.echo(f"   Path: {connection['path']}")
        click.echo(f"   Status: {connection['status']}")
        click.echo(f"   Connected: {connection['connected_at']}")
        click.echo(f"   Last heartbeat: {connection['last_heartbeat']}")


@streaming.command()
@click.option('--message', '-m', required=True, help='Message to stream')
@click.option('--stream-id', '-s', help='Custom stream ID')
def test_stream(message, stream_id):
    """Test streaming with a sample message."""
    click.secho(f"🧪 TESTING STREAMING", fg="blue", bold=True)
    click.echo(f"Message: {message}")
    click.echo()
    
    if not stream_id:
        stream_id = f"test_stream_{int(time.time())}"
    
    streaming_manager = get_streaming_manager()
    
    # Create test response
    test_response = f"This is a test response to: '{message}'. The streaming system is working correctly."
    
    click.secho("Streaming response:", fg="yellow")
    
    async def run_test():
        async for chunk in streaming_manager.stream_response(message, test_response, stream_id):
            if chunk.chunk_type == "metadata":
                click.echo(f"📊 Metadata: {chunk.metadata}")
            elif chunk.chunk_type == "token":
                click.echo(f"📝 {chunk.content}", nl=False)
            elif chunk.chunk_type == "complete":
                click.echo(f"\n✅ Stream completed: {chunk.metadata}")
    
    asyncio.run(run_test())


@streaming.command()
def cleanup():
    """Clean up expired streams and connections."""
    click.secho("🧹 CLEANING UP EXPIRED STREAMS AND CONNECTIONS", fg="blue", bold=True)
    click.echo()
    
    streaming_manager = get_streaming_manager()
    websocket_manager = get_websocket_manager()
    
    # Clean up expired streams
    expired_streams = streaming_manager.cleanup_expired_streams()
    click.echo(f"Cleaned up {expired_streams} expired streams")
    
    # Clean up expired connections
    asyncio.run(websocket_manager.cleanup_expired_connections())
    
    click.secho("✅ Cleanup completed", fg="green")


@streaming.command()
def config():
    """Show streaming configuration."""
    click.secho("⚙️  STREAMING CONFIGURATION", fg="blue", bold=True)
    click.echo()
    
    streaming_manager = get_streaming_manager()
    websocket_manager = get_websocket_manager()
    
    click.secho("STREAMING CONFIG:", fg="yellow", bold=True)
    for key, value in streaming_manager.stream_config.items():
        click.echo(f"  {key}: {value}")
    
    click.echo()
    click.secho("WEBSOCKET CONFIG:", fg="yellow", bold=True)
    for key, value in websocket_manager.websocket_config.items():
        click.echo(f"  {key}: {value}")


@streaming.command()
@click.option('--host', default='localhost', help='Host to bind to')
@click.option('--port', default=8765, help='Port to bind to')
def start_websocket_server(host, port):
    """Start WebSocket server."""
    click.secho(f"🚀 STARTING WEBSOCKET SERVER", fg="blue", bold=True)
    click.echo(f"Host: {host}")
    click.echo(f"Port: {port}")
    click.echo()
    
    from core.websocket_manager import start_websocket_server
    
    async def run_server():
        server = await start_websocket_server(host, port)
        click.secho(f"✅ WebSocket server started on ws://{host}:{port}", fg="green")
        
        # Keep server running
        await asyncio.Future()  # Run forever
    
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        click.secho("\n👋 WebSocket server stopped", fg="blue")


@streaming.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
def start_fastapi_server(host, port):
    """Start FastAPI server."""
    click.secho(f"🚀 STARTING FASTAPI SERVER", fg="blue", bold=True)
    click.echo(f"Host: {host}")
    click.echo(f"Port: {port}")
    click.echo()
    
    import uvicorn
    from web.fastapi_app import app
    
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info"
        )
    except KeyboardInterrupt:
        click.secho("\n👋 FastAPI server stopped", fg="blue")


@streaming.command()
def safety_status():
    """Check safety status for streaming operations."""
    click.secho("🔒 STREAMING SAFETY STATUS", fg="blue", bold=True)
    click.echo()
    
    safety_enforcement = get_safety_enforcement()
    safety_report = safety_enforcement.get_safety_report()
    
    click.secho("SAFETY STATISTICS:", fg="yellow", bold=True)
    click.echo(f"Total actions: {safety_report['total_actions']}")
    click.echo(f"Blocked actions: {safety_report['blocked_actions']}")
    
    if safety_report['blocked_actions'] > 0:
        click.secho("⚠️  Some actions have been blocked for safety reasons", fg="yellow")
    
    click.echo()
    click.secho("✅ Safety system is protecting streaming operations", fg="green")


if __name__ == "__main__":
    streaming() 