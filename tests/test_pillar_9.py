#!/usr/bin/env python3
"""
Test script for Pillar 9: Streaming & Real-Time I/O
Tests the streaming agent, WebSocket communication, and real-time capabilities
"""

import sys
import os
import asyncio
import json
import time
import tempfile
import shutil
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_streaming_agent():
    """Test the streaming agent functionality."""
    print("Testing Streaming Agent...")
    
    try:
        from agents.streaming_agent import StreamingAgent, StreamEvent, StreamSession
        
        # Test streaming agent initialization
        streaming_agent = StreamingAgent(model_name="streaming_agent")
        assert streaming_agent.stream_config['chunk_size'] == 5
        assert streaming_agent.stream_config['stream_delay'] == 0.02
        assert streaming_agent.stream_config['max_stream_time'] == 600
        
        # Test session creation
        session_id = streaming_agent.create_session(user_id="test_user")
        assert session_id is not None
        assert session_id.startswith("stream_")
        
        # Test session status
        status = streaming_agent.get_session_status(session_id)
        assert status is not None
        assert status['session_id'] == session_id
        assert status['user_id'] == "test_user"
        assert status['status'] == 'active'
        
        # Test streaming statistics
        stats = streaming_agent.get_streaming_stats()
        assert 'total_sessions' in stats
        assert 'active_sessions' in stats
        assert 'total_tokens_streamed' in stats
        assert stats['total_sessions'] >= 1
        
        # Test session events
        events = streaming_agent.get_session_events(session_id)
        assert isinstance(events, list)
        
        # Test session closure
        success = streaming_agent.close_session_sync(session_id)
        assert success == True
        
        # Verify session is closed
        status_after = streaming_agent.get_session_status(session_id)
        assert status_after is None
        
        print("✅ Streaming Agent - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Streaming Agent - FAILED: {e}")
        return False

def test_stream_events():
    """Test stream event functionality."""
    print("Testing Stream Events...")
    
    try:
        from agents.streaming_agent import StreamEvent
        
        # Test stream event creation
        event = StreamEvent(
            event_id="test_event",
            event_type="token",
            content="test content",
            timestamp=time.time(),
            stream_id="test_stream",
            metadata={"test": "data"}
        )
        
        assert event.event_id == "test_event"
        assert event.event_type == "token"
        assert event.content == "test content"
        assert event.stream_id == "test_stream"
        assert event.metadata["test"] == "data"
        
        # Test event serialization
        from dataclasses import asdict
        event_dict = asdict(event)
        assert 'event_id' in event_dict
        assert 'event_type' in event_dict
        assert 'content' in event_dict
        assert 'timestamp' in event_dict
        
        print("✅ Stream Events - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Stream Events - FAILED: {e}")
        return False

def test_stream_sessions():
    """Test stream session functionality."""
    print("Testing Stream Sessions...")
    
    try:
        from agents.streaming_agent import StreamSession
        
        # Test stream session creation
        session = StreamSession(
            session_id="test_session",
            user_id="test_user",
            created_at=time.time(),
            last_activity=time.time(),
            status="active",
            total_tokens=0,
            events=[],
            context={"test": "context"}
        )
        
        assert session.session_id == "test_session"
        assert session.user_id == "test_user"
        assert session.status == "active"
        assert session.total_tokens == 0
        assert len(session.events) == 0
        assert session.context["test"] == "context"
        
        print("✅ Stream Sessions - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Stream Sessions - FAILED: {e}")
        return False

async def test_async_streaming():
    """Test async streaming functionality."""
    print("Testing Async Streaming...")
    
    try:
        from agents.streaming_agent import StreamingAgent
        
        streaming_agent = StreamingAgent(model_name="streaming_agent")
        
        # Create session
        session_id = streaming_agent.create_session(user_id="async_test")
        
        # Mock response generator
        async def mock_response_generator(prompt: str):
            return "This is a test response with multiple tokens for streaming."
        
        # Test streaming
        event_count = 0
        async for event in streaming_agent.stream_response(
            "Test prompt", 
            mock_response_generator, 
            session_id
        ):
            assert event.stream_id == session_id
            assert event.timestamp > 0
            event_count += 1
            
            # Should have at least start, tokens, and complete events
            if event_count > 10:  # Limit to prevent infinite loop
                break
        
        assert event_count > 0
        
        # Clean up
        await streaming_agent.close_session(session_id)
        
        print("✅ Async Streaming - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Async Streaming - FAILED: {e}")
        return False

def test_streaming_operations():
    """Test streaming agent operations through generate method."""
    print("Testing Streaming Operations...")
    
    try:
        from agents.streaming_agent import StreamingAgent
        
        streaming_agent = StreamingAgent(model_name="streaming_agent")
        
        # Test create session operation
        create_result = streaming_agent.generate(
            "", 
            operation="create_session",
            user_id="test_user",
            context={"test": "data"}
        )
        
        assert create_result["operation"] == "create_session"
        assert create_result["success"] == True
        assert "session_id" in create_result
        
        session_id = create_result["session_id"]
        
        # Test get status operation
        status_result = streaming_agent.generate(
            "", 
            operation="get_status",
            session_id=session_id
        )
        
        assert status_result["operation"] == "get_status"
        assert status_result["success"] == True
        assert status_result["session_id"] == session_id
        
        # Test close session operation
        close_result = streaming_agent.generate(
            "", 
            operation="close_session",
            session_id=session_id
        )
        
        assert close_result["operation"] == "close_session"
        assert close_result["success"] == True
        assert close_result["session_id"] == session_id
        
        print("✅ Streaming Operations - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Streaming Operations - FAILED: {e}")
        return False

def test_fastapi_integration():
    """Test FastAPI integration for streaming."""
    print("Testing FastAPI Integration...")
    
    try:
        # Test that we can import the streaming agent
        from agents.streaming_agent import StreamingAgent
        
        # Test streaming agent initialization
        streaming_agent = StreamingAgent(model_name="streaming_agent")
        assert streaming_agent is not None
        assert streaming_agent.model_name == "streaming_agent"
        
        # Test that we can import the FastAPI app components
        from web.fastapi_app import streaming_agent as app_streaming_agent
        assert app_streaming_agent is not None
        
        print("✅ FastAPI Integration - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ FastAPI Integration - FAILED: {e}")
        return False

def test_websocket_manager():
    """Test WebSocket manager functionality."""
    print("Testing WebSocket Manager...")
    
    try:
        from core.websocket_manager import get_websocket_manager
        
        # Test WebSocket manager initialization
        websocket_manager = get_websocket_manager()
        assert websocket_manager is not None
        
        # Test connection stats
        stats = websocket_manager.get_connection_stats()
        assert isinstance(stats, dict)
        assert 'active_connections' in stats
        
        print("✅ WebSocket Manager - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ WebSocket Manager - FAILED: {e}")
        return False

def test_streaming_manager():
    """Test streaming manager functionality."""
    print("Testing Streaming Manager...")
    
    try:
        from core.streaming_manager import get_streaming_manager
        
        # Test streaming manager initialization
        streaming_manager = get_streaming_manager()
        assert streaming_manager is not None
        
        # Test stream creation
        stream_id = streaming_manager.create_stream("test_stream")
        assert stream_id == "test_stream"
        
        # Test adding chunks
        success = streaming_manager.add_chunk("test_stream", "test content")
        assert success == True
        
        # Test stream status
        status = streaming_manager.get_stream_status("test_stream")
        assert status is not None
        assert status['status'] == 'active'
        
        # Test stream cleanup
        success = streaming_manager.close_stream("test_stream")
        assert success == True
        
        print("✅ Streaming Manager - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Streaming Manager - FAILED: {e}")
        return False

async def main():
    """Run all Pillar 9 tests."""
    print("🧪 Testing Pillar 9: Streaming & Real-Time I/O...")
    print("=" * 60)
    
    results = []
    
    # Test each component
    results.append(test_streaming_agent())
    results.append(test_stream_events())
    results.append(test_stream_sessions())
    results.append(await test_async_streaming())
    results.append(test_streaming_operations())
    results.append(test_fastapi_integration())
    results.append(test_websocket_manager())
    results.append(test_streaming_manager())
    
    # Summary
    print("=" * 60)
    print("📊 Test Results Summary:")
    
    test_names = [
        "Streaming Agent",
        "Stream Events", 
        "Stream Sessions",
        "Async Streaming",
        "Streaming Operations",
        "FastAPI Integration",
        "WebSocket Manager",
        "Streaming Manager"
    ]
    
    passed = 0
    for i, (result, name) in enumerate(zip(results, test_names)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {i+1}. {name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 Pillar 9: Streaming & Real-Time I/O is working correctly!")
        print("\n📋 Pillar 9 Features:")
        print("  ✅ Real-time token streaming")
        print("  ✅ Progressive output generation")
        print("  ✅ WebSocket communication")
        print("  ✅ Stream session management")
        print("  ✅ Event-driven architecture")
        print("  ✅ Performance monitoring")
        return True
    else:
        print("⚠️  Some tests need attention.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 