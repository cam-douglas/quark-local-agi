"""
MEMORY SYSTEM TESTS
==================

Unit tests for the memory system components.
"""

import unittest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from agents.memory_agent import MemoryAgent
from core.context_window_manager import ContextWindowManager
from core.memory_eviction import MemoryEvictionManager


class TestMemoryAgent(unittest.TestCase):
    """Test cases for MemoryAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.memory_agent = MemoryAgent(db_path=self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_memory_agent_initialization(self):
        """Test that memory agent initializes correctly."""
        self.assertIsNotNone(self.memory_agent)
        self.assertIsNotNone(self.memory_agent.client)
        
    def test_store_memory(self):
        """Test storing a memory."""
        content = "User likes pizza"
        metadata = {"user_id": "test_user", "timestamp": "2023-01-01"}
        
        result = self.memory_agent.store_memory(content, metadata)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result["success"])
        self.assertIn("memory_id", result)
        
    def test_retrieve_memories(self):
        """Test retrieving memories."""
        # Store a memory first
        content = "User prefers Italian food"
        metadata = {"user_id": "test_user", "category": "preferences"}
        self.memory_agent.store_memory(content, metadata)
        
        # Retrieve memories
        query = "food preferences"
        result = self.memory_agent.retrieve_memories(query, limit=5)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result["success"])
        self.assertIn("memories", result)
        self.assertIsInstance(result["memories"], list)
        
    def test_search_memories(self):
        """Test searching memories."""
        # Store multiple memories
        memories = [
            ("User likes pizza", {"category": "food"}),
            ("User works at tech company", {"category": "work"}),
            ("User enjoys hiking", {"category": "hobbies"})
        ]
        
        for content, metadata in memories:
            self.memory_agent.store_memory(content, metadata)
        
        # Search for specific memories
        result = self.memory_agent.search_memories("pizza", limit=3)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result["success"])
        self.assertIn("memories", result)
        
    def test_delete_memory(self):
        """Test deleting a memory."""
        # Store a memory first
        content = "Temporary memory"
        metadata = {"category": "test"}
        store_result = self.memory_agent.store_memory(content, metadata)
        memory_id = store_result["memory_id"]
        
        # Delete the memory
        result = self.memory_agent.delete_memory(memory_id)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result["success"])
        
    def test_get_memory_stats(self):
        """Test getting memory statistics."""
        # Store some memories
        for i in range(3):
            content = f"Memory {i}"
            metadata = {"category": "test"}
            self.memory_agent.store_memory(content, metadata)
        
        result = self.memory_agent.get_memory_stats()
        
        self.assertIsInstance(result, dict)
        self.assertIn("total_memories", result)
        self.assertIn("categories", result)
        self.assertIn("recent_activity", result)
        
    def test_update_memory(self):
        """Test updating a memory."""
        # Store a memory first
        content = "Original content"
        metadata = {"category": "test"}
        store_result = self.memory_agent.store_memory(content, metadata)
        memory_id = store_result["memory_id"]
        
        # Update the memory
        new_content = "Updated content"
        new_metadata = {"category": "updated"}
        result = self.memory_agent.update_memory(memory_id, new_content, new_metadata)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result["success"])
        
    def test_get_memory_by_id(self):
        """Test getting a specific memory by ID."""
        # Store a memory first
        content = "Specific memory"
        metadata = {"category": "specific"}
        store_result = self.memory_agent.store_memory(content, metadata)
        memory_id = store_result["memory_id"]
        
        # Get the memory
        result = self.memory_agent.get_memory_by_id(memory_id)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result["success"])
        self.assertIn("memory", result)
        self.assertEqual(result["memory"]["content"], content)


class TestContextWindowManager(unittest.TestCase):
    """Test cases for ContextWindowManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.context_manager = ContextWindowManager()
        
    def test_context_manager_initialization(self):
        """Test that context manager initializes correctly."""
        self.assertIsNotNone(self.context_manager)
        self.assertIsInstance(self.context_manager.messages, list)
        self.assertIsInstance(self.context_manager.max_tokens, int)
        self.assertIsInstance(self.context_manager.max_messages, int)
        
    def test_add_message(self):
        """Test adding a message to context."""
        message = {"role": "user", "content": "Hello"}
        
        result = self.context_manager.add_message(message)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result["success"])
        self.assertIn("message_id", result)
        
    def test_get_context(self):
        """Test getting current context."""
        # Add some messages
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        for message in messages:
            self.context_manager.add_message(message)
        
        result = self.context_manager.get_context()
        
        self.assertIsInstance(result, dict)
        self.assertIn("messages", result)
        self.assertIn("token_count", result)
        self.assertIn("message_count", result)
        
    def test_clear_context(self):
        """Test clearing the context."""
        # Add a message first
        message = {"role": "user", "content": "Test"}
        self.context_manager.add_message(message)
        
        # Clear context
        result = self.context_manager.clear_context()
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result["success"])
        
        # Verify context is empty
        context = self.context_manager.get_context()
        self.assertEqual(context["message_count"], 0)
        
    def test_context_overflow(self):
        """Test handling context overflow."""
        # Add many messages to trigger overflow
        for i in range(20):
            message = {"role": "user", "content": f"Message {i}"}
            self.context_manager.add_message(message)
        
        context = self.context_manager.get_context()
        
        # Should not exceed max messages
        self.assertLessEqual(context["message_count"], self.context_manager.max_messages)
        
    def test_token_counting(self):
        """Test token counting functionality."""
        message = {"role": "user", "content": "This is a test message with multiple words"}
        
        self.context_manager.add_message(message)
        context = self.context_manager.get_context()
        
        self.assertIsInstance(context["token_count"], int)
        self.assertGreater(context["token_count"], 0)


class TestMemoryEvictionManager(unittest.TestCase):
    """Test cases for MemoryEvictionManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.eviction_manager = MemoryEvictionManager()
        
    def test_eviction_manager_initialization(self):
        """Test that eviction manager initializes correctly."""
        self.assertIsNotNone(self.eviction_manager)
        self.assertIsNotNone(self.eviction_manager.eviction_policies)
        
    def test_time_based_eviction(self):
        """Test time-based eviction policy."""
        memories = [
            {"id": "1", "timestamp": "2023-01-01", "content": "Old memory"},
            {"id": "2", "timestamp": "2023-12-01", "content": "Recent memory"}
        ]
        
        result = self.eviction_manager.apply_time_based_eviction(memories, days_threshold=30)
        
        self.assertIsInstance(result, list)
        # Should keep recent memories
        
    def test_size_based_eviction(self):
        """Test size-based eviction policy."""
        memories = [
            {"id": "1", "size": 100, "content": "Small memory"},
            {"id": "2", "size": 1000, "content": "Large memory"}
        ]
        
        result = self.eviction_manager.apply_size_based_eviction(memories, max_size=500)
        
        self.assertIsInstance(result, list)
        # Should keep memories under size limit
        
    def test_relevance_based_eviction(self):
        """Test relevance-based eviction policy."""
        memories = [
            {"id": "1", "relevance_score": 0.1, "content": "Low relevance"},
            {"id": "2", "relevance_score": 0.9, "content": "High relevance"}
        ]
        
        result = self.eviction_manager.apply_relevance_based_eviction(memories, threshold=0.5)
        
        self.assertIsInstance(result, list)
        # Should keep high relevance memories
        
    def test_hybrid_eviction(self):
        """Test hybrid eviction combining multiple policies."""
        memories = [
            {"id": "1", "timestamp": "2023-01-01", "size": 100, "relevance_score": 0.1},
            {"id": "2", "timestamp": "2023-12-01", "size": 50, "relevance_score": 0.9}
        ]
        
        result = self.eviction_manager.apply_hybrid_eviction(memories)
        
        self.assertIsInstance(result, list)
        # Should apply multiple eviction criteria
        
    def test_get_eviction_stats(self):
        """Test getting eviction statistics."""
        result = self.eviction_manager.get_eviction_stats()
        
        self.assertIsInstance(result, dict)
        self.assertIn("total_evictions", result)
        self.assertIn("eviction_policies", result)
        self.assertIn("last_eviction", result)


if __name__ == "__main__":
    unittest.main() 