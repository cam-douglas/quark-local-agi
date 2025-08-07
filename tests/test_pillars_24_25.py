"""
Tests for Pillars 24-25: Continuous Learning and Meta-Learning
"""

import pytest
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any

from agents.continuous_learning_agent import (
    ContinuousLearningAgent, 
    LearningMode, 
    LearningStrategy,
    LearningExample,
    LearningSession
)
from agents.meta_learning_agent import (
    MetaLearningAgent,
    MetaLearningMode,
    TrainingSignalType,
    TrainingSignal,
    SelfCritique,
    MetaLearningSession
)


class TestPillar24ContinuousLearning:
    """Tests for Pillar 24: Continuous Online & Few-Shot Learning"""
    
    @pytest.fixture
    async def continuous_learning_agent(self):
        """Create a continuous learning agent for testing"""
        agent = ContinuousLearningAgent()
        yield agent
        # Cleanup if needed
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, continuous_learning_agent):
        """Test agent initialization"""
        assert continuous_learning_agent is not None
        assert continuous_learning_agent.name == "continuous_learning"
        assert len(continuous_learning_agent.learning_modes) == 5
        assert len(continuous_learning_agent.learning_strategies) == 5
        assert len(continuous_learning_agent.replay_buffers) == 5
    
    @pytest.mark.asyncio
    async def test_few_shot_learning(self, continuous_learning_agent):
        """Test few-shot learning mode"""
        examples = [
            {
                "input": "What is AI?",
                "output": "AI is artificial intelligence",
                "confidence": 0.9,
                "source": "test"
            },
            {
                "input": "How does ML work?",
                "output": "ML uses algorithms to learn patterns",
                "confidence": 0.85,
                "source": "test"
            }
        ]
        
        message = {
            "learning_mode": LearningMode.FEW_SHOT.value,
            "strategy": LearningStrategy.GRADIENT_DESCENT.value,
            "examples": examples
        }
        
        result = await continuous_learning_agent.process_message(message)
        
        assert result["status"] == "success"
        assert "session_id" in result
        assert "result" in result
        assert result["result"]["mode"] == "few_shot"
        assert result["result"]["examples_processed"] == 2
    
    @pytest.mark.asyncio
    async def test_online_learning(self, continuous_learning_agent):
        """Test online learning mode"""
        examples = [
            {
                "input": "Real-time data point 1",
                "output": "Processed result 1",
                "confidence": 0.8,
                "source": "stream"
            }
        ]
        
        message = {
            "learning_mode": LearningMode.ONLINE.value,
            "strategy": LearningStrategy.ACTIVE_LEARNING.value,
            "examples": examples
        }
        
        result = await continuous_learning_agent.process_message(message)
        
        assert result["status"] == "success"
        assert result["result"]["mode"] == "online"
        assert result["result"]["examples_processed"] == 1
    
    @pytest.mark.asyncio
    async def test_incremental_learning(self, continuous_learning_agent):
        """Test incremental learning mode"""
        examples = [
            {
                "input": "Incremental update 1",
                "output": "Updated result 1",
                "confidence": 0.7,
                "source": "incremental"
            }
        ]
        
        message = {
            "learning_mode": LearningMode.INCREMENTAL.value,
            "strategy": LearningStrategy.TRANSFER_LEARNING.value,
            "examples": examples
        }
        
        result = await continuous_learning_agent.process_message(message)
        
        assert result["status"] == "success"
        assert result["result"]["mode"] == "incremental"
        assert result["result"]["examples_processed"] == 1
    
    @pytest.mark.asyncio
    async def test_replay_learning(self, continuous_learning_agent):
        """Test replay learning mode"""
        examples = [
            {
                "input": "Replay example 1",
                "output": "Replay result 1",
                "confidence": 0.75,
                "source": "replay"
            }
        ]
        
        message = {
            "learning_mode": LearningMode.REPLAY.value,
            "strategy": LearningStrategy.SELF_SUPERVISED.value,
            "examples": examples
        }
        
        result = await continuous_learning_agent.process_message(message)
        
        assert result["status"] == "success"
        assert result["result"]["mode"] == "replay"
        assert result["result"]["examples_processed"] == 1
    
    @pytest.mark.asyncio
    async def test_adaptive_learning(self, continuous_learning_agent):
        """Test adaptive learning mode"""
        examples = [
            {
                "input": "Adaptive example 1",
                "output": "Adaptive result 1",
                "confidence": 0.6,
                "source": "adaptive"
            }
        ]
        
        message = {
            "learning_mode": LearningMode.ADAPTIVE.value,
            "strategy": LearningStrategy.META_LEARNING.value,
            "examples": examples
        }
        
        result = await continuous_learning_agent.process_message(message)
        
        assert result["status"] == "success"
        assert result["result"]["mode"] == "adaptive"
        assert result["result"]["examples_processed"] == 1
    
    @pytest.mark.asyncio
    async def test_learning_stats(self, continuous_learning_agent):
        """Test learning statistics retrieval"""
        stats = await continuous_learning_agent.get_learning_stats()
        
        assert "learning_sessions" in stats
        assert "learning_stats" in stats
        assert "replay_buffers" in stats
        assert "performance_metrics" in stats
    
    @pytest.mark.asyncio
    async def test_recent_sessions(self, continuous_learning_agent):
        """Test recent sessions retrieval"""
        # First, create some sessions
        examples = [{"input": "test", "output": "result", "confidence": 0.8}]
        message = {
            "learning_mode": LearningMode.FEW_SHOT.value,
            "strategy": LearningStrategy.GRADIENT_DESCENT.value,
            "examples": examples
        }
        
        await continuous_learning_agent.process_message(message)
        
        recent_sessions = await continuous_learning_agent.get_recent_sessions(limit=5)
        
        assert isinstance(recent_sessions, list)
        if recent_sessions:
            assert "session_id" in recent_sessions[0]
            assert "mode" in recent_sessions[0]
            assert "examples_count" in recent_sessions[0]
    
    @pytest.mark.asyncio
    async def test_clear_replay_buffer(self, continuous_learning_agent):
        """Test replay buffer clearing"""
        buffer_id = "replay_buffer_few_shot"
        
        result = await continuous_learning_agent.clear_replay_buffer(buffer_id)
        
        assert result["status"] == "success"
        assert "cleared_examples" in result
    
    @pytest.mark.asyncio
    async def test_agent_info(self, continuous_learning_agent):
        """Test agent information retrieval"""
        info = continuous_learning_agent.get_agent_info()
        
        assert info["name"] == "ContinuousLearningAgent"
        assert "description" in info
        assert "capabilities" in info
        assert "status" in info
        assert "learning_modes" in info
        assert "learning_strategies" in info
        assert "stats" in info


class TestPillar25MetaLearning:
    """Tests for Pillar 25: Meta-Learning & Self-Supervision"""
    
    @pytest.fixture
    async def meta_learning_agent(self):
        """Create a meta-learning agent for testing"""
        agent = MetaLearningAgent()
        yield agent
        # Cleanup if needed
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, meta_learning_agent):
        """Test agent initialization"""
        assert meta_learning_agent is not None
        assert meta_learning_agent.name == "meta_learning"
        assert len(meta_learning_agent.meta_learning_modes) == 5
        assert len(meta_learning_agent.signal_generators) == 5
    
    @pytest.mark.asyncio
    async def test_self_supervised_learning(self, meta_learning_agent):
        """Test self-supervised learning mode"""
        message = {
            "learning_mode": MetaLearningMode.SELF_SUPERVISED.value,
            "signal_types": [TrainingSignalType.QA_PAIRS.value],
            "session_config": {}
        }
        
        result = await meta_learning_agent.process_message(message)
        
        assert result["status"] == "success"
        assert "session_id" in result
        assert "result" in result
        assert result["result"]["mode"] == "self_supervised"
        assert "signals_generated" in result["result"]
    
    @pytest.mark.asyncio
    async def test_synthetic_generation(self, meta_learning_agent):
        """Test synthetic generation mode"""
        message = {
            "learning_mode": MetaLearningMode.SYNTHETIC_GENERATION.value,
            "signal_types": [TrainingSignalType.COMPLETION_TASKS.value],
            "session_config": {}
        }
        
        result = await meta_learning_agent.process_message(message)
        
        assert result["status"] == "success"
        assert result["result"]["mode"] == "synthetic_generation"
        assert "signals_generated" in result["result"]
    
    @pytest.mark.asyncio
    async def test_self_critique_learning(self, meta_learning_agent):
        """Test self-critique learning mode"""
        message = {
            "learning_mode": MetaLearningMode.SELF_CRITIQUE.value,
            "signal_types": [TrainingSignalType.PARAPHRASING.value],
            "session_config": {}
        }
        
        result = await meta_learning_agent.process_message(message)
        
        assert result["status"] == "success"
        assert result["result"]["mode"] == "self_critique"
        assert "critiques_generated" in result["result"]
    
    @pytest.mark.asyncio
    async def test_rapid_adaptation(self, meta_learning_agent):
        """Test rapid adaptation mode"""
        message = {
            "learning_mode": MetaLearningMode.RAPID_ADAPTATION.value,
            "signal_types": [TrainingSignalType.SUMMARIZATION.value],
            "session_config": {}
        }
        
        result = await meta_learning_agent.process_message(message)
        
        assert result["status"] == "success"
        assert result["result"]["mode"] == "rapid_adaptation"
        assert "adaptations_made" in result["result"]
    
    @pytest.mark.asyncio
    async def test_knowledge_synthesis(self, meta_learning_agent):
        """Test knowledge synthesis mode"""
        message = {
            "learning_mode": MetaLearningMode.KNOWLEDGE_SYNTHESIS.value,
            "signal_types": [TrainingSignalType.REASONING_CHAINS.value],
            "session_config": {}
        }
        
        result = await meta_learning_agent.process_message(message)
        
        assert result["status"] == "success"
        assert result["result"]["mode"] == "knowledge_synthesis"
        assert "synthesis_operations" in result["result"]
    
    @pytest.mark.asyncio
    async def test_generate_training_signals(self, meta_learning_agent):
        """Test training signal generation"""
        signal_types = [TrainingSignalType.QA_PAIRS.value, TrainingSignalType.COMPLETION_TASKS.value]
        
        result = await meta_learning_agent.generate_training_signals(signal_types, count=5)
        
        assert result["status"] == "success"
        assert result["signals_generated"] == 10  # 2 types * 5 count
        assert result["signal_types"] == signal_types
        assert len(result["signals"]) == 10
    
    @pytest.mark.asyncio
    async def test_perform_self_critique(self, meta_learning_agent):
        """Test self-critique performance"""
        output = "This is a sample output that needs critique"
        context = {"task": "summarization", "domain": "AI"}
        
        result = await meta_learning_agent.perform_self_critique(output, context)
        
        assert result["status"] == "success"
        assert "critique_id" in result
        assert "critique_score" in result
        assert "improvement_suggestions" in result
        assert "confidence" in result
    
    @pytest.mark.asyncio
    async def test_meta_learning_stats(self, meta_learning_agent):
        """Test meta-learning statistics retrieval"""
        stats = await meta_learning_agent.get_meta_learning_stats()
        
        assert "meta_learning_sessions" in stats
        assert "meta_learning_stats" in stats
        assert "signal_generation_stats" in stats
        assert "critique_stats" in stats
        assert "total_signals_generated" in stats
        assert "total_critiques_performed" in stats
    
    @pytest.mark.asyncio
    async def test_recent_sessions(self, meta_learning_agent):
        """Test recent sessions retrieval"""
        # First, create some sessions
        message = {
            "learning_mode": MetaLearningMode.SELF_SUPERVISED.value,
            "signal_types": [TrainingSignalType.QA_PAIRS.value],
            "session_config": {}
        }
        
        await meta_learning_agent.process_message(message)
        
        recent_sessions = await meta_learning_agent.get_recent_sessions(limit=5)
        
        assert isinstance(recent_sessions, list)
        if recent_sessions:
            assert "session_id" in recent_sessions[0]
            assert "mode" in recent_sessions[0]
            assert "signals_generated" in recent_sessions[0]
            assert "critiques_performed" in recent_sessions[0]
            assert "adaptations_made" in recent_sessions[0]
    
    @pytest.mark.asyncio
    async def test_agent_info(self, meta_learning_agent):
        """Test agent information retrieval"""
        info = meta_learning_agent.get_agent_info()
        
        assert info["name"] == "MetaLearningAgent"
        assert "description" in info
        assert "capabilities" in info
        assert "status" in info
        assert "meta_learning_modes" in info
        assert "training_signal_types" in info
        assert "stats" in info


class TestPillars24_25Integration:
    """Integration tests for Pillars 24-25"""
    
    @pytest.fixture
    async def agents(self):
        """Create both agents for integration testing"""
        continuous_agent = ContinuousLearningAgent()
        meta_agent = MetaLearningAgent()
        yield continuous_agent, meta_agent
    
    @pytest.mark.asyncio
    async def test_continuous_and_meta_learning_integration(self, agents):
        """Test integration between continuous learning and meta-learning"""
        continuous_agent, meta_agent = agents
        
        # Test continuous learning with meta-learning feedback
        examples = [
            {
                "input": "What is meta-learning?",
                "output": "Meta-learning is learning to learn",
                "confidence": 0.9,
                "source": "meta_learning"
            }
        ]
        
        # Process with continuous learning
        continuous_message = {
            "learning_mode": LearningMode.ADAPTIVE.value,
            "strategy": LearningStrategy.META_LEARNING.value,
            "examples": examples
        }
        
        continuous_result = await continuous_agent.process_message(continuous_message)
        
        # Process with meta-learning
        meta_message = {
            "learning_mode": MetaLearningMode.SELF_SUPERVISED.value,
            "signal_types": [TrainingSignalType.QA_PAIRS.value],
            "session_config": {}
        }
        
        meta_result = await meta_agent.process_message(meta_message)
        
        # Verify both agents work together
        assert continuous_result["status"] == "success"
        assert meta_result["status"] == "success"
        assert continuous_result["result"]["mode"] == "adaptive"
        assert meta_result["result"]["mode"] == "self_supervised"
    
    @pytest.mark.asyncio
    async def test_learning_workflow(self, agents):
        """Test complete learning workflow"""
        continuous_agent, meta_agent = agents
        
        # Step 1: Generate training signals with meta-learning
        signal_result = await meta_agent.generate_training_signals(
            [TrainingSignalType.QA_PAIRS.value], count=3
        )
        
        # Step 2: Use signals for continuous learning
        examples = []
        for signal in signal_result["signals"]:
            examples.append({
                "input": signal["input_data"],
                "output": signal["target_output"],
                "confidence": signal["confidence"],
                "source": "meta_generated"
            })
        
        continuous_message = {
            "learning_mode": LearningMode.FEW_SHOT.value,
            "strategy": LearningStrategy.GRADIENT_DESCENT.value,
            "examples": examples
        }
        
        continuous_result = await continuous_agent.process_message(continuous_message)
        
        # Step 3: Perform self-critique on results
        critique_result = await meta_agent.perform_self_critique(
            "Sample learning output",
            {"source": "continuous_learning", "session_id": continuous_result["session_id"]}
        )
        
        # Verify workflow
        assert signal_result["status"] == "success"
        assert continuous_result["status"] == "success"
        assert critique_result["status"] == "success"
        assert signal_result["signals_generated"] == 3
        assert continuous_result["result"]["examples_processed"] == 3
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, agents):
        """Test performance monitoring across both agents"""
        continuous_agent, meta_agent = agents
        
        # Get stats from both agents
        continuous_stats = await continuous_agent.get_learning_stats()
        meta_stats = await meta_agent.get_meta_learning_stats()
        
        # Verify stats structure
        assert "learning_sessions" in continuous_stats
        assert "learning_stats" in continuous_stats
        assert "replay_buffers" in continuous_stats
        
        assert "meta_learning_sessions" in meta_stats
        assert "meta_learning_stats" in meta_stats
        assert "signal_generation_stats" in meta_stats
        assert "critique_stats" in meta_stats
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agents):
        """Test error handling in both agents"""
        continuous_agent, meta_agent = agents
        
        # Test invalid learning mode
        invalid_message = {
            "learning_mode": "invalid_mode",
            "examples": []
        }
        
        continuous_result = await continuous_agent.process_message(invalid_message)
        assert continuous_result["status"] == "error"
        
        # Test invalid meta-learning mode
        invalid_meta_message = {
            "learning_mode": "invalid_meta_mode",
            "signal_types": []
        }
        
        meta_result = await meta_agent.process_message(invalid_meta_message)
        assert meta_result["status"] == "error"
    
    @pytest.mark.asyncio
    async def test_agent_capabilities(self, agents):
        """Test agent capabilities and information"""
        continuous_agent, meta_agent = agents
        
        # Test continuous learning capabilities
        continuous_info = continuous_agent.get_agent_info()
        assert "Few-shot learning" in continuous_info["capabilities"]
        assert "Online learning" in continuous_info["capabilities"]
        assert "Incremental learning" in continuous_info["capabilities"]
        assert "Replay learning" in continuous_info["capabilities"]
        assert "Adaptive learning" in continuous_info["capabilities"]
        
        # Test meta-learning capabilities
        meta_info = meta_agent.get_agent_info()
        assert "Self-supervised learning" in meta_info["capabilities"]
        assert "Synthetic generation" in meta_info["capabilities"]
        assert "Self-critique mechanisms" in meta_info["capabilities"]
        assert "Rapid adaptation" in meta_info["capabilities"]
        assert "Knowledge synthesis" in meta_info["capabilities"]


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"]) 