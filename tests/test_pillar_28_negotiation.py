"""
Test Suite for Pillar 28: Multi-Agent Negotiation & Coordination
Tests specialized agent roles, task bargaining, working group formation, 
collaborative problem solving, and distributed decision making
"""

import asyncio
import json
import pytest
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.negotiation_agent import (
    NegotiationAgent, AgentRole, TaskPriority, NegotiationStatus, 
    DecisionType, AgentCapability, TaskSpecification, NegotiationProposal,
    WorkingGroup, CollaborativeDecision
)


class TestPillar28Negotiation:
    """Test suite for Pillar 28: Multi-Agent Negotiation & Coordination"""
    
    @pytest.fixture
    async def negotiation_agent(self):
        """Create a negotiation agent for testing"""
        agent = NegotiationAgent()
        return agent
    
    @pytest.fixture
    def sample_agents(self):
        """Sample agents for testing"""
        return [
            {
                "agent_id": "agent_001",
                "capabilities": [
                    {
                        "name": "text_generation",
                        "description": "Generate text content",
                        "confidence": 0.9,
                        "performance_metrics": {"accuracy": 0.95},
                        "resource_requirements": {"memory": "2GB"}
                    },
                    {
                        "name": "reasoning",
                        "description": "Logical reasoning",
                        "confidence": 0.8,
                        "performance_metrics": {"accuracy": 0.88},
                        "resource_requirements": {"cpu": "4 cores"}
                    }
                ],
                "performance_metrics": {"response_time": 0.5},
                "resource_requirements": {"memory": "4GB", "cpu": "2 cores"}
            },
            {
                "agent_id": "agent_002",
                "capabilities": [
                    {
                        "name": "memory_retrieval",
                        "description": "Retrieve from memory",
                        "confidence": 0.85,
                        "performance_metrics": {"recall": 0.92},
                        "resource_requirements": {"memory": "1GB"}
                    }
                ],
                "performance_metrics": {"response_time": 0.3},
                "resource_requirements": {"memory": "2GB", "cpu": "1 core"}
            },
            {
                "agent_id": "agent_003",
                "capabilities": [
                    {
                        "name": "validation",
                        "description": "Validate results",
                        "confidence": 0.9,
                        "performance_metrics": {"precision": 0.94},
                        "resource_requirements": {"cpu": "2 cores"}
                    }
                ],
                "performance_metrics": {"response_time": 0.7},
                "resource_requirements": {"memory": "1GB", "cpu": "2 cores"}
            }
        ]
    
    @pytest.fixture
    def sample_tasks(self):
        """Sample tasks for testing"""
        return [
            {
                "task_id": "task_001",
                "title": "Generate Report",
                "description": "Generate a comprehensive report",
                "priority": "high",
                "complexity": 0.7,
                "estimated_duration": 120.0,
                "required_capabilities": ["text_generation", "reasoning"],
                "constraints": {"max_time": 300},
                "deadline": datetime.now() + timedelta(hours=2),
                "dependencies": []
            },
            {
                "task_id": "task_002",
                "title": "Memory Analysis",
                "description": "Analyze memory patterns",
                "priority": "medium",
                "complexity": 0.5,
                "estimated_duration": 60.0,
                "required_capabilities": ["memory_retrieval"],
                "constraints": {"accuracy": 0.9},
                "deadline": datetime.now() + timedelta(hours=1),
                "dependencies": []
            }
        ]
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, negotiation_agent, sample_agents):
        """Test agent registration functionality"""
        # Test registering multiple agents
        for agent_data in sample_agents:
            result = await negotiation_agent.process_message({
                "type": "register_agent",
                **agent_data
            })
            
            assert result["status"] == "success"
            assert "capabilities_count" in result
        
        # Verify agents are registered
        result = await negotiation_agent.process_message({
            "type": "get_agent_capabilities"
        })
        
        assert result["status"] == "success"
        assert result["registered_agents"] == len(sample_agents)
        
        # Test getting specific agent capabilities
        result = await negotiation_agent.process_message({
            "type": "get_agent_capabilities",
            "agent_id": "agent_001"
        })
        
        assert result["status"] == "success"
        assert "agent_capabilities" in result
        assert len(result["agent_capabilities"]) == 2  # agent_001 has 2 capabilities
    
    @pytest.mark.asyncio
    async def test_task_proposal(self, negotiation_agent, sample_agents, sample_tasks):
        """Test task proposal functionality"""
        # Register agents first
        for agent_data in sample_agents:
            await negotiation_agent.process_message({
                "type": "register_agent",
                **agent_data
            })
        
        # Test proposing a task
        task_data = sample_tasks[0]
        result = await negotiation_agent.process_message({
            "type": "propose_task",
            "proposer_id": "test_proposer",
            **task_data
        })
        
        assert result["status"] == "success"
        assert "negotiation_id" in result
        assert "proposal" in result
        assert "suitable_agents" in result
        assert len(result["suitable_agents"]) > 0
        
        # Verify negotiation is active
        negotiation_id = result["negotiation_id"]
        result = await negotiation_agent.process_message({
            "type": "get_negotiation_status",
            "negotiation_id": negotiation_id
        })
        
        assert result["status"] == "success"
        assert result["negotiation"]["status"] == "pending"
    
    @pytest.mark.asyncio
    async def test_task_negotiation(self, negotiation_agent, sample_agents, sample_tasks):
        """Test task negotiation process"""
        # Register agents first
        for agent_data in sample_agents:
            await negotiation_agent.process_message({
                "type": "register_agent",
                **agent_data
            })
        
        # Propose a task
        task_data = sample_tasks[0]
        result = await negotiation_agent.process_message({
            "type": "propose_task",
            "proposer_id": "test_proposer",
            **task_data
        })
        
        negotiation_id = result["negotiation_id"]
        participants = result["suitable_agents"]
        
        # Simulate agent responses
        for i, agent_id in enumerate(participants):
            response = {
                "agreement": i < len(participants) - 1,  # All but one agree
                "confidence": 0.8,
                "comments": f"Agent {agent_id} response"
            }
            
            result = await negotiation_agent.process_message({
                "type": "negotiate_task",
                "negotiation_id": negotiation_id,
                "agent_id": agent_id,
                "response": response
            })
            
            if i < len(participants) - 1:
                assert result["status"] == "success"
                assert "remaining_participants" in result
            else:
                # Last response should finalize negotiation
                assert result["status"] == "success"
                assert "result" in result
                assert "status" in result["result"]
    
    @pytest.mark.asyncio
    async def test_working_group_formation(self, negotiation_agent):
        """Test working group formation"""
        # Test forming a working group
        result = await negotiation_agent.process_message({
            "type": "form_working_group",
            "group_id": "wg_001",
            "name": "Test Working Group",
            "purpose": "Collaborative problem solving",
            "members": ["agent_001", "agent_002", "agent_003"],
            "coordinator": "agent_001",
            "expected_completion": datetime.now() + timedelta(hours=1)
        })
        
        assert result["status"] == "success"
        assert "working_group" in result
        assert result["working_group"]["group_id"] == "wg_001"
        assert result["working_group"]["name"] == "Test Working Group"
        assert len(result["working_group"]["members"]) == 3
        assert result["working_group"]["coordinator"] == "agent_001"
        
        # Test getting working group information
        result = await negotiation_agent.process_message({
            "type": "get_working_groups",
            "group_id": "wg_001"
        })
        
        assert result["status"] == "success"
        assert "working_group" in result
        assert result["working_group"]["group_id"] == "wg_001"
    
    @pytest.mark.asyncio
    async def test_collaborative_decision_making(self, negotiation_agent):
        """Test collaborative decision making"""
        # Test consensus decision
        result = await negotiation_agent.process_message({
            "type": "collaborative_decision",
            "decision_id": "dec_001",
            "decision_type": "consensus",
            "participants": ["agent_001", "agent_002", "agent_003"],
            "options": [
                {"id": "option_1", "name": "Option 1", "description": "First option"},
                {"id": "option_2", "name": "Option 2", "description": "Second option"},
                {"id": "option_3", "name": "Option 3", "description": "Third option"}
            ]
        })
        
        assert result["status"] == "success"
        assert "decision" in result
        assert result["decision"]["decision_id"] == "dec_001"
        assert result["decision"]["decision_type"] == "consensus"
        assert len(result["decision"]["participants"]) == 3
        assert "selected_option" in result["decision"]
        assert "confidence" in result["decision"]
        assert "reasoning" in result["decision"]
        
        # Test majority decision
        result = await negotiation_agent.process_message({
            "type": "collaborative_decision",
            "decision_id": "dec_002",
            "decision_type": "majority",
            "participants": ["agent_001", "agent_002"],
            "options": [
                {"id": "option_a", "name": "Option A", "description": "Option A"},
                {"id": "option_b", "name": "Option B", "description": "Option B"}
            ]
        })
        
        assert result["status"] == "success"
        assert result["decision"]["decision_type"] == "majority"
        
        # Test leader decision
        result = await negotiation_agent.process_message({
            "type": "collaborative_decision",
            "decision_id": "dec_003",
            "decision_type": "leader",
            "participants": ["agent_001", "agent_002"],
            "options": [
                {"id": "option_x", "name": "Option X", "description": "Option X"},
                {"id": "option_y", "name": "Option Y", "description": "Option Y"}
            ]
        })
        
        assert result["status"] == "success"
        assert result["decision"]["decision_type"] == "leader"
    
    @pytest.mark.asyncio
    async def test_agent_role_assignment(self, negotiation_agent, sample_agents, sample_tasks):
        """Test agent role assignment"""
        # Register agents
        for agent_data in sample_agents:
            await negotiation_agent.process_message({
                "type": "register_agent",
                **agent_data
            })
        
        # Test role assignment in task proposal
        task_data = sample_tasks[0]
        result = await negotiation_agent.process_message({
            "type": "propose_task",
            "proposer_id": "test_proposer",
            **task_data
        })
        
        assert result["status"] == "success"
        proposal = result["proposal"]
        assert "proposed_roles" in proposal
        
        # Verify roles are assigned
        roles = proposal["proposed_roles"]
        assert len(roles) > 0
        
        # Check that coordinator is assigned
        coordinators = [agent for agent, role in roles.items() if role == "coordinator"]
        assert len(coordinators) > 0
    
    @pytest.mark.asyncio
    async def test_negotiation_statistics(self, negotiation_agent):
        """Test negotiation statistics tracking"""
        # Get initial stats
        stats = await negotiation_agent.get_negotiation_stats()
        
        assert "stats" in stats
        assert "active_negotiations" in stats
        assert "working_groups" in stats
        assert "collaborative_decisions" in stats
        assert "registered_agents" in stats
        
        # Verify stats structure
        negotiation_stats = stats["stats"]
        assert "total_negotiations" in negotiation_stats
        assert "successful_negotiations" in negotiation_stats
        assert "average_negotiation_time" in negotiation_stats
        assert "working_groups_formed" in negotiation_stats
        assert "collaborative_decisions" in negotiation_stats
    
    @pytest.mark.asyncio
    async def test_error_handling(self, negotiation_agent):
        """Test error handling in negotiation agent"""
        # Test unknown message type
        result = await negotiation_agent.process_message({
            "type": "unknown_message_type"
        })
        
        assert result["status"] == "error"
        assert "Unknown message type" in result["message"]
        
        # Test missing required fields
        result = await negotiation_agent.process_message({
            "type": "collaborative_decision"
            # Missing required fields
        })
        
        assert result["status"] == "error"
        assert "Participants and options are required" in result["message"]
        
        # Test non-existent negotiation
        result = await negotiation_agent.process_message({
            "type": "negotiate_task",
            "negotiation_id": "non_existent_id",
            "agent_id": "test_agent",
            "response": {"agreement": True}
        })
        
        assert result["status"] == "error"
        assert "not found" in result["message"]
    
    @pytest.mark.asyncio
    async def test_agent_capabilities_matching(self, negotiation_agent, sample_agents, sample_tasks):
        """Test agent capabilities matching for tasks"""
        # Register agents
        for agent_data in sample_agents:
            await negotiation_agent.process_message({
                "type": "register_agent",
                **agent_data
            })
        
        # Test task that requires specific capabilities
        task_with_requirements = {
            "task_id": "task_003",
            "title": "Specialized Task",
            "description": "Task requiring specific capabilities",
            "priority": "high",
            "complexity": 0.8,
            "estimated_duration": 180.0,
            "required_capabilities": ["text_generation", "validation"],
            "constraints": {"accuracy": 0.95},
            "deadline": datetime.now() + timedelta(hours=3),
            "dependencies": []
        }
        
        result = await negotiation_agent.process_message({
            "type": "propose_task",
            "proposer_id": "test_proposer",
            **task_with_requirements
        })
        
        assert result["status"] == "success"
        assert "suitable_agents" in result
        
        # Verify that only agents with required capabilities are selected
        suitable_agents = result["suitable_agents"]
        assert len(suitable_agents) > 0
        
        # Check that selected agents have the required capabilities
        for agent_id in suitable_agents:
            agent_capabilities = await negotiation_agent._find_suitable_agents(
                TaskSpecification(**task_with_requirements)
            )
            assert agent_id in agent_capabilities
    
    @pytest.mark.asyncio
    async def test_working_group_communication_channels(self, negotiation_agent):
        """Test working group communication channel setup"""
        # Form a working group
        result = await negotiation_agent.process_message({
            "type": "form_working_group",
            "group_id": "wg_002",
            "name": "Communication Test Group",
            "purpose": "Testing communication channels",
            "members": ["agent_001", "agent_002"],
            "coordinator": "agent_001"
        })
        
        assert result["status"] == "success"
        working_group = result["working_group"]
        
        # Verify communication channels are set up
        assert "communication_channels" in working_group
        channels = working_group["communication_channels"]
        assert len(channels) > 0
        
        # Check for expected channel types
        channel_names = [ch.split("_")[-1] for ch in channels]
        assert "general" in channel_names
        assert "coordinator" in channel_names
        assert "status" in channel_names
        
        # Check for member-specific channels
        for member in working_group["members"]:
            member_channels = [ch for ch in channels if member in ch]
            assert len(member_channels) > 0
    
    @pytest.mark.asyncio
    async def test_decision_types_comparison(self, negotiation_agent):
        """Test different decision types and their outcomes"""
        participants = ["agent_001", "agent_002", "agent_003"]
        options = [
            {"id": "opt_1", "name": "Option 1", "value": 10},
            {"id": "opt_2", "name": "Option 2", "value": 20},
            {"id": "opt_3", "name": "Option 3", "value": 30}
        ]
        
        decision_types = ["consensus", "majority", "leader", "weighted", "autonomous"]
        
        for decision_type in decision_types:
            result = await negotiation_agent.process_message({
                "type": "collaborative_decision",
                "decision_id": f"dec_{decision_type}",
                "decision_type": decision_type,
                "participants": participants,
                "options": options
            })
            
            assert result["status"] == "success"
            assert result["decision"]["decision_type"] == decision_type
            assert "selected_option" in result["decision"]
            assert "confidence" in result["decision"]
            assert "reasoning" in result["decision"]
    
    @pytest.mark.asyncio
    async def test_agent_info_and_capabilities(self, negotiation_agent):
        """Test agent information and capabilities retrieval"""
        # Test agent info
        agent_info = negotiation_agent.get_agent_info()
        
        assert agent_info["agent_name"] == "NegotiationAgent"
        assert "multi_agent_coordination" in agent_info["capabilities"]
        assert "task_negotiation" in agent_info["capabilities"]
        assert "working_group_formation" in agent_info["capabilities"]
        assert "collaborative_problem_solving" in agent_info["capabilities"]
        assert "distributed_decision_making" in agent_info["capabilities"]
        assert agent_info["status"] == "active"
        assert agent_info["version"] == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_comprehensive_workflow(self, negotiation_agent, sample_agents, sample_tasks):
        """Test complete negotiation workflow from registration to decision"""
        # Step 1: Register agents
        for agent_data in sample_agents:
            result = await negotiation_agent.process_message({
                "type": "register_agent",
                **agent_data
            })
            assert result["status"] == "success"
        
        # Step 2: Propose a task
        task_data = sample_tasks[0]
        result = await negotiation_agent.process_message({
            "type": "propose_task",
            "proposer_id": "workflow_test",
            **task_data
        })
        assert result["status"] == "success"
        negotiation_id = result["negotiation_id"]
        participants = result["suitable_agents"]
        
        # Step 3: Negotiate the task
        for agent_id in participants:
            response = {
                "agreement": True,
                "confidence": 0.85,
                "comments": f"Agent {agent_id} agrees to participate"
            }
            
            result = await negotiation_agent.process_message({
                "type": "negotiate_task",
                "negotiation_id": negotiation_id,
                "agent_id": agent_id,
                "response": response
            })
            assert result["status"] == "success"
        
        # Step 4: Form working group after successful negotiation
        result = await negotiation_agent.process_message({
            "type": "form_working_group",
            "group_id": "workflow_group",
            "name": "Workflow Test Group",
            "purpose": "Complete workflow testing",
            "members": participants,
            "coordinator": participants[0] if participants else None
        })
        assert result["status"] == "success"
        
        # Step 5: Make collaborative decision
        result = await negotiation_agent.process_message({
            "type": "collaborative_decision",
            "decision_id": "workflow_decision",
            "decision_type": "consensus",
            "participants": participants,
            "options": [
                {"id": "approach_1", "name": "Approach 1", "description": "First approach"},
                {"id": "approach_2", "name": "Approach 2", "description": "Second approach"}
            ]
        })
        assert result["status"] == "success"
        
        # Step 6: Verify final state
        stats = await negotiation_agent.get_negotiation_stats()
        assert stats["active_negotiations"] >= 0
        assert stats["working_groups"] >= 1
        assert stats["collaborative_decisions"] >= 1
        assert stats["registered_agents"] == len(sample_agents)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 