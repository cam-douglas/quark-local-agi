"""
Test Suite for Pillar 29: Advanced Tool Discovery & Integration
Tests automatic tool discovery, evaluation, integration, and capability expansion
"""

import asyncio
import json
import pytest
import pytest_asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.tool_discovery_agent import (
    ToolDiscoveryAgent, ToolCategory, ToolStatus, ToolPriority,
    ToolSpecification, ToolEvaluation, ToolIntegration
)

class TestPillar29ToolDiscovery:
    """Test suite for Pillar 29: Advanced Tool Discovery & Integration"""
    
    @pytest_asyncio.fixture
    async def tool_discovery_agent(self):
        """Create a tool discovery agent for testing"""
        agent = ToolDiscoveryAgent()
        return agent
    
    @pytest.fixture
    def sample_tool_spec(self):
        """Sample tool specification"""
        return ToolSpecification(
            name="test-tool",
            description="A test tool for evaluation",
            category=ToolCategory.UTILITY,
            priority=ToolPriority.MEDIUM,
            dependencies=["requests", "numpy"],
            requirements={"python_version": ">=3.7"},
            capabilities=["data_processing", "automation"],
            integration_method="python_package",
            source="pypi",
            version="1.0.0",
            author="Test Author",
            license="MIT",
            documentation_url="https://test-tool.com/docs",
            repository_url="https://github.com/test/tool",
            last_updated=datetime.now(),
            status=ToolStatus.DISCOVERED
        )
    
    @pytest.fixture
    def sample_tool_evaluation(self):
        """Sample tool evaluation"""
        return ToolEvaluation(
            tool_name="test-tool",
            evaluation_date=datetime.now(),
            functionality_score=0.8,
            performance_score=0.7,
            security_score=0.9,
            compatibility_score=0.8,
            documentation_score=0.7,
            community_score=0.8,
            overall_score=0.78,
            recommendations=["Test the tool before integration"],
            risks=["May have performance issues"],
            integration_effort="low",
            estimated_time="1-2 hours"
        )
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, tool_discovery_agent):
        """Test tool discovery agent initialization"""
        assert tool_discovery_agent.agent_name == "ToolDiscoveryAgent"
        assert "tool_discovery" in tool_discovery_agent.agent_capabilities
        assert "tool_evaluation" in tool_discovery_agent.agent_capabilities
        assert "tool_integration" in tool_discovery_agent.agent_capabilities
        assert len(tool_discovery_agent.discovery_sources) > 0
        assert len(tool_discovery_agent.capability_mapping) > 0
        assert len(tool_discovery_agent.integration_templates) > 0
    
    @pytest.mark.asyncio
    async def test_tool_requirements_extraction(self, tool_discovery_agent):
        """Test extraction of tool requirements from messages"""
        message = "I need a tool for text processing and data analysis with high priority"
        requirements = tool_discovery_agent._extract_tool_requirements(message)
        
        assert "text_processing" in requirements['capabilities']
        assert "data_analysis" in requirements['capabilities']
        assert requirements['priority'] == ToolPriority.HIGH
    
    @pytest.mark.asyncio
    async def test_tool_discovery_process(self, tool_discovery_agent):
        """Test the complete tool discovery process"""
        message = "Find tools for machine learning and data visualization"
        
        with patch.object(tool_discovery_agent, '_discover_tools') as mock_discover, \
             patch.object(tool_discovery_agent, '_evaluate_tools') as mock_evaluate, \
             patch.object(tool_discovery_agent, '_generate_recommendations') as mock_recommend:
            
            mock_discover.return_value = [Mock()]
            mock_evaluate.return_value = [Mock()]
            mock_recommend.return_value = [{"tool_name": "test", "overall_score": 0.8}]
            
            response = await tool_discovery_agent.process_message(message)
            
            assert "Tool Discovery Results" in response
            assert mock_discover.called
            assert mock_evaluate.called
            assert mock_recommend.called
    
    @pytest.mark.asyncio
    async def test_pypi_search(self, tool_discovery_agent):
        """Test PyPI tool search"""
        requirements = {"capabilities": ["text_processing"]}
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "info": {
                    "name": "test-package",
                    "summary": "A test package",
                    "version": "1.0.0",
                    "author": "Test Author",
                    "license": "MIT",
                    "home_page": "https://test.com",
                    "project_urls": {"Repository": "https://github.com/test"},
                    "requires_dist": ["requests"],
                    "requires_python": ">=3.7"
                }
            }
            mock_get.return_value = mock_response
            
            tools = await tool_discovery_agent._search_pypi(requirements)
            
            assert len(tools) > 0
            assert tools[0].name == "test-package"
            assert tools[0].source == "pypi"
    
    @pytest.mark.asyncio
    async def test_github_search(self, tool_discovery_agent):
        """Test GitHub tool search"""
        requirements = {"capabilities": ["automation"]}
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "items": [{
                    "name": "test-repo",
                    "description": "A test repository",
                    "owner": {"login": "testuser"},
                    "html_url": "https://github.com/test/repo",
                    "license": {"name": "MIT"}
                }]
            }
            mock_get.return_value = mock_response
            
            tools = await tool_discovery_agent._search_github(requirements)
            
            assert len(tools) > 0
            assert tools[0].name == "test-repo"
            assert tools[0].source == "github"
    
    @pytest.mark.asyncio
    async def test_huggingface_search(self, tool_discovery_agent):
        """Test Hugging Face tool search"""
        requirements = {"capabilities": ["text_generation"]}
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = [{
                "id": "test-model",
                "description": "A test model",
                "author": {"name": "Test Author"},
                "license": "MIT"
            }]
            mock_get.return_value = mock_response
            
            tools = await tool_discovery_agent._search_huggingface(requirements)
            
            assert len(tools) > 0
            assert tools[0].name == "test-model"
            assert tools[0].source == "huggingface"
    
    @pytest.mark.asyncio
    async def test_tool_evaluation(self, tool_discovery_agent, sample_tool_spec):
        """Test tool evaluation process"""
        evaluation = await tool_discovery_agent._evaluate_single_tool(sample_tool_spec)
        
        assert evaluation.tool_name == sample_tool_spec.name
        assert evaluation.overall_score >= 0.0
        assert evaluation.overall_score <= 1.0
        assert len(evaluation.recommendations) >= 0
        assert len(evaluation.risks) >= 0
        assert evaluation.integration_effort in ["low", "medium", "high", "unknown"]
    
    @pytest.mark.asyncio
    async def test_functionality_evaluation(self, tool_discovery_agent, sample_tool_spec):
        """Test functionality evaluation"""
        score = await tool_discovery_agent._evaluate_functionality(sample_tool_spec)
        
        assert score >= 0.0
        assert score <= 1.0
    
    @pytest.mark.asyncio
    async def test_performance_evaluation(self, tool_discovery_agent, sample_tool_spec):
        """Test performance evaluation"""
        score = await tool_discovery_agent._evaluate_performance(sample_tool_spec)
        
        assert score >= 0.0
        assert score <= 1.0
    
    @pytest.mark.asyncio
    async def test_security_evaluation(self, tool_discovery_agent, sample_tool_spec):
        """Test security evaluation"""
        score = await tool_discovery_agent._evaluate_security(sample_tool_spec)
        
        assert score >= 0.0
        assert score <= 1.0
    
    @pytest.mark.asyncio
    async def test_compatibility_evaluation(self, tool_discovery_agent, sample_tool_spec):
        """Test compatibility evaluation"""
        score = await tool_discovery_agent._evaluate_compatibility(sample_tool_spec)
        
        assert score >= 0.0
        assert score <= 1.0
    
    @pytest.mark.asyncio
    async def test_documentation_evaluation(self, tool_discovery_agent, sample_tool_spec):
        """Test documentation evaluation"""
        score = await tool_discovery_agent._evaluate_documentation(sample_tool_spec)
        
        assert score >= 0.0
        assert score <= 1.0
    
    @pytest.mark.asyncio
    async def test_community_evaluation(self, tool_discovery_agent, sample_tool_spec):
        """Test community evaluation"""
        score = await tool_discovery_agent._evaluate_community(sample_tool_spec)
        
        assert score >= 0.0
        assert score <= 1.0
    
    @pytest.mark.asyncio
    async def test_recommendation_generation(self, tool_discovery_agent, sample_tool_evaluation):
        """Test recommendation generation"""
        recommendations = tool_discovery_agent._generate_recommendations([sample_tool_evaluation])
        
        assert len(recommendations) > 0
        assert "tool_name" in recommendations[0]
        assert "overall_score" in recommendations[0]
        assert "integration_effort" in recommendations[0]
    
    @pytest.mark.asyncio
    async def test_risk_identification(self, tool_discovery_agent, sample_tool_spec):
        """Test risk identification"""
        scores = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # Mixed scores
        risks = tool_discovery_agent._identify_risks_for_tool(sample_tool_spec, scores)
        
        assert len(risks) > 0
        assert all(isinstance(risk, str) for risk in risks)
    
    @pytest.mark.asyncio
    async def test_integration_effort_estimation(self, tool_discovery_agent, sample_tool_spec):
        """Test integration effort estimation"""
        effort = tool_discovery_agent._estimate_integration_effort(sample_tool_spec)
        
        assert effort in ["low", "medium", "high", "unknown"]
    
    @pytest.mark.asyncio
    async def test_integration_time_estimation(self, tool_discovery_agent, sample_tool_spec):
        """Test integration time estimation"""
        time_estimate = tool_discovery_agent._estimate_integration_time(sample_tool_spec)
        
        assert "hours" in time_estimate or "days" in time_estimate or time_estimate == "unknown"
    
    @pytest.mark.asyncio
    async def test_python_package_integration(self, tool_discovery_agent, sample_tool_spec):
        """Test Python package integration"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            
            result = await tool_discovery_agent._integrate_python_package(sample_tool_spec)
            
            assert "success" in result
            assert mock_run.called
    
    @pytest.mark.asyncio
    async def test_api_service_integration(self, tool_discovery_agent, sample_tool_spec):
        """Test API service integration"""
        result = await tool_discovery_agent._integrate_api_service(sample_tool_spec)
        
        assert result["success"] == True
        assert "configuration" in result
        assert "endpoints" in result
        assert "api_keys" in result
    
    @pytest.mark.asyncio
    async def test_docker_container_integration(self, tool_discovery_agent, sample_tool_spec):
        """Test Docker container integration"""
        result = await tool_discovery_agent._integrate_docker_container(sample_tool_spec)
        
        assert result["success"] == True
        assert "configuration" in result
        assert "endpoints" in result
    
    @pytest.mark.asyncio
    async def test_webhook_integration(self, tool_discovery_agent, sample_tool_spec):
        """Test webhook integration"""
        result = await tool_discovery_agent._integrate_webhook(sample_tool_spec)
        
        assert result["success"] == True
        assert "configuration" in result
        assert "endpoints" in result
        assert "environment_variables" in result
    
    @pytest.mark.asyncio
    async def test_custom_module_integration(self, tool_discovery_agent, sample_tool_spec):
        """Test custom module integration"""
        result = await tool_discovery_agent._integrate_custom_module(sample_tool_spec)
        
        assert result["success"] == True
        assert "configuration" in result
    
    @pytest.mark.asyncio
    async def test_tool_integration_process(self, tool_discovery_agent, sample_tool_spec):
        """Test complete tool integration process"""
        # Add tool to discovered and evaluated registries
        tool_discovery_agent.discovered_tools[sample_tool_spec.name] = sample_tool_spec
        tool_discovery_agent.evaluated_tools[sample_tool_spec.name] = ToolEvaluation(
            tool_name=sample_tool_spec.name,
            evaluation_date=datetime.now(),
            functionality_score=0.8,
            performance_score=0.8,
            security_score=0.8,
            compatibility_score=0.8,
            documentation_score=0.8,
            community_score=0.8,
            overall_score=0.8,
            recommendations=[],
            risks=[],
            integration_effort="low",
            estimated_time="1-2 hours"
        )
        
        with patch.object(tool_discovery_agent, '_perform_integration') as mock_integrate:
            mock_integrate.return_value = {"success": True, "message": "Integration successful"}
            
            result = await tool_discovery_agent.integrate_tool(sample_tool_spec.name)
            
            assert "Successfully integrated" in result
            assert mock_integrate.called
    
    @pytest.mark.asyncio
    async def test_tool_integration_rejection(self, tool_discovery_agent, sample_tool_spec):
        """Test tool integration rejection for low scores"""
        # Add tool to discovered and evaluated registries with low score
        tool_discovery_agent.discovered_tools[sample_tool_spec.name] = sample_tool_spec
        tool_discovery_agent.evaluated_tools[sample_tool_spec.name] = ToolEvaluation(
            tool_name=sample_tool_spec.name,
            evaluation_date=datetime.now(),
            functionality_score=0.3,
            performance_score=0.3,
            security_score=0.3,
            compatibility_score=0.3,
            documentation_score=0.3,
            community_score=0.3,
            overall_score=0.3,
            recommendations=[],
            risks=[],
            integration_effort="low",
            estimated_time="1-2 hours"
        )
        
        result = await tool_discovery_agent.integrate_tool(sample_tool_spec.name)
        
        assert "low evaluation score" in result
        assert "not recommended" in result
    
    @pytest.mark.asyncio
    async def test_discovery_statistics(self, tool_discovery_agent):
        """Test discovery statistics"""
        stats = tool_discovery_agent.get_discovery_stats()
        
        assert "total_discovered" in stats
        assert "total_evaluated" in stats
        assert "total_integrated" in stats
        assert "success_rate" in stats
        assert "average_evaluation_time" in stats
        assert "average_integration_time" in stats
    
    @pytest.mark.asyncio
    async def test_tool_information_retrieval(self, tool_discovery_agent, sample_tool_spec):
        """Test tool information retrieval"""
        # Add tool to registries
        tool_discovery_agent.discovered_tools[sample_tool_spec.name] = sample_tool_spec
        tool_discovery_agent.evaluated_tools[sample_tool_spec.name] = Mock()
        tool_discovery_agent.integrated_tools[sample_tool_spec.name] = Mock()
        
        info = tool_discovery_agent.get_tool_info(sample_tool_spec.name)
        
        assert info is not None
        assert "specification" in info
        assert "evaluation" in info
        assert "integration" in info
    
    @pytest.mark.asyncio
    async def test_capability_mapping(self, tool_discovery_agent):
        """Test capability mapping functionality"""
        current_capabilities = tool_discovery_agent._get_current_capabilities()
        missing_capabilities = tool_discovery_agent._identify_missing_capabilities(current_capabilities)
        
        assert isinstance(current_capabilities, list)
        assert isinstance(missing_capabilities, list)
        assert len(missing_capabilities) <= 3  # Should return top 3 missing capabilities
    
    @pytest.mark.asyncio
    async def test_registry_persistence(self, tool_discovery_agent, sample_tool_spec):
        """Test registry persistence functionality"""
        # Add tool to registry
        tool_discovery_agent.discovered_tools[sample_tool_spec.name] = sample_tool_spec
        
        # Test save functionality
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            tool_discovery_agent._save_discovered_tools()
            
            assert mock_open.called
            assert mock_file.write.called
    
    @pytest.mark.asyncio
    async def test_error_handling(self, tool_discovery_agent):
        """Test error handling in tool discovery"""
        # Test with invalid message
        response = await tool_discovery_agent.process_message("")
        
        assert "Tool Discovery Agent" in response
        assert "help you discover" in response
    
    @pytest.mark.asyncio
    async def test_comprehensive_workflow(self, tool_discovery_agent):
        """Test comprehensive tool discovery workflow"""
        message = "I need tools for machine learning and data visualization with high priority"
        
        with patch.object(tool_discovery_agent, '_discover_tools') as mock_discover, \
             patch.object(tool_discovery_agent, '_evaluate_tools') as mock_evaluate, \
             patch.object(tool_discovery_agent, '_generate_recommendations') as mock_recommend:
            
            # Mock discovered tools
            mock_tool = Mock()
            mock_tool.name = "test-ml-tool"
            mock_discover.return_value = [mock_tool]
            
            # Mock evaluations
            mock_eval = Mock()
            mock_eval.overall_score = 0.85
            mock_evaluate.return_value = [mock_eval]
            
            # Mock recommendations
            mock_recommend.return_value = [{
                "tool_name": "test-ml-tool",
                "overall_score": 0.85,
                "integration_effort": "medium",
                "estimated_time": "4-8 hours"
            }]
            
            response = await tool_discovery_agent.process_message(message)
            
            # Verify the workflow
            assert "Tool Discovery Results" in response
            assert "Discovered Tools" in response
            assert "Evaluated Tools" in response
            assert "Top Recommendations" in response
            assert mock_discover.called
            assert mock_evaluate.called
            assert mock_recommend.called

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 