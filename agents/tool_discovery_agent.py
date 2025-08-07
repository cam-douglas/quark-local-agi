"""
Advanced Tool Discovery & Integration Agent
Pillar 29: Automatic tool discovery, evaluation, integration, and capability expansion
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from pathlib import Path
from collections import defaultdict, deque
import requests
import subprocess
import importlib
import inspect
import sys
import os
from .base import Agent as BaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolCategory(Enum):
    """Tool categories for classification"""
    UTILITY = "utility"
    AI_MODEL = "ai_model"
    DATA_PROCESSING = "data_processing"
    COMMUNICATION = "communication"
    VISUALIZATION = "visualization"
    AUTOMATION = "automation"
    SECURITY = "security"
    MONITORING = "monitoring"
    INTEGRATION = "integration"
    CUSTOM = "custom"

class ToolStatus(Enum):
    """Tool integration status"""
    DISCOVERED = "discovered"
    EVALUATING = "evaluating"
    APPROVED = "approved"
    INTEGRATED = "integrated"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"

class ToolPriority(Enum):
    """Tool priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"

@dataclass
class ToolSpecification:
    """Tool specification and metadata"""
    name: str
    description: str
    category: ToolCategory
    priority: ToolPriority
    dependencies: List[str]
    requirements: Dict[str, Any]
    capabilities: List[str]
    integration_method: str
    source: str
    version: str
    author: str
    license: str
    documentation_url: str
    repository_url: str
    last_updated: datetime
    status: ToolStatus
    evaluation_score: float = 0.0
    integration_complexity: str = "unknown"
    resource_requirements: Dict[str, Any] = None
    safety_assessment: Dict[str, Any] = None

@dataclass
class ToolEvaluation:
    """Tool evaluation results"""
    tool_name: str
    evaluation_date: datetime
    functionality_score: float
    performance_score: float
    security_score: float
    compatibility_score: float
    documentation_score: float
    community_score: float
    overall_score: float
    recommendations: List[str]
    risks: List[str]
    integration_effort: str
    estimated_time: str

@dataclass
class ToolIntegration:
    """Tool integration details"""
    tool_name: str
    integration_date: datetime
    integration_method: str
    configuration: Dict[str, Any]
    endpoints: List[str]
    api_keys: List[str]
    environment_variables: List[str]
    dependencies_installed: List[str]
    test_results: Dict[str, Any]
    monitoring_config: Dict[str, Any]

class ToolDiscoveryAgent(BaseAgent):
    """Advanced Tool Discovery & Integration Agent"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("tool_discovery_model")
        self.agent_name = "ToolDiscoveryAgent"
        self.agent_capabilities = [
            "tool_discovery", "tool_evaluation", "tool_integration", "capability_expansion",
            "dependency_management", "safety_assessment", "performance_monitoring"
        ]
        
        # Initialize tool discovery systems
        self._initialize_tool_discovery_systems()
        
        # Tool registries
        self.discovered_tools: Dict[str, ToolSpecification] = {}
        self.evaluated_tools: Dict[str, ToolEvaluation] = {}
        self.integrated_tools: Dict[str, ToolIntegration] = {}
        self.tool_capabilities: Dict[str, List[str]] = {}
        
        # Discovery sources
        self.discovery_sources = {
            'pypi': 'https://pypi.org/pypi/{package}/json',
            'github': 'https://api.github.com/search/repositories?q={query}',
            'huggingface': 'https://huggingface.co/api/models?search={query}',
            'custom_registry': 'https://quark-tools-registry.com/api/tools'
        }
        
        # Tool categories and capabilities mapping
        self.capability_mapping = {
            'text_generation': ['transformers', 'openai', 'anthropic'],
            'image_processing': ['pillow', 'opencv', 'torchvision'],
            'data_analysis': ['pandas', 'numpy', 'scipy'],
            'machine_learning': ['scikit-learn', 'tensorflow', 'pytorch'],
            'web_scraping': ['requests', 'beautifulsoup4', 'selenium'],
            'automation': ['selenium', 'pyautogui', 'schedule'],
            'monitoring': ['prometheus', 'grafana', 'datadog'],
            'security': ['cryptography', 'bcrypt', 'jwt']
        }
        
        # Integration templates
        self.integration_templates = {
            'python_package': self._integrate_python_package,
            'api_service': self._integrate_api_service,
            'docker_container': self._integrate_docker_container,
            'webhook': self._integrate_webhook,
            'custom_module': self._integrate_custom_module
        }
        
        # Statistics
        self.discovery_stats = {
            'total_discovered': 0,
            'total_evaluated': 0,
            'total_integrated': 0,
            'success_rate': 0.0,
            'average_evaluation_time': 0.0,
            'average_integration_time': 0.0
        }
        
        logger.info(f"Initialized {self.agent_name} with tool discovery capabilities")
    
    def load_model(self):
        """Load the tool discovery model"""
        # For tool discovery, we don't need a specific model
        # The agent works with external APIs and registries
        return None
    
    def generate(self, prompt: str, **kwargs):
        """Generate tool discovery response"""
        # This would be implemented to handle tool discovery requests
        return f"Tool discovery response for: {prompt}"
    
    def _initialize_tool_discovery_systems(self):
        """Initialize tool discovery systems"""
        # Create necessary directories
        os.makedirs('tools', exist_ok=True)
        os.makedirs('tools/discovered', exist_ok=True)
        os.makedirs('tools/evaluated', exist_ok=True)
        os.makedirs('tools/integrated', exist_ok=True)
        os.makedirs('tools/templates', exist_ok=True)
        
        # Load existing tool registries
        self._load_tool_registries()
        
        # Initialize discovery workers
        self._start_discovery_workers()
    
    def _load_tool_registries(self):
        """Load existing tool registries from disk"""
        registry_files = [
            'tools/discovered/registry.json',
            'tools/evaluated/registry.json',
            'tools/integrated/registry.json'
        ]
        
        for registry_file in registry_files:
            if os.path.exists(registry_file):
                try:
                    with open(registry_file, 'r') as f:
                        data = json.load(f)
                        if 'discovered_tools' in data:
                            self.discovered_tools.update(data['discovered_tools'])
                        if 'evaluated_tools' in data:
                            self.evaluated_tools.update(data['evaluated_tools'])
                        if 'integrated_tools' in data:
                            self.integrated_tools.update(data['integrated_tools'])
                except Exception as e:
                    logger.error(f"Failed to load registry {registry_file}: {e}")
    
    def _start_discovery_workers(self):
        """Start background discovery workers"""
        def discovery_worker():
            while True:
                try:
                    self._continuous_discovery()
                    time.sleep(3600)  # Run every hour
                except Exception as e:
                    logger.error(f"Discovery worker error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        import threading
        discovery_thread = threading.Thread(target=discovery_worker, daemon=True)
        discovery_thread.start()
        logger.info("Started tool discovery worker")
    
    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process incoming messages for tool discovery requests"""
        try:
            # Parse the message for tool discovery intent
            if any(keyword in message.lower() for keyword in ['tool', 'discover', 'integrate', 'capability']):
                return await self._handle_tool_discovery_request(message, context)
            else:
                return "I'm the Tool Discovery Agent. I can help you discover, evaluate, and integrate new tools and capabilities. What would you like to do?"
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"Error processing tool discovery request: {str(e)}"
    
    async def _handle_tool_discovery_request(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Handle tool discovery requests"""
        try:
            # Extract tool requirements from message
            requirements = self._extract_tool_requirements(message)
            
            # Discover relevant tools
            discovered_tools = await self._discover_tools(requirements)
            
            # Evaluate discovered tools
            evaluated_tools = await self._evaluate_tools(discovered_tools)
            
            # Recommend best tools
            recommendations = self._generate_recommendations(evaluated_tools)
            
            # Format response
            response = self._format_discovery_response(discovered_tools, evaluated_tools, recommendations)
            
            return response
        
        except Exception as e:
            logger.error(f"Error handling tool discovery request: {e}")
            return f"Error in tool discovery: {str(e)}"
    
    def _extract_tool_requirements(self, message: str) -> Dict[str, Any]:
        """Extract tool requirements from user message"""
        requirements = {
            'capabilities': [],
            'category': None,
            'priority': ToolPriority.MEDIUM,
            'constraints': []
        }
        
        # Extract capabilities
        capability_keywords = {
            'text': 'text_processing',
            'image': 'image_processing',
            'data': 'data_analysis',
            'ml': 'machine_learning',
            'web': 'web_scraping',
            'auto': 'automation',
            'monitor': 'monitoring',
            'security': 'security'
        }
        
        for keyword, capability in capability_keywords.items():
            if keyword in message.lower():
                requirements['capabilities'].append(capability)
        
        # Extract category
        for category in ToolCategory:
            if category.value in message.lower():
                requirements['category'] = category
                break
        
        # Extract priority
        if 'critical' in message.lower() or 'urgent' in message.lower():
            requirements['priority'] = ToolPriority.CRITICAL
        elif 'high' in message.lower():
            requirements['priority'] = ToolPriority.HIGH
        elif 'low' in message.lower():
            requirements['priority'] = ToolPriority.LOW
        
        return requirements
    
    async def _discover_tools(self, requirements: Dict[str, Any]) -> List[ToolSpecification]:
        """Discover tools based on requirements"""
        discovered_tools = []
        
        try:
            # Search PyPI for relevant packages
            pypi_tools = await self._search_pypi(requirements)
            discovered_tools.extend(pypi_tools)
            
            # Search GitHub for relevant repositories
            github_tools = await self._search_github(requirements)
            discovered_tools.extend(github_tools)
            
            # Search Hugging Face for models
            hf_tools = await self._search_huggingface(requirements)
            discovered_tools.extend(hf_tools)
            
            # Search custom registry
            custom_tools = await self._search_custom_registry(requirements)
            discovered_tools.extend(custom_tools)
            
            # Update discovered tools registry
            for tool in discovered_tools:
                self.discovered_tools[tool.name] = tool
            
            # Save to disk
            self._save_discovered_tools()
            
            logger.info(f"Discovered {len(discovered_tools)} tools")
            return discovered_tools
        
        except Exception as e:
            logger.error(f"Error discovering tools: {e}")
            return []
    
    async def _search_pypi(self, requirements: Dict[str, Any]) -> List[ToolSpecification]:
        """Search PyPI for relevant packages"""
        tools = []
        
        try:
            # Search for packages based on capabilities
            for capability in requirements.get('capabilities', []):
                search_terms = self.capability_mapping.get(capability, [capability])
                
                for term in search_terms:
                    url = f"https://pypi.org/pypi/{term}/json"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        tool = ToolSpecification(
                            name=data['info']['name'],
                            description=data['info']['summary'],
                            category=ToolCategory.UTILITY,
                            priority=requirements.get('priority', ToolPriority.MEDIUM),
                            dependencies=data['info'].get('requires_dist', []),
                            requirements={'python_version': data['info'].get('requires_python', '>=3.7')},
                            capabilities=[capability],
                            integration_method='python_package',
                            source='pypi',
                            version=data['info']['version'],
                            author=data['info'].get('author', 'Unknown'),
                            license=data['info'].get('license', 'Unknown'),
                            documentation_url=data['info'].get('home_page', ''),
                            repository_url=data['info'].get('project_urls', {}).get('Repository', ''),
                            last_updated=datetime.now(),
                            status=ToolStatus.DISCOVERED
                        )
                        
                        tools.append(tool)
        
        except Exception as e:
            logger.error(f"Error searching PyPI: {e}")
        
        return tools
    
    async def _search_github(self, requirements: Dict[str, Any]) -> List[ToolSpecification]:
        """Search GitHub for relevant repositories"""
        tools = []
        
        try:
            # Search for repositories based on capabilities
            for capability in requirements.get('capabilities', []):
                search_terms = self.capability_mapping.get(capability, [capability])
                
                for term in search_terms:
                    url = f"https://api.github.com/search/repositories?q={term}+language:python&sort=stars&order=desc"
                    headers = {'Accept': 'application/vnd.github.v3+json'}
                    
                    if 'GITHUB_TOKEN' in os.environ:
                        headers['Authorization'] = f"token {os.environ['GITHUB_TOKEN']}"
                    
                    response = requests.get(url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        for repo in data.get('items', [])[:5]:  # Top 5 results
                            tool = ToolSpecification(
                                name=repo['name'],
                                description=repo['description'] or 'No description available',
                                category=ToolCategory.CUSTOM,
                                priority=requirements.get('priority', ToolPriority.MEDIUM),
                                dependencies=[],
                                requirements={'python_version': '>=3.7'},
                                capabilities=[capability],
                                integration_method='custom_module',
                                source='github',
                                version='latest',
                                author=repo['owner']['login'],
                                license=repo.get('license', {}).get('name', 'Unknown'),
                                documentation_url=repo['html_url'],
                                repository_url=repo['html_url'],
                                last_updated=datetime.now(),
                                status=ToolStatus.DISCOVERED
                            )
                            
                            tools.append(tool)
        
        except Exception as e:
            logger.error(f"Error searching GitHub: {e}")
        
        return tools
    
    async def _search_huggingface(self, requirements: Dict[str, Any]) -> List[ToolSpecification]:
        """Search Hugging Face for relevant models"""
        tools = []
        
        try:
            # Search for models based on capabilities
            for capability in requirements.get('capabilities', []):
                if capability == 'text_generation':
                    url = "https://huggingface.co/api/models?search=text-generation&sort=downloads&direction=-1"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        for model in data[:5]:  # Top 5 models
                            tool = ToolSpecification(
                                name=model['id'],
                                description=f"Hugging Face model: {model.get('description', 'No description')}",
                                category=ToolCategory.AI_MODEL,
                                priority=requirements.get('priority', ToolPriority.MEDIUM),
                                dependencies=['transformers', 'torch'],
                                requirements={'python_version': '>=3.7'},
                                capabilities=['text_generation'],
                                integration_method='ai_model',
                                source='huggingface',
                                version='latest',
                                author=model.get('author', {}).get('name', 'Unknown'),
                                license=model.get('license', 'Unknown'),
                                documentation_url=f"https://huggingface.co/{model['id']}",
                                repository_url=f"https://huggingface.co/{model['id']}",
                                last_updated=datetime.now(),
                                status=ToolStatus.DISCOVERED
                            )
                            
                            tools.append(tool)
        
        except Exception as e:
            logger.error(f"Error searching Hugging Face: {e}")
        
        return tools
    
    async def _search_custom_registry(self, requirements: Dict[str, Any]) -> List[ToolSpecification]:
        """Search custom tool registry"""
        tools = []
        
        try:
            # This would connect to a custom tool registry
            # For now, return empty list
            pass
        
        except Exception as e:
            logger.error(f"Error searching custom registry: {e}")
        
        return tools
    
    async def _evaluate_tools(self, tools: List[ToolSpecification]) -> List[ToolEvaluation]:
        """Evaluate discovered tools"""
        evaluations = []
        
        for tool in tools:
            try:
                evaluation = await self._evaluate_single_tool(tool)
                evaluations.append(evaluation)
                
                # Update evaluated tools registry
                self.evaluated_tools[tool.name] = evaluation
                
            except Exception as e:
                logger.error(f"Error evaluating tool {tool.name}: {e}")
        
        # Save to disk
        self._save_evaluated_tools()
        
        return evaluations
    
    async def _evaluate_single_tool(self, tool: ToolSpecification) -> ToolEvaluation:
        """Evaluate a single tool"""
        start_time = time.time()
        
        # Evaluate different aspects
        functionality_score = await self._evaluate_functionality(tool)
        performance_score = await self._evaluate_performance(tool)
        security_score = await self._evaluate_security(tool)
        compatibility_score = await self._evaluate_compatibility(tool)
        documentation_score = await self._evaluate_documentation(tool)
        community_score = await self._evaluate_community(tool)
        
        # Calculate overall score
        scores = [functionality_score, performance_score, security_score, 
                 compatibility_score, documentation_score, community_score]
        overall_score = sum(scores) / len(scores)
        
        # Generate recommendations and risks
        recommendations = self._generate_recommendations_for_tool(tool, scores)
        risks = self._identify_risks_for_tool(tool, scores)
        
        evaluation_time = time.time() - start_time
        
        return ToolEvaluation(
            tool_name=tool.name,
            evaluation_date=datetime.now(),
            functionality_score=functionality_score,
            performance_score=performance_score,
            security_score=security_score,
            compatibility_score=compatibility_score,
            documentation_score=documentation_score,
            community_score=community_score,
            overall_score=overall_score,
            recommendations=recommendations,
            risks=risks,
            integration_effort=self._estimate_integration_effort(tool),
            estimated_time=self._estimate_integration_time(tool)
        )
    
    async def _evaluate_functionality(self, tool: ToolSpecification) -> float:
        """Evaluate tool functionality"""
        score = 0.5  # Base score
        
        # Check if tool has clear capabilities
        if tool.capabilities:
            score += 0.2
        
        # Check if tool has dependencies
        if tool.dependencies:
            score += 0.1
        
        # Check if tool has documentation
        if tool.documentation_url:
            score += 0.2
        
        return min(score, 1.0)
    
    async def _evaluate_performance(self, tool: ToolSpecification) -> float:
        """Evaluate tool performance"""
        score = 0.5  # Base score
        
        # Check if tool is from reputable source
        if tool.source in ['pypi', 'github']:
            score += 0.3
        
        # Check if tool has recent updates
        if tool.last_updated:
            days_since_update = (datetime.now() - tool.last_updated).days
            if days_since_update < 365:  # Updated within a year
                score += 0.2
        
        return min(score, 1.0)
    
    async def _evaluate_security(self, tool: ToolSpecification) -> float:
        """Evaluate tool security"""
        score = 0.5  # Base score
        
        # Check if tool has license
        if tool.license and tool.license != 'Unknown':
            score += 0.2
        
        # Check if tool is from trusted source
        if tool.source in ['pypi', 'huggingface']:
            score += 0.3
        
        return min(score, 1.0)
    
    async def _evaluate_compatibility(self, tool: ToolSpecification) -> float:
        """Evaluate tool compatibility"""
        score = 0.5  # Base score
        
        # Check Python version compatibility
        if tool.requirements and 'python_version' in tool.requirements:
            score += 0.3
        
        # Check if dependencies are reasonable
        if tool.dependencies and len(tool.dependencies) < 10:
            score += 0.2
        
        return min(score, 1.0)
    
    async def _evaluate_documentation(self, tool: ToolSpecification) -> float:
        """Evaluate tool documentation"""
        score = 0.5  # Base score
        
        # Check if documentation URL exists
        if tool.documentation_url:
            score += 0.3
        
        # Check if tool has description
        if tool.description and len(tool.description) > 10:
            score += 0.2
        
        return min(score, 1.0)
    
    async def _evaluate_community(self, tool: ToolSpecification) -> float:
        """Evaluate tool community support"""
        score = 0.5  # Base score
        
        # Check if tool has repository
        if tool.repository_url:
            score += 0.3
        
        # Check if tool has author
        if tool.author and tool.author != 'Unknown':
            score += 0.2
        
        return min(score, 1.0)
    
    def _generate_recommendations_for_tool(self, tool: ToolSpecification, scores: List[float]) -> List[str]:
        """Generate recommendations for a tool"""
        recommendations = []
        
        if scores[0] < 0.7:  # Functionality score
            recommendations.append("Consider testing the tool before integration")
        
        if scores[1] < 0.7:  # Performance score
            recommendations.append("Monitor performance after integration")
        
        if scores[2] < 0.7:  # Security score
            recommendations.append("Review security implications before integration")
        
        if scores[3] < 0.7:  # Compatibility score
            recommendations.append("Test compatibility with existing system")
        
        if scores[4] < 0.7:  # Documentation score
            recommendations.append("Consider creating internal documentation")
        
        if scores[5] < 0.7:  # Community score
            recommendations.append("Monitor community activity and updates")
        
        return recommendations
    
    def _identify_risks_for_tool(self, tool: ToolSpecification, scores: List[float]) -> List[str]:
        """Identify risks for a tool"""
        risks = []
        
        if scores[0] < 0.5:  # Functionality score
            risks.append("Tool may not work as expected")
        
        if scores[1] < 0.5:  # Performance score
            risks.append("Tool may have performance issues")
        
        if scores[2] < 0.5:  # Security score
            risks.append("Tool may have security vulnerabilities")
        
        if scores[3] < 0.5:  # Compatibility score
            risks.append("Tool may not be compatible with system")
        
        if scores[4] < 0.5:  # Documentation score
            risks.append("Tool may be difficult to use without documentation")
        
        if scores[5] < 0.5:  # Community score
            risks.append("Tool may not have community support")
        
        return risks
    
    def _estimate_integration_effort(self, tool: ToolSpecification) -> str:
        """Estimate integration effort"""
        if tool.integration_method == 'python_package':
            return 'low'
        elif tool.integration_method == 'api_service':
            return 'medium'
        elif tool.integration_method == 'docker_container':
            return 'high'
        else:
            return 'unknown'
    
    def _estimate_integration_time(self, tool: ToolSpecification) -> str:
        """Estimate integration time"""
        effort = self._estimate_integration_effort(tool)
        
        if effort == 'low':
            return '1-2 hours'
        elif effort == 'medium':
            return '4-8 hours'
        elif effort == 'high':
            return '1-2 days'
        else:
            return 'unknown'
    
    def _generate_recommendations(self, evaluations: List[ToolEvaluation]) -> List[Dict[str, Any]]:
        """Generate tool recommendations"""
        recommendations = []
        
        # Sort by overall score
        sorted_evaluations = sorted(evaluations, key=lambda x: x.overall_score, reverse=True)
        
        for evaluation in sorted_evaluations[:5]:  # Top 5 recommendations
            recommendation = {
                'tool_name': evaluation.tool_name,
                'overall_score': evaluation.overall_score,
                'integration_effort': evaluation.integration_effort,
                'estimated_time': evaluation.estimated_time,
                'recommendations': evaluation.recommendations,
                'risks': evaluation.risks
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _format_discovery_response(self, discovered_tools: List[ToolSpecification], 
                                 evaluations: List[ToolEvaluation], 
                                 recommendations: List[Dict[str, Any]]) -> str:
        """Format discovery response"""
        response = f"🔍 **Tool Discovery Results**\n\n"
        
        response += f"📦 **Discovered Tools**: {len(discovered_tools)}\n"
        response += f"📊 **Evaluated Tools**: {len(evaluations)}\n"
        response += f"⭐ **Top Recommendations**: {len(recommendations)}\n\n"
        
        if recommendations:
            response += "**Top Tool Recommendations:**\n\n"
            for i, rec in enumerate(recommendations, 1):
                response += f"{i}. **{rec['tool_name']}** (Score: {rec['overall_score']:.2f})\n"
                response += f"   - Integration Effort: {rec['integration_effort']}\n"
                response += f"   - Estimated Time: {rec['estimated_time']}\n"
                if rec['recommendations']:
                    response += f"   - Recommendations: {', '.join(rec['recommendations'][:2])}\n"
                if rec['risks']:
                    response += f"   - Risks: {', '.join(rec['risks'][:2])}\n"
                response += "\n"
        
        return response
    
    async def integrate_tool(self, tool_name: str) -> str:
        """Integrate a specific tool"""
        try:
            if tool_name not in self.evaluated_tools:
                return f"Tool {tool_name} has not been evaluated yet."
            
            evaluation = self.evaluated_tools[tool_name]
            if evaluation.overall_score < 0.6:
                return f"Tool {tool_name} has low evaluation score ({evaluation.overall_score:.2f}). Integration not recommended."
            
            # Find tool specification
            tool_spec = None
            for tool in self.discovered_tools.values():
                if tool.name == tool_name:
                    tool_spec = tool
                    break
            
            if not tool_spec:
                return f"Tool specification for {tool_name} not found."
            
            # Perform integration
            integration_result = await self._perform_integration(tool_spec)
            
            if integration_result['success']:
                return f"✅ Successfully integrated {tool_name}! {integration_result['message']}"
            else:
                return f"❌ Failed to integrate {tool_name}: {integration_result['message']}"
        
        except Exception as e:
            logger.error(f"Error integrating tool {tool_name}: {e}")
            return f"Error integrating tool {tool_name}: {str(e)}"
    
    async def _perform_integration(self, tool: ToolSpecification) -> Dict[str, Any]:
        """Perform tool integration"""
        try:
            start_time = time.time()
            
            # Get integration method
            integration_method = tool.integration_method
            if integration_method in self.integration_templates:
                integration_func = self.integration_templates[integration_method]
                result = await integration_func(tool)
            else:
                result = {'success': False, 'message': f'Unknown integration method: {integration_method}'}
            
            integration_time = time.time() - start_time
            
            if result['success']:
                # Create integration record
                integration = ToolIntegration(
                    tool_name=tool.name,
                    integration_date=datetime.now(),
                    integration_method=integration_method,
                    configuration=result.get('configuration', {}),
                    endpoints=result.get('endpoints', []),
                    api_keys=result.get('api_keys', []),
                    environment_variables=result.get('environment_variables', []),
                    dependencies_installed=result.get('dependencies_installed', []),
                    test_results=result.get('test_results', {}),
                    monitoring_config=result.get('monitoring_config', {})
                )
                
                # Update integrated tools registry
                self.integrated_tools[tool.name] = integration
                
                # Update tool status
                tool.status = ToolStatus.INTEGRATED
                
                # Save to disk
                self._save_integrated_tools()
                
                # Update statistics
                self.discovery_stats['total_integrated'] += 1
                self.discovery_stats['average_integration_time'] = (
                    (self.discovery_stats['average_integration_time'] * 
                     (self.discovery_stats['total_integrated'] - 1) + integration_time) /
                    self.discovery_stats['total_integrated']
                )
            
            return result
        
        except Exception as e:
            logger.error(f"Error performing integration: {e}")
            return {'success': False, 'message': str(e)}
    
    async def _integrate_python_package(self, tool: ToolSpecification) -> Dict[str, Any]:
        """Integrate a Python package"""
        try:
            # Install package
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', tool.name
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Test import
                try:
                    importlib.import_module(tool.name)
                    return {
                        'success': True,
                        'message': f'Package {tool.name} installed and imported successfully',
                        'dependencies_installed': [tool.name],
                        'test_results': {'import_test': 'passed'}
                    }
                except ImportError as e:
                    return {
                        'success': False,
                        'message': f'Package installed but import failed: {str(e)}'
                    }
            else:
                return {
                    'success': False,
                    'message': f'Package installation failed: {result.stderr}'
                }
        
        except Exception as e:
            return {
                'success': False,
                'message': f'Integration error: {str(e)}'
            }
    
    async def _integrate_api_service(self, tool: ToolSpecification) -> Dict[str, Any]:
        """Integrate an API service"""
        try:
            # Create API client configuration
            config = {
                'base_url': tool.documentation_url,
                'api_key_required': True,
                'rate_limit': 1000,  # requests per hour
                'timeout': 30
            }
            
            return {
                'success': True,
                'message': f'API service {tool.name} configured',
                'configuration': config,
                'endpoints': ['/api/v1'],
                'api_keys': ['API_KEY'],
                'environment_variables': [f'{tool.name.upper()}_API_KEY']
            }
        
        except Exception as e:
            return {
                'success': False,
                'message': f'API integration error: {str(e)}'
            }
    
    async def _integrate_docker_container(self, tool: ToolSpecification) -> Dict[str, Any]:
        """Integrate a Docker container"""
        try:
            # Create Docker configuration
            config = {
                'image': f'{tool.name}:latest',
                'ports': ['8080:8080'],
                'environment': [],
                'volumes': []
            }
            
            return {
                'success': True,
                'message': f'Docker container {tool.name} configured',
                'configuration': config,
                'endpoints': ['http://localhost:8080'],
                'environment_variables': []
            }
        
        except Exception as e:
            return {
                'success': False,
                'message': f'Docker integration error: {str(e)}'
            }
    
    async def _integrate_webhook(self, tool: ToolSpecification) -> Dict[str, Any]:
        """Integrate a webhook"""
        try:
            # Create webhook configuration
            config = {
                'url': f'https://api.{tool.name}.com/webhook',
                'events': ['push', 'pull_request'],
                'secret': 'webhook_secret'
            }
            
            return {
                'success': True,
                'message': f'Webhook {tool.name} configured',
                'configuration': config,
                'endpoints': [config['url']],
                'environment_variables': [f'{tool.name.upper()}_WEBHOOK_SECRET']
            }
        
        except Exception as e:
            return {
                'success': False,
                'message': f'Webhook integration error: {str(e)}'
            }
    
    async def _integrate_custom_module(self, tool: ToolSpecification) -> Dict[str, Any]:
        """Integrate a custom module"""
        try:
            # Create custom module configuration
            config = {
                'module_path': f'tools/custom/{tool.name}',
                'entry_point': 'main',
                'dependencies': tool.dependencies
            }
            
            return {
                'success': True,
                'message': f'Custom module {tool.name} configured',
                'configuration': config,
                'endpoints': [],
                'environment_variables': []
            }
        
        except Exception as e:
            return {
                'success': False,
                'message': f'Custom module integration error: {str(e)}'
            }
    
    def _continuous_discovery(self):
        """Continuous tool discovery process"""
        try:
            # Discover new tools based on current needs
            current_capabilities = self._get_current_capabilities()
            missing_capabilities = self._identify_missing_capabilities(current_capabilities)
            
            if missing_capabilities:
                # Discover tools for missing capabilities
                requirements = {
                    'capabilities': missing_capabilities,
                    'priority': ToolPriority.MEDIUM
                }
                
                # This would be async in a real implementation
                logger.info(f"Continuous discovery: Looking for tools for capabilities: {missing_capabilities}")
        
        except Exception as e:
            logger.error(f"Error in continuous discovery: {e}")
    
    def _get_current_capabilities(self) -> List[str]:
        """Get current system capabilities"""
        capabilities = []
        
        # Check integrated tools
        for tool_name, integration in self.integrated_tools.items():
            if tool_name in self.discovered_tools:
                tool = self.discovered_tools[tool_name]
                capabilities.extend(tool.capabilities)
        
        return list(set(capabilities))
    
    def _identify_missing_capabilities(self, current_capabilities: List[str]) -> List[str]:
        """Identify missing capabilities"""
        all_capabilities = list(self.capability_mapping.keys())
        missing = [cap for cap in all_capabilities if cap not in current_capabilities]
        return missing[:3]  # Return top 3 missing capabilities
    
    def _save_discovered_tools(self):
        """Save discovered tools to disk"""
        try:
            data = {
                'discovered_tools': {
                    name: asdict(tool) for name, tool in self.discovered_tools.items()
                }
            }
            
            with open('tools/discovered/registry.json', 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Error saving discovered tools: {e}")
    
    def _save_evaluated_tools(self):
        """Save evaluated tools to disk"""
        try:
            data = {
                'evaluated_tools': {
                    name: asdict(evaluation) for name, evaluation in self.evaluated_tools.items()
                }
            }
            
            with open('tools/evaluated/registry.json', 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Error saving evaluated tools: {e}")
    
    def _save_integrated_tools(self):
        """Save integrated tools to disk"""
        try:
            data = {
                'integrated_tools': {
                    name: asdict(integration) for name, integration in self.integrated_tools.items()
                }
            }
            
            with open('tools/integrated/registry.json', 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Error saving integrated tools: {e}")
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get tool discovery statistics"""
        return {
            'total_discovered': len(self.discovered_tools),
            'total_evaluated': len(self.evaluated_tools),
            'total_integrated': len(self.integrated_tools),
            'success_rate': self.discovery_stats['success_rate'],
            'average_evaluation_time': self.discovery_stats['average_evaluation_time'],
            'average_integration_time': self.discovery_stats['average_integration_time']
        }
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool"""
        if tool_name in self.discovered_tools:
            tool = self.discovered_tools[tool_name]
            evaluation = self.evaluated_tools.get(tool_name)
            integration = self.integrated_tools.get(tool_name)
            
            return {
                'specification': asdict(tool),
                'evaluation': asdict(evaluation) if evaluation else None,
                'integration': asdict(integration) if integration else None
            }
        
        return None 