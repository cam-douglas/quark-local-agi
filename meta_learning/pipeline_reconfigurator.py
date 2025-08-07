#!/usr/bin/env python3
"""
Pipeline Reconfigurator for Quark AI Assistant
Dynamically reconfigures agent pipelines based on performance analysis

Part of Pillar 16: Meta-Learning & Self-Reflection
"""

import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import copy

@dataclass
class PipelineConfiguration:
    """A pipeline configuration for optimization."""
    config_id: str
    agent_sequence: List[str]
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    created_at: float
    last_updated: float
    status: str  # active, inactive, testing

@dataclass
class ReconfigurationAction:
    """An action to reconfigure the pipeline."""
    action_id: str
    action_type: str  # add_agent, remove_agent, reorder_agents, adjust_parameters
    target_agent: str
    parameters: Dict[str, Any]
    reason: str
    priority: str  # low, medium, high, critical
    timestamp: float
    applied: bool

class PipelineReconfigurator:
    def __init__(self, config_dir: str = None):
        self.config_dir = config_dir or os.path.join(os.path.dirname(__file__), '..', 'meta_learning')
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Pipeline configurations
        self.pipeline_configurations = []
        self.active_configuration = None
        self.reconfiguration_history = []
        
        # Reconfiguration settings
        self.auto_reconfiguration = True
        self.testing_mode = False
        self.rollback_enabled = True
        self.performance_threshold = 0.8
        
        # Agent capabilities and dependencies
        self.agent_capabilities = {
            'NLU': ['intent_classification', 'entity_recognition'],
            'Retrieval': ['semantic_search', 'keyword_search'],
            'Reasoning': ['chain_of_thought', 'logical_reasoning'],
            'Planning': ['task_decomposition', 'sequence_planning'],
            'Memory': ['context_management', 'long_term_storage'],
            'Metrics': ['performance_monitoring', 'evaluation'],
            'SelfImprovement': ['learning', 'optimization']
        }
        
        self.agent_dependencies = {
            'Reasoning': ['Retrieval'],
            'Planning': ['Reasoning'],
            'SelfImprovement': ['Metrics']
        }
        
        # Load existing configurations
        self._load_pipeline_configurations()
        
    def _load_pipeline_configurations(self):
        """Load existing pipeline configurations."""
        try:
            config_file = os.path.join(self.config_dir, 'pipeline_configurations.json')
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    self.pipeline_configurations = [PipelineConfiguration(**c) for c in data]
                    
                    # Set active configuration
                    active_configs = [c for c in self.pipeline_configurations if c.status == 'active']
                    if active_configs:
                        self.active_configuration = active_configs[0]
                        
        except Exception as e:
            print(f"Warning: Could not load pipeline configurations: {e}")
    
    def _save_pipeline_configurations(self):
        """Save pipeline configurations to disk."""
        try:
            config_file = os.path.join(self.config_dir, 'pipeline_configurations.json')
            with open(config_file, 'w') as f:
                json.dump([asdict(c) for c in self.pipeline_configurations], f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save pipeline configurations: {e}")
    
    def create_pipeline_configuration(self, agent_sequence: List[str], 
                                    parameters: Dict[str, Any] = None) -> str:
        """Create a new pipeline configuration."""
        config_id = f"config_{int(time.time() * 1000)}"
        
        # Validate agent sequence
        if not self._validate_agent_sequence(agent_sequence):
            raise ValueError("Invalid agent sequence")
        
        config = PipelineConfiguration(
            config_id=config_id,
            agent_sequence=agent_sequence,
            parameters=parameters or {},
            performance_metrics={},
            created_at=time.time(),
            last_updated=time.time(),
            status='testing'
        )
        
        self.pipeline_configurations.append(config)
        self._save_pipeline_configurations()
        
        return config_id
    
    def _validate_agent_sequence(self, agent_sequence: List[str]) -> bool:
        """Validate that the agent sequence is feasible."""
        # Check if all agents exist
        available_agents = set(self.agent_capabilities.keys())
        sequence_agents = set(agent_sequence)
        
        if not sequence_agents.issubset(available_agents):
            return False
        
        # Check dependencies
        for agent in agent_sequence:
            dependencies = self.agent_dependencies.get(agent, [])
            for dep in dependencies:
                if dep not in agent_sequence or agent_sequence.index(dep) >= agent_sequence.index(agent):
                    return False
        
        return True
    
    def analyze_pipeline_performance(self, config_id: str, 
                                   performance_data: Dict[str, float]) -> Dict[str, Any]:
        """Analyze performance of a pipeline configuration."""
        config = self._get_configuration(config_id)
        if not config:
            return {'status': 'error', 'message': 'Configuration not found'}
        
        # Update performance metrics
        config.performance_metrics.update(performance_data)
        config.last_updated = time.time()
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(config.performance_metrics)
        
        # Determine if reconfiguration is needed
        reconfiguration_needed = performance_score < self.performance_threshold
        
        # Generate recommendations
        recommendations = self._generate_reconfiguration_recommendations(config, performance_score)
        
        return {
            'status': 'success',
            'config_id': config_id,
            'performance_score': performance_score,
            'reconfiguration_needed': reconfiguration_needed,
            'recommendations': recommendations,
            'performance_metrics': config.performance_metrics
        }
    
    def _get_configuration(self, config_id: str) -> Optional[PipelineConfiguration]:
        """Get a pipeline configuration by ID."""
        for config in self.pipeline_configurations:
            if config.config_id == config_id:
                return config
        return None
    
    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall performance score from metrics."""
        if not metrics:
            return 0.0
        
        # Weight different metrics
        weights = {
            'accuracy': 0.4,
            'response_time': 0.3,
            'throughput': 0.2,
            'user_satisfaction': 0.1
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            weight = weights.get(metric, 0.1)
            
            # Normalize values (higher is better for most metrics)
            if metric == 'response_time':
                # Lower response time is better
                normalized_value = max(0.0, 1.0 - (value / 10.0))  # Assume 10s is max
            else:
                # Higher values are better
                normalized_value = min(1.0, value)
            
            score += normalized_value * weight
            total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _generate_reconfiguration_recommendations(self, config: PipelineConfiguration, 
                                                performance_score: float) -> List[Dict[str, Any]]:
        """Generate reconfiguration recommendations."""
        recommendations = []
        
        # Check for underperforming agents
        for agent in config.agent_sequence:
            agent_metrics = {k: v for k, v in config.performance_metrics.items() 
                           if k.startswith(f"{agent}_")}
            
            if agent_metrics:
                agent_score = self._calculate_performance_score(agent_metrics)
                if agent_score < 0.6:
                    recommendations.append({
                        'type': 'replace_agent',
                        'target_agent': agent,
                        'reason': f'Agent {agent} is underperforming (score: {agent_score:.2f})',
                        'priority': 'high' if agent_score < 0.4 else 'medium',
                        'confidence': 0.8
                    })
        
        # Check for missing agents that could improve performance
        missing_agents = set(self.agent_capabilities.keys()) - set(config.agent_sequence)
        for agent in missing_agents:
            if self._would_agent_improve_performance(agent, config):
                recommendations.append({
                    'type': 'add_agent',
                    'target_agent': agent,
                    'reason': f'Adding {agent} could improve performance',
                    'priority': 'medium',
                    'confidence': 0.6
                })
        
        # Check for parameter optimization
        if performance_score < 0.7:
            recommendations.append({
                'type': 'optimize_parameters',
                'target_agent': 'all',
                'reason': 'Parameter optimization could improve performance',
                'priority': 'medium',
                'confidence': 0.7
            })
        
        return recommendations
    
    def _would_agent_improve_performance(self, agent: str, config: PipelineConfiguration) -> bool:
        """Check if adding an agent would likely improve performance."""
        # Simple heuristic: check if agent provides unique capabilities
        current_capabilities = set()
        for existing_agent in config.agent_sequence:
            current_capabilities.update(self.agent_capabilities.get(existing_agent, []))
        
        new_capabilities = set(self.agent_capabilities.get(agent, []))
        unique_capabilities = new_capabilities - current_capabilities
        
        return len(unique_capabilities) > 0
    
    def apply_reconfiguration(self, config_id: str, 
                            recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply reconfiguration based on recommendations."""
        if not self.auto_reconfiguration:
            return {'status': 'disabled', 'message': 'Auto-reconfiguration is disabled'}
        
        config = self._get_configuration(config_id)
        if not config:
            return {'status': 'error', 'message': 'Configuration not found'}
        
        # Create backup of current configuration
        backup_config = copy.deepcopy(config)
        
        applied_actions = []
        failed_actions = []
        
        for recommendation in recommendations:
            try:
                action = self._create_reconfiguration_action(config, recommendation)
                if action:
                    success = self._apply_reconfiguration_action(config, action)
                    if success:
                        applied_actions.append(action)
                    else:
                        failed_actions.append(action)
            except Exception as e:
                failed_actions.append({
                    'recommendation': recommendation,
                    'error': str(e)
                })
        
        # Update configuration
        config.last_updated = time.time()
        self._save_pipeline_configurations()
        
        # Record reconfiguration
        self.reconfiguration_history.append({
            'timestamp': time.time(),
            'config_id': config_id,
            'applied_actions': len(applied_actions),
            'failed_actions': len(failed_actions),
            'backup_config': asdict(backup_config)
        })
        
        return {
            'status': 'completed',
            'config_id': config_id,
            'applied_actions': applied_actions,
            'failed_actions': failed_actions,
            'total_recommendations': len(recommendations)
        }
    
    def _create_reconfiguration_action(self, config: PipelineConfiguration, 
                                     recommendation: Dict[str, Any]) -> Optional[ReconfigurationAction]:
        """Create a reconfiguration action from a recommendation."""
        action_id = f"action_{int(time.time() * 1000)}"
        
        action_type = recommendation['type']
        target_agent = recommendation.get('target_agent', 'unknown')
        
        if action_type == 'replace_agent':
            # Find replacement agent
            replacement = self._find_replacement_agent(target_agent, config)
            if replacement:
                return ReconfigurationAction(
                    action_id=action_id,
                    action_type='replace_agent',
                    target_agent=target_agent,
                    parameters={'replacement_agent': replacement},
                    reason=recommendation['reason'],
                    priority=recommendation['priority'],
                    timestamp=time.time(),
                    applied=False
                )
        
        elif action_type == 'add_agent':
            return ReconfigurationAction(
                action_id=action_id,
                action_type='add_agent',
                target_agent=target_agent,
                parameters={'position': len(config.agent_sequence)},
                reason=recommendation['reason'],
                priority=recommendation['priority'],
                timestamp=time.time(),
                applied=False
            )
        
        elif action_type == 'optimize_parameters':
            return ReconfigurationAction(
                action_id=action_id,
                action_type='optimize_parameters',
                target_agent='all',
                parameters={'optimization_type': 'performance'},
                reason=recommendation['reason'],
                priority=recommendation['priority'],
                timestamp=time.time(),
                applied=False
            )
        
        return None
    
    def _find_replacement_agent(self, target_agent: str, config: PipelineConfiguration) -> Optional[str]:
        """Find a suitable replacement for an underperforming agent."""
        # Simple replacement logic
        replacements = {
            'NLU': ['Retrieval'],  # Use retrieval as fallback for NLU
            'Retrieval': ['Memory'],  # Use memory as fallback for retrieval
            'Reasoning': ['Planning'],  # Use planning as fallback for reasoning
            'Planning': ['Reasoning'],  # Use reasoning as fallback for planning
        }
        
        return replacements.get(target_agent)
    
    def _apply_reconfiguration_action(self, config: PipelineConfiguration, 
                                    action: ReconfigurationAction) -> bool:
        """Apply a reconfiguration action to the configuration."""
        try:
            if action.action_type == 'replace_agent':
                replacement = action.parameters.get('replacement_agent')
                if replacement and action.target_agent in config.agent_sequence:
                    idx = config.agent_sequence.index(action.target_agent)
                    config.agent_sequence[idx] = replacement
                    action.applied = True
            
            elif action.action_type == 'add_agent':
                position = action.parameters.get('position', len(config.agent_sequence))
                if action.target_agent not in config.agent_sequence:
                    config.agent_sequence.insert(position, action.target_agent)
                    action.applied = True
            
            elif action.action_type == 'optimize_parameters':
                # Apply parameter optimizations
                config.parameters.update(action.parameters)
                action.applied = True
            
            return action.applied
            
        except Exception as e:
            print(f"Error applying reconfiguration action: {e}")
            return False
    
    def activate_configuration(self, config_id: str) -> Dict[str, Any]:
        """Activate a pipeline configuration."""
        config = self._get_configuration(config_id)
        if not config:
            return {'status': 'error', 'message': 'Configuration not found'}
        
        # Deactivate current active configuration
        if self.active_configuration:
            self.active_configuration.status = 'inactive'
        
        # Activate new configuration
        config.status = 'active'
        self.active_configuration = config
        
        self._save_pipeline_configurations()
        
        return {
            'status': 'success',
            'config_id': config_id,
            'agent_sequence': config.agent_sequence,
            'parameters': config.parameters
        }
    
    def get_reconfiguration_statistics(self) -> Dict[str, Any]:
        """Get reconfiguration statistics."""
        return {
            'total_configurations': len(self.pipeline_configurations),
            'active_configuration': self.active_configuration.config_id if self.active_configuration else None,
            'total_reconfigurations': len(self.reconfiguration_history),
            'auto_reconfiguration': self.auto_reconfiguration,
            'testing_mode': self.testing_mode,
            'rollback_enabled': self.rollback_enabled,
            'performance_threshold': self.performance_threshold
        } 