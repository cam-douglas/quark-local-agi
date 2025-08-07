"""
Multi-Agent Negotiation & Coordination Agent
Pillar 28: Specialized agent roles, task bargaining, working group formation, 
collaborative problem solving, and distributed decision making
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
import random

from .base import Agent as BaseAgent


class AgentRole(Enum):
    """Specialized agent roles"""
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    VALIDATOR = "validator"
    EXECUTOR = "executor"
    OBSERVER = "observer"
    MEDIATOR = "mediator"


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class NegotiationStatus(Enum):
    """Negotiation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AGREED = "agreed"
    DISAGREED = "disagreed"
    TIMEOUT = "timeout"


class DecisionType(Enum):
    """Types of distributed decisions"""
    CONSENSUS = "consensus"
    MAJORITY = "majority"
    LEADER = "leader"
    WEIGHTED = "weighted"
    AUTONOMOUS = "autonomous"


@dataclass
class AgentCapability:
    """Agent capability specification"""
    capability_id: str
    name: str
    description: str
    confidence: float
    performance_metrics: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    availability: bool


@dataclass
class TaskSpecification:
    """Task specification for negotiation"""
    task_id: str
    title: str
    description: str
    priority: TaskPriority
    complexity: float
    estimated_duration: float
    required_capabilities: List[str]
    constraints: Dict[str, Any]
    deadline: Optional[datetime]
    dependencies: List[str]


@dataclass
class NegotiationProposal:
    """Negotiation proposal between agents"""
    proposal_id: str
    proposer_id: str
    task_id: str
    proposed_agents: List[str]
    proposed_roles: Dict[str, AgentRole]
    proposed_timeline: float
    proposed_resources: Dict[str, Any]
    confidence: float
    rationale: str
    alternatives: List[Dict[str, Any]]
    timestamp: datetime


@dataclass
class WorkingGroup:
    """Working group for collaborative problem solving"""
    group_id: str
    name: str
    purpose: str
    members: List[str]
    roles: Dict[str, AgentRole]
    coordinator: str
    status: str
    created_time: datetime
    expected_completion: Optional[datetime]
    progress: float
    communication_channels: List[str]


@dataclass
class CollaborativeDecision:
    """Distributed decision result"""
    decision_id: str
    decision_type: DecisionType
    participants: List[str]
    options: List[Dict[str, Any]]
    selected_option: Dict[str, Any]
    confidence: float
    reasoning: str
    consensus_level: float
    timestamp: datetime
    metadata: Dict[str, Any]


class NegotiationAgent(BaseAgent):
    """
    Multi-Agent Negotiation & Coordination Agent
    
    Implements specialized agent roles, task bargaining and negotiation,
    working group formation, collaborative problem solving, and distributed decision making.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.agent_name = "NegotiationAgent"
        self.agent_capabilities = [
            "multi_agent_coordination",
            "task_negotiation",
            "working_group_formation",
            "collaborative_problem_solving",
            "distributed_decision_making"
        ]
        
        # Initialize negotiation systems
        self._initialize_negotiation_systems()
        
        # Agent registry and capabilities
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.agent_capabilities_db: Dict[str, AgentCapability] = {}
        
        # Active negotiations and working groups
        self.active_negotiations: Dict[str, Dict[str, Any]] = {}
        self.working_groups: Dict[str, WorkingGroup] = {}
        self.collaborative_decisions: Dict[str, CollaborativeDecision] = {}
        
        # Communication channels
        self.communication_channels: Dict[str, List[str]] = defaultdict(list)
        
        # Performance tracking
        self.negotiation_stats = {
            "total_negotiations": 0,
            "successful_negotiations": 0,
            "average_negotiation_time": 0.0,
            "working_groups_formed": 0,
            "collaborative_decisions": 0
        }
        
        logging.info(f"Initialized {self.agent_name} with negotiation capabilities")
    
    def load_model(self):
        """Load the negotiation model"""
        # For now, use a simple rule-based system
        # In production, this could be a trained model
        self.model = "negotiation_rules"
        logging.info(f"Loaded negotiation model: {self.model}")
    
    def generate(self, prompt: str, **kwargs):
        """Generate negotiation response"""
        # Simple response generation for now
        return f"Negotiation response: {prompt}"
    
    def _initialize_negotiation_systems(self):
        """Initialize negotiation and coordination systems"""
        self.negotiation_rules = {
            "timeout_duration": 300,  # 5 minutes
            "max_retries": 3,
            "consensus_threshold": 0.7,
            "majority_threshold": 0.5,
            "min_participants": 2,
            "max_participants": 10
        }
        
        self.coordination_strategies = {
            "hierarchical": self._hierarchical_coordination,
            "peer_to_peer": self._peer_to_peer_coordination,
            "consensus_based": self._consensus_based_coordination,
            "leader_based": self._leader_based_coordination
        }
        
        logging.info("Initialized negotiation systems")
    
    def _hierarchical_coordination(self, agents: List[str], task_spec: TaskSpecification) -> Dict[str, Any]:
        """Hierarchical coordination strategy"""
        return {
            "strategy": "hierarchical",
            "coordinator": agents[0] if agents else None,
            "subordinates": agents[1:] if len(agents) > 1 else [],
            "communication_flow": "top_down"
        }
    
    def _peer_to_peer_coordination(self, agents: List[str], task_spec: TaskSpecification) -> Dict[str, Any]:
        """Peer-to-peer coordination strategy"""
        return {
            "strategy": "peer_to_peer",
            "participants": agents,
            "communication_flow": "distributed"
        }
    
    def _consensus_based_coordination(self, agents: List[str], task_spec: TaskSpecification) -> Dict[str, Any]:
        """Consensus-based coordination strategy"""
        return {
            "strategy": "consensus_based",
            "participants": agents,
            "consensus_threshold": 0.7,
            "communication_flow": "collaborative"
        }
    
    def _leader_based_coordination(self, agents: List[str], task_spec: TaskSpecification) -> Dict[str, Any]:
        """Leader-based coordination strategy"""
        return {
            "strategy": "leader_based",
            "leader": agents[0] if agents else None,
            "followers": agents[1:] if len(agents) > 1 else [],
            "communication_flow": "leader_follower"
        }
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming negotiation messages"""
        try:
            message_type = message.get("type", "")
            
            if message_type == "register_agent":
                return await self._register_agent(message)
            elif message_type == "propose_task":
                return await self._propose_task(message)
            elif message_type == "negotiate_task":
                return await self._negotiate_task(message)
            elif message_type == "form_working_group":
                return await self._form_working_group(message)
            elif message_type == "collaborative_decision":
                return await self._make_collaborative_decision(message)
            elif message_type == "get_negotiation_status":
                return await self._get_negotiation_status(message)
            elif message_type == "get_working_groups":
                return await self._get_working_groups(message)
            elif message_type == "get_agent_capabilities":
                return await self._get_agent_capabilities(message)
            else:
                return {
                    "status": "error",
                    "message": f"Unknown message type: {message_type}",
                    "agent": self.agent_name
                }
                
        except Exception as e:
            logging.error(f"Error in negotiation agent: {str(e)}")
            return {
                "status": "error",
                "message": f"Negotiation error: {str(e)}",
                "agent": self.agent_name
            }
    
    async def _register_agent(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Register an agent with its capabilities"""
        try:
            agent_id = message.get("agent_id")
            capabilities = message.get("capabilities", [])
            performance_metrics = message.get("performance_metrics", {})
            resource_requirements = message.get("resource_requirements", {})
            
            # Create capability objects
            agent_capabilities = []
            for cap in capabilities:
                capability = AgentCapability(
                    capability_id=str(uuid.uuid4()),
                    name=cap.get("name", ""),
                    description=cap.get("description", ""),
                    confidence=cap.get("confidence", 0.8),
                    performance_metrics=cap.get("performance_metrics", {}),
                    resource_requirements=cap.get("resource_requirements", {}),
                    availability=True
                )
                agent_capabilities.append(capability)
                self.agent_capabilities_db[capability.capability_id] = capability
            
            # Register agent
            self.agent_registry[agent_id] = {
                "capabilities": [cap.capability_id for cap in agent_capabilities],
                "performance_metrics": performance_metrics,
                "resource_requirements": resource_requirements,
                "registration_time": datetime.now(),
                "status": "active"
            }
            
            logging.info(f"Registered agent {agent_id} with {len(capabilities)} capabilities")
            
            return {
                "status": "success",
                "message": f"Agent {agent_id} registered successfully",
                "agent": self.agent_name,
                "capabilities_count": len(capabilities)
            }
            
        except Exception as e:
            logging.error(f"Error registering agent: {str(e)}")
            return {
                "status": "error",
                "message": f"Registration error: {str(e)}",
                "agent": self.agent_name
            }
    
    async def _propose_task(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Propose a task for negotiation"""
        try:
            task_spec = TaskSpecification(
                task_id=message.get("task_id", str(uuid.uuid4())),
                title=message.get("title", ""),
                description=message.get("description", ""),
                priority=TaskPriority(message.get("priority", "medium")),
                complexity=message.get("complexity", 0.5),
                estimated_duration=message.get("estimated_duration", 60.0),
                required_capabilities=message.get("required_capabilities", []),
                constraints=message.get("constraints", {}),
                deadline=message.get("deadline"),
                dependencies=message.get("dependencies", [])
            )
            
            # Find suitable agents
            suitable_agents = await self._find_suitable_agents(task_spec)
            
            # Create initial proposal
            proposal = NegotiationProposal(
                proposal_id=str(uuid.uuid4()),
                proposer_id=message.get("proposer_id", self.agent_name),
                task_id=task_spec.task_id,
                proposed_agents=suitable_agents,
                proposed_roles=await self._assign_roles(suitable_agents, task_spec),
                proposed_timeline=task_spec.estimated_duration,
                proposed_resources=await self._estimate_resources(task_spec),
                confidence=await self._calculate_confidence(task_spec, suitable_agents),
                rationale=await self._generate_rationale(task_spec, suitable_agents),
                alternatives=await self._generate_alternatives(task_spec),
                timestamp=datetime.now()
            )
            
            # Start negotiation
            negotiation_id = str(uuid.uuid4())
            self.active_negotiations[negotiation_id] = {
                "task_spec": task_spec,
                "proposal": proposal,
                "status": NegotiationStatus.PENDING,
                "participants": suitable_agents,
                "start_time": datetime.now(),
                "responses": {}
            }
            
            self.negotiation_stats["total_negotiations"] += 1
            
            logging.info(f"Proposed task {task_spec.task_id} with {len(suitable_agents)} agents")
            
            return {
                "status": "success",
                "message": f"Task {task_spec.task_id} proposed for negotiation",
                "agent": self.agent_name,
                "negotiation_id": negotiation_id,
                "proposal": asdict(proposal),
                "suitable_agents": suitable_agents
            }
            
        except Exception as e:
            logging.error(f"Error proposing task: {str(e)}")
            return {
                "status": "error",
                "message": f"Task proposal error: {str(e)}",
                "agent": self.agent_name
            }
    
    async def _negotiate_task(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Negotiate a task with multiple agents"""
        try:
            negotiation_id = message.get("negotiation_id")
            response = message.get("response", {})
            agent_id = message.get("agent_id")
            
            if negotiation_id not in self.active_negotiations:
                return {
                    "status": "error",
                    "message": f"Negotiation {negotiation_id} not found",
                    "agent": self.agent_name
                }
            
            negotiation = self.active_negotiations[negotiation_id]
            negotiation["responses"][agent_id] = response
            
            # Check if all participants have responded
            if len(negotiation["responses"]) == len(negotiation["participants"]):
                result = await self._finalize_negotiation(negotiation_id)
                return {
                    "status": "success",
                    "message": "Negotiation finalized",
                    "agent": self.agent_name,
                    "result": result
                }
            else:
                return {
                    "status": "success",
                    "message": f"Response recorded, {len(negotiation['responses'])}/{len(negotiation['participants'])} responses",
                    "agent": self.agent_name,
                    "remaining_participants": len(negotiation["participants"]) - len(negotiation["responses"])
                }
                
        except Exception as e:
            logging.error(f"Error in task negotiation: {str(e)}")
            return {
                "status": "error",
                "message": f"Negotiation error: {str(e)}",
                "agent": self.agent_name
            }
    
    async def _form_working_group(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Form a working group for collaborative problem solving"""
        try:
            group_id = message.get("group_id", str(uuid.uuid4()))
            name = message.get("name", f"Working Group {group_id[:8]}")
            purpose = message.get("purpose", "")
            members = message.get("members", [])
            coordinator = message.get("coordinator")
            
            if not coordinator and members:
                coordinator = members[0]
            
            # Assign roles to members
            roles = await self._assign_group_roles(members, purpose)
            
            # Create working group
            working_group = WorkingGroup(
                group_id=group_id,
                name=name,
                purpose=purpose,
                members=members,
                roles=roles,
                coordinator=coordinator,
                status="active",
                created_time=datetime.now(),
                expected_completion=message.get("expected_completion"),
                progress=0.0,
                communication_channels=await self._setup_communication_channels(group_id, members)
            )
            
            self.working_groups[group_id] = working_group
            self.negotiation_stats["working_groups_formed"] += 1
            
            logging.info(f"Formed working group {group_id} with {len(members)} members")
            
            return {
                "status": "success",
                "message": f"Working group {group_id} formed successfully",
                "agent": self.agent_name,
                "working_group": asdict(working_group)
            }
            
        except Exception as e:
            logging.error(f"Error forming working group: {str(e)}")
            return {
                "status": "error",
                "message": f"Working group formation error: {str(e)}",
                "agent": self.agent_name
            }
    
    async def _make_collaborative_decision(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Make a collaborative decision with multiple agents"""
        try:
            decision_id = message.get("decision_id", str(uuid.uuid4()))
            decision_type = DecisionType(message.get("decision_type", "consensus"))
            participants = message.get("participants", [])
            options = message.get("options", [])
            
            if not participants or not options:
                return {
                    "status": "error",
                    "message": "Participants and options are required",
                    "agent": self.agent_name
                }
            
            # Collect votes from participants
            votes = await self._collect_votes(participants, options)
            
            # Make decision based on type
            if decision_type == DecisionType.CONSENSUS:
                result = await self._consensus_decision(votes, options)
            elif decision_type == DecisionType.MAJORITY:
                result = await self._majority_decision(votes, options)
            elif decision_type == DecisionType.LEADER:
                result = await self._leader_decision(participants, options)
            elif decision_type == DecisionType.WEIGHTED:
                result = await self._weighted_decision(votes, options, participants)
            else:
                result = await self._autonomous_decision(options)
            
            # Create collaborative decision record
            collaborative_decision = CollaborativeDecision(
                decision_id=decision_id,
                decision_type=decision_type,
                participants=participants,
                options=options,
                selected_option=result["selected_option"],
                confidence=result["confidence"],
                reasoning=result["reasoning"],
                consensus_level=result.get("consensus_level", 0.0),
                timestamp=datetime.now(),
                metadata=result.get("metadata", {})
            )
            
            self.collaborative_decisions[decision_id] = collaborative_decision
            self.negotiation_stats["collaborative_decisions"] += 1
            
            logging.info(f"Made collaborative decision {decision_id} with {len(participants)} participants")
            
            return {
                "status": "success",
                "message": f"Collaborative decision {decision_id} made successfully",
                "agent": self.agent_name,
                "decision": asdict(collaborative_decision)
            }
            
        except Exception as e:
            logging.error(f"Error making collaborative decision: {str(e)}")
            return {
                "status": "error",
                "message": f"Collaborative decision error: {str(e)}",
                "agent": self.agent_name
            }
    
    async def _find_suitable_agents(self, task_spec: TaskSpecification) -> List[str]:
        """Find agents suitable for a task"""
        suitable_agents = []
        
        for agent_id, agent_info in self.agent_registry.items():
            if agent_info["status"] != "active":
                continue
                
            agent_capabilities = [self.agent_capabilities_db[cap_id] for cap_id in agent_info["capabilities"]]
            
            # Check if agent has required capabilities
            has_required_capabilities = all(
                any(cap.name in task_spec.required_capabilities for cap in agent_capabilities)
            )
            
            if has_required_capabilities:
                suitable_agents.append(agent_id)
        
        return suitable_agents
    
    async def _assign_roles(self, agents: List[str], task_spec: TaskSpecification) -> Dict[str, AgentRole]:
        """Assign roles to agents based on task requirements"""
        roles = {}
        
        if not agents:
            return roles
        
        # Simple role assignment logic
        if len(agents) >= 1:
            roles[agents[0]] = AgentRole.COORDINATOR
        
        if len(agents) >= 2:
            roles[agents[1]] = AgentRole.SPECIALIST
        
        if len(agents) >= 3:
            roles[agents[2]] = AgentRole.VALIDATOR
        
        for i, agent in enumerate(agents[3:], 3):
            if i % 2 == 0:
                roles[agent] = AgentRole.EXECUTOR
            else:
                roles[agent] = AgentRole.OBSERVER
        
        return roles
    
    async def _estimate_resources(self, task_spec: TaskSpecification) -> Dict[str, Any]:
        """Estimate resources needed for task"""
        return {
            "cpu_cores": max(1, int(task_spec.complexity * 4)),
            "memory_gb": max(1, int(task_spec.complexity * 8)),
            "gpu_required": task_spec.complexity > 0.7,
            "network_bandwidth": "medium",
            "estimated_cost": task_spec.complexity * 100
        }
    
    async def _calculate_confidence(self, task_spec: TaskSpecification, agents: List[str]) -> float:
        """Calculate confidence in task completion"""
        if not agents:
            return 0.0
        
        # Simple confidence calculation
        base_confidence = 0.5
        agent_factor = min(1.0, len(agents) / 5.0)
        complexity_factor = 1.0 - task_spec.complexity
        
        return min(0.95, base_confidence + agent_factor + complexity_factor)
    
    async def _generate_rationale(self, task_spec: TaskSpecification, agents: List[str]) -> str:
        """Generate rationale for task proposal"""
        return f"Task '{task_spec.title}' requires {len(task_spec.required_capabilities)} capabilities. " \
               f"Found {len(agents)} suitable agents with matching capabilities. " \
               f"Estimated duration: {task_spec.estimated_duration} seconds."
    
    async def _generate_alternatives(self, task_spec: TaskSpecification) -> List[Dict[str, Any]]:
        """Generate alternative approaches for task"""
        alternatives = []
        
        # Alternative 1: Different agent combination
        if len(self.agent_registry) > 3:
            alternative_agents = list(self.agent_registry.keys())[:3]
            alternatives.append({
                "type": "different_agents",
                "agents": alternative_agents,
                "confidence": 0.6,
                "rationale": "Alternative agent combination"
            })
        
        # Alternative 2: Simplified approach
        alternatives.append({
            "type": "simplified_approach",
            "complexity_reduction": 0.3,
            "confidence": 0.7,
            "rationale": "Simplified task approach"
        })
        
        return alternatives
    
    async def _finalize_negotiation(self, negotiation_id: str) -> Dict[str, Any]:
        """Finalize a negotiation based on responses"""
        negotiation = self.active_negotiations[negotiation_id]
        
        # Analyze responses
        agreed_count = sum(1 for response in negotiation["responses"].values() 
                          if response.get("agreement", False))
        total_participants = len(negotiation["participants"])
        
        if agreed_count / total_participants >= self.negotiation_rules["consensus_threshold"]:
            negotiation["status"] = NegotiationStatus.AGREED
            self.negotiation_stats["successful_negotiations"] += 1
            
            # Calculate negotiation time
            negotiation_time = (datetime.now() - negotiation["start_time"]).total_seconds()
            self.negotiation_stats["average_negotiation_time"] = (
                (self.negotiation_stats["average_negotiation_time"] * (self.negotiation_stats["total_negotiations"] - 1) + negotiation_time) /
                self.negotiation_stats["total_negotiations"]
            )
            
            return {
                "status": "agreed",
                "agreement_level": agreed_count / total_participants,
                "negotiation_time": negotiation_time,
                "working_group_ready": True
            }
        else:
            negotiation["status"] = NegotiationStatus.DISAGREED
            return {
                "status": "disagreed",
                "agreement_level": agreed_count / total_participants,
                "reason": "Insufficient consensus"
            }
    
    async def _assign_group_roles(self, members: List[str], purpose: str) -> Dict[str, AgentRole]:
        """Assign roles to working group members"""
        roles = {}
        
        if not members:
            return roles
        
        # Assign coordinator
        roles[members[0]] = AgentRole.COORDINATOR
        
        # Assign other roles based on purpose
        if "validation" in purpose.lower():
            if len(members) > 1:
                roles[members[1]] = AgentRole.VALIDATOR
        
        if "specialization" in purpose.lower():
            for i, member in enumerate(members[1:], 1):
                roles[member] = AgentRole.SPECIALIST
        
        # Default role assignment
        for member in members:
            if member not in roles:
                roles[member] = AgentRole.EXECUTOR
        
        return roles
    
    async def _setup_communication_channels(self, group_id: str, members: List[str]) -> List[str]:
        """Setup communication channels for working group"""
        channels = [
            f"group_{group_id}_general",
            f"group_{group_id}_coordinator",
            f"group_{group_id}_status"
        ]
        
        # Add member-specific channels
        for member in members:
            channels.append(f"group_{group_id}_{member}")
        
        return channels
    
    async def _collect_votes(self, participants: List[str], options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect votes from participants"""
        votes = {}
        
        # Simulate voting process
        for participant in participants:
            # In a real system, this would involve actual communication with agents
            selected_option = random.choice(options)
            votes[participant] = {
                "selected_option": selected_option,
                "confidence": random.uniform(0.6, 0.9),
                "reasoning": f"Agent {participant} selected option based on analysis"
            }
        
        return votes
    
    async def _consensus_decision(self, votes: Dict[str, Any], options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make consensus-based decision"""
        option_counts = defaultdict(int)
        
        for vote in votes.values():
            option_counts[str(vote["selected_option"])] += 1
        
        # Find most voted option
        most_voted = max(option_counts.items(), key=lambda x: x[1])
        consensus_level = most_voted[1] / len(votes)
        
        return {
            "selected_option": eval(most_voted[0]),
            "confidence": consensus_level,
            "reasoning": f"Consensus decision with {consensus_level:.2f} agreement",
            "consensus_level": consensus_level
        }
    
    async def _majority_decision(self, votes: Dict[str, Any], options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make majority-based decision"""
        return await self._consensus_decision(votes, options)
    
    async def _leader_decision(self, participants: List[str], options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make leader-based decision"""
        leader = participants[0] if participants else None
        
        if leader:
            selected_option = random.choice(options)
            return {
                "selected_option": selected_option,
                "confidence": 0.8,
                "reasoning": f"Leader {leader} made the decision"
            }
        else:
            return {
                "selected_option": options[0],
                "confidence": 0.5,
                "reasoning": "No leader available, default selection"
            }
    
    async def _weighted_decision(self, votes: Dict[str, Any], options: List[Dict[str, Any]], participants: List[str]) -> Dict[str, Any]:
        """Make weighted decision based on agent capabilities"""
        weighted_votes = defaultdict(float)
        
        for participant, vote in votes.items():
            # Weight based on agent capabilities (simplified)
            weight = 1.0
            if participant in self.agent_registry:
                capabilities_count = len(self.agent_registry[participant]["capabilities"])
                weight = min(2.0, 1.0 + capabilities_count * 0.1)
            
            weighted_votes[str(vote["selected_option"])] += weight
        
        most_voted = max(weighted_votes.items(), key=lambda x: x[1])
        
        return {
            "selected_option": eval(most_voted[0]),
            "confidence": most_voted[1] / sum(weighted_votes.values()),
            "reasoning": f"Weighted decision with {most_voted[1]:.2f} total weight"
        }
    
    async def _autonomous_decision(self, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make autonomous decision"""
        selected_option = random.choice(options)
        
        return {
            "selected_option": selected_option,
            "confidence": 0.7,
            "reasoning": "Autonomous decision based on internal analysis"
        }
    
    async def _get_negotiation_status(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Get status of active negotiations"""
        negotiation_id = message.get("negotiation_id")
        
        if negotiation_id and negotiation_id in self.active_negotiations:
            negotiation = self.active_negotiations[negotiation_id]
            return {
                "status": "success",
                "message": f"Negotiation {negotiation_id} status retrieved",
                "agent": self.agent_name,
                "negotiation": {
                    "status": negotiation["status"].value,
                    "participants": negotiation["participants"],
                    "responses_count": len(negotiation["responses"]),
                    "start_time": negotiation["start_time"].isoformat()
                }
            }
        else:
            return {
                "status": "success",
                "message": "All active negotiations",
                "agent": self.agent_name,
                "active_negotiations": len(self.active_negotiations),
                "negotiation_stats": self.negotiation_stats
            }
    
    async def _get_working_groups(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Get information about working groups"""
        group_id = message.get("group_id")
        
        if group_id and group_id in self.working_groups:
            working_group = self.working_groups[group_id]
            return {
                "status": "success",
                "message": f"Working group {group_id} information retrieved",
                "agent": self.agent_name,
                "working_group": asdict(working_group)
            }
        else:
            return {
                "status": "success",
                "message": "All working groups",
                "agent": self.agent_name,
                "working_groups_count": len(self.working_groups),
                "working_groups": [asdict(wg) for wg in self.working_groups.values()]
            }
    
    async def _get_agent_capabilities(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Get agent capabilities information"""
        agent_id = message.get("agent_id")
        
        if agent_id and agent_id in self.agent_registry:
            agent_info = self.agent_registry[agent_id]
            capabilities = [self.agent_capabilities_db[cap_id] for cap_id in agent_info["capabilities"]]
            
            return {
                "status": "success",
                "message": f"Agent {agent_id} capabilities retrieved",
                "agent": self.agent_name,
                "agent_capabilities": [asdict(cap) for cap in capabilities],
                "agent_info": agent_info
            }
        else:
            return {
                "status": "success",
                "message": "All registered agents",
                "agent": self.agent_name,
                "registered_agents": len(self.agent_registry),
                "agent_registry": self.agent_registry
            }
    
    async def get_negotiation_stats(self) -> Dict[str, Any]:
        """Get negotiation statistics"""
        return {
            "agent": self.agent_name,
            "stats": self.negotiation_stats,
            "active_negotiations": len(self.active_negotiations),
            "working_groups": len(self.working_groups),
            "collaborative_decisions": len(self.collaborative_decisions),
            "registered_agents": len(self.agent_registry)
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "agent_name": self.agent_name,
            "capabilities": self.agent_capabilities,
            "description": "Multi-Agent Negotiation & Coordination Agent",
            "version": "1.0.0",
            "status": "active"
        } 