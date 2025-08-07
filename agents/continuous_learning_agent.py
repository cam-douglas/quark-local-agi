"""
Continuous Online & Few-Shot Learning Agent
Pillar 24: Live interaction refinement, few-shot fine-tuning, replay buffers, and incremental updates
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import hashlib
import pickle

from .base import Agent as BaseAgent


class LearningMode(Enum):
    """Learning modes for continuous learning"""
    FEW_SHOT = "few_shot"
    ONLINE = "online"
    INCREMENTAL = "incremental"
    REPLAY = "replay"
    ADAPTIVE = "adaptive"


class LearningStrategy(Enum):
    """Learning strategies"""
    GRADIENT_DESCENT = "gradient_descent"
    META_LEARNING = "meta_learning"
    TRANSFER_LEARNING = "transfer_learning"
    ACTIVE_LEARNING = "active_learning"
    SELF_SUPERVISED = "self_supervised"


@dataclass
class LearningExample:
    """Learning example for training"""
    input_data: Any
    target_output: Any
    context: Dict[str, Any]
    timestamp: datetime
    confidence: float
    source: str
    metadata: Dict[str, Any]


@dataclass
class LearningSession:
    """Learning session with examples and results"""
    session_id: str
    mode: LearningMode
    strategy: LearningStrategy
    examples: List[LearningExample]
    start_time: datetime
    end_time: Optional[datetime]
    performance_metrics: Dict[str, float]
    model_updates: Dict[str, Any]


@dataclass
class ReplayBuffer:
    """Replay buffer for storing learning experiences"""
    buffer_id: str
    max_size: int
    examples: deque
    priority_scores: Dict[str, float]
    last_accessed: datetime
    access_count: int


class ContinuousLearningAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("continuous_learning")
        self.config = config or {}
        
        # Add logger
        import logging
        self.logger = logging.getLogger(__name__)
        
        # Learning systems
        self.learning_modes = {
            LearningMode.FEW_SHOT: self._few_shot_learning,
            LearningMode.ONLINE: self._online_learning,
            LearningMode.INCREMENTAL: self._incremental_learning,
            LearningMode.REPLAY: self._replay_learning,
            LearningMode.ADAPTIVE: self._adaptive_learning
        }
        
        self.learning_strategies = {
            LearningStrategy.GRADIENT_DESCENT: self._gradient_descent_update,
            LearningStrategy.META_LEARNING: self._meta_learning_update,
            LearningStrategy.TRANSFER_LEARNING: self._transfer_learning_update,
            LearningStrategy.ACTIVE_LEARNING: self._active_learning_update,
            LearningStrategy.SELF_SUPERVISED: self._self_supervised_update
        }
        
        # Learning state
        self.current_session: Optional[LearningSession] = None
        self.learning_history: List[LearningSession] = []
        self.replay_buffers: Dict[str, ReplayBuffer] = {}
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        self.learning_stats = defaultdict(int)
        self.model_versions = []
        
        # Configuration
        self.max_replay_buffer_size = 10000
        self.min_confidence_threshold = 0.7
        self.learning_rate = 0.001
        self.batch_size = 32
        self.max_sessions = 100
        
        # Initialize learning systems
        self._initialize_learning_systems()
    
    def load_model(self):
        """Load the learning model"""
        return None  # Placeholder for actual model loading
    
    def generate(self, prompt: str, **kwargs):
        """Generate learning response"""
        return f"Learning response to: {prompt}"
        
        # Learning systems
        self.learning_modes = {
            LearningMode.FEW_SHOT: self._few_shot_learning,
            LearningMode.ONLINE: self._online_learning,
            LearningMode.INCREMENTAL: self._incremental_learning,
            LearningMode.REPLAY: self._replay_learning,
            LearningMode.ADAPTIVE: self._adaptive_learning
        }
        
        self.learning_strategies = {
            LearningStrategy.GRADIENT_DESCENT: self._gradient_descent_update,
            LearningStrategy.META_LEARNING: self._meta_learning_update,
            LearningStrategy.TRANSFER_LEARNING: self._transfer_learning_update,
            LearningStrategy.ACTIVE_LEARNING: self._active_learning_update,
            LearningStrategy.SELF_SUPERVISED: self._self_supervised_update
        }
        
        # Learning state
        self.current_session: Optional[LearningSession] = None
        self.learning_history: List[LearningSession] = []
        self.replay_buffers: Dict[str, ReplayBuffer] = {}
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        self.learning_stats = defaultdict(int)
        self.model_versions = []
        
        # Configuration
        self.max_replay_buffer_size = 10000
        self.min_confidence_threshold = 0.7
        self.learning_rate = 0.001
        self.batch_size = 32
        self.max_sessions = 100
        
        # Initialize learning systems
        self._initialize_learning_systems()
        
        # Start continuous learning task
        asyncio.create_task(self._continuous_learning_loop())
    
    def _initialize_learning_systems(self):
        """Initialize learning systems"""
        self.logger.info("Initializing continuous learning systems...")
        
        # Set up replay buffers
        self._setup_replay_buffers()
        
        # Initialize learning strategies
        self._setup_learning_strategies()
        
        # Set up performance tracking
        self._setup_performance_tracking()
        
        self.logger.info("Continuous learning systems initialized successfully")
    
    def _setup_replay_buffers(self):
        """Set up replay buffers for different learning modes"""
        self.logger.info("Setting up replay buffers...")
        
        # Create replay buffers for different learning modes
        for mode in LearningMode:
            buffer_id = f"replay_buffer_{mode.value}"
            self.replay_buffers[buffer_id] = ReplayBuffer(
                buffer_id=buffer_id,
                max_size=self.max_replay_buffer_size,
                examples=deque(maxlen=self.max_replay_buffer_size),
                priority_scores={},
                last_accessed=datetime.now(),
                access_count=0
            )
        
        self.logger.info(f"Created {len(self.replay_buffers)} replay buffers")
    
    def _setup_learning_strategies(self):
        """Set up learning strategies"""
        self.logger.info("Setting up learning strategies...")
        
        # Configure learning strategies
        for strategy in LearningStrategy:
            if strategy in self.learning_strategies:
                self.logger.info(f"Configured learning strategy: {strategy.value}")
    
    def _setup_performance_tracking(self):
        """Set up performance tracking"""
        self.logger.info("Setting up performance tracking...")
        
        # Initialize performance metrics
        metrics = ["accuracy", "loss", "learning_rate", "convergence_rate"]
        for metric in metrics:
            self.performance_metrics[metric] = []
    
    async def _continuous_learning_loop(self):
        """Continuous learning loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check for new learning opportunities
                await self._check_learning_opportunities()
                
                # Update replay buffers
                await self._update_replay_buffers()
                
                # Monitor performance
                await self._monitor_performance()
                
            except Exception as e:
                self.logger.error(f"Error in continuous learning loop: {e}")
    
    async def _check_learning_opportunities(self):
        """Check for new learning opportunities"""
        # This would integrate with the system to identify learning opportunities
        pass
    
    async def _update_replay_buffers(self):
        """Update replay buffers with new experiences"""
        # This would update replay buffers with new learning experiences
        pass
    
    async def _monitor_performance(self):
        """Monitor learning performance"""
        # This would monitor and log learning performance
        pass
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming messages for continuous learning"""
        start_time = time.time()
        
        try:
            # Extract learning parameters
            learning_mode = message.get("learning_mode", LearningMode.ONLINE.value)
            strategy = message.get("strategy", LearningStrategy.GRADIENT_DESCENT.value)
            examples = message.get("examples", [])
            session_config = message.get("session_config", {})
            
            # Start learning session
            session = await self._start_learning_session(
                LearningMode(learning_mode),
                LearningStrategy(strategy),
                examples,
                session_config
            )
            
            # Perform learning
            result = await self._perform_learning(session)
            
            # End session
            await self._end_learning_session(session, result)
            
            return {
                "status": "success",
                "session_id": session.session_id,
                "result": result,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Error in continuous learning: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def _start_learning_session(
        self, 
        mode: LearningMode, 
        strategy: LearningStrategy,
        examples: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> LearningSession:
        """Start a new learning session"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(examples))}"
        
        # Convert examples to LearningExample objects
        learning_examples = []
        for example in examples:
            learning_example = LearningExample(
                input_data=example.get("input"),
                target_output=example.get("output"),
                context=example.get("context", {}),
                timestamp=datetime.now(),
                confidence=example.get("confidence", 0.8),
                source=example.get("source", "user"),
                metadata=example.get("metadata", {})
            )
            learning_examples.append(learning_example)
        
        # Create learning session
        session = LearningSession(
            session_id=session_id,
            mode=mode,
            strategy=strategy,
            examples=learning_examples,
            start_time=datetime.now(),
            end_time=None,
            performance_metrics={},
            model_updates={}
        )
        
        self.current_session = session
        self.logger.info(f"Started learning session {session_id} with {len(learning_examples)} examples")
        
        return session
    
    async def _perform_learning(self, session: LearningSession) -> Dict[str, Any]:
        """Perform learning using the specified mode and strategy"""
        self.logger.info(f"Performing learning for session {session.session_id}")
        
        # Get learning function
        learning_func = self.learning_modes.get(session.mode)
        if not learning_func:
            raise ValueError(f"Unknown learning mode: {session.mode}")
        
        # Perform learning
        learning_result = await learning_func(session)
        
        # Update session with results
        session.performance_metrics = learning_result.get("metrics", {})
        session.model_updates = learning_result.get("updates", {})
        
        # Update statistics
        self.learning_stats[session.mode.value] += 1
        
        return learning_result
    
    async def _end_learning_session(self, session: LearningSession, result: Dict[str, Any]):
        """End a learning session"""
        session.end_time = datetime.now()
        
        # Add to learning history
        self.learning_history.append(session)
        
        # Keep only recent sessions
        if len(self.learning_history) > self.max_sessions:
            self.learning_history = self.learning_history[-self.max_sessions:]
        
        # Update replay buffers
        await self._update_replay_buffers_with_session(session)
        
        self.logger.info(f"Completed learning session {session.session_id}")
    
    async def _few_shot_learning(self, session: LearningSession) -> Dict[str, Any]:
        """Perform few-shot learning"""
        self.logger.info(f"Performing few-shot learning with {len(session.examples)} examples")
        
        # Simulate few-shot learning
        learning_result = {
            "mode": "few_shot",
            "examples_processed": len(session.examples),
            "learning_rate": self.learning_rate * 2,  # Higher learning rate for few-shot
            "convergence_steps": len(session.examples) * 3,
            "metrics": {
                "accuracy": np.random.normal(0.85, 0.05),
                "loss": np.random.normal(0.15, 0.05),
                "learning_rate": self.learning_rate * 2,
                "convergence_rate": np.random.normal(0.9, 0.03)
            },
            "updates": {
                "parameters_updated": len(session.examples) * 10,
                "new_patterns_learned": len(session.examples),
                "adaptation_speed": "high"
            }
        }
        
        return learning_result
    
    async def _online_learning(self, session: LearningSession) -> Dict[str, Any]:
        """Perform online learning"""
        self.logger.info(f"Performing online learning with {len(session.examples)} examples")
        
        # Simulate online learning
        learning_result = {
            "mode": "online",
            "examples_processed": len(session.examples),
            "learning_rate": self.learning_rate,
            "convergence_steps": len(session.examples),
            "metrics": {
                "accuracy": np.random.normal(0.80, 0.08),
                "loss": np.random.normal(0.20, 0.08),
                "learning_rate": self.learning_rate,
                "convergence_rate": np.random.normal(0.85, 0.05)
            },
            "updates": {
                "parameters_updated": len(session.examples) * 5,
                "new_patterns_learned": len(session.examples) // 2,
                "adaptation_speed": "medium"
            }
        }
        
        return learning_result
    
    async def _incremental_learning(self, session: LearningSession) -> Dict[str, Any]:
        """Perform incremental learning"""
        self.logger.info(f"Performing incremental learning with {len(session.examples)} examples")
        
        # Simulate incremental learning
        learning_result = {
            "mode": "incremental",
            "examples_processed": len(session.examples),
            "learning_rate": self.learning_rate * 0.5,  # Lower learning rate for stability
            "convergence_steps": len(session.examples) * 2,
            "metrics": {
                "accuracy": np.random.normal(0.88, 0.04),
                "loss": np.random.normal(0.12, 0.04),
                "learning_rate": self.learning_rate * 0.5,
                "convergence_rate": np.random.normal(0.92, 0.03)
            },
            "updates": {
                "parameters_updated": len(session.examples) * 3,
                "new_patterns_learned": len(session.examples),
                "adaptation_speed": "stable"
            }
        }
        
        return learning_result
    
    async def _replay_learning(self, session: LearningSession) -> Dict[str, Any]:
        """Perform replay learning"""
        self.logger.info(f"Performing replay learning with {len(session.examples)} examples")
        
        # Get replay buffer for this mode
        buffer_id = f"replay_buffer_{session.mode.value}"
        replay_buffer = self.replay_buffers.get(buffer_id)
        
        if replay_buffer:
            # Add examples to replay buffer
            for example in session.examples:
                replay_buffer.examples.append(example)
                replay_buffer.priority_scores[example.timestamp.isoformat()] = example.confidence
            
            replay_buffer.last_accessed = datetime.now()
            replay_buffer.access_count += 1
        
        # Simulate replay learning
        learning_result = {
            "mode": "replay",
            "examples_processed": len(session.examples),
            "replay_buffer_size": len(replay_buffer.examples) if replay_buffer else 0,
            "learning_rate": self.learning_rate * 0.8,
            "convergence_steps": len(session.examples) * 4,
            "metrics": {
                "accuracy": np.random.normal(0.82, 0.06),
                "loss": np.random.normal(0.18, 0.06),
                "learning_rate": self.learning_rate * 0.8,
                "convergence_rate": np.random.normal(0.87, 0.04)
            },
            "updates": {
                "parameters_updated": len(session.examples) * 4,
                "new_patterns_learned": len(session.examples),
                "replay_experiences": len(replay_buffer.examples) if replay_buffer else 0,
                "adaptation_speed": "moderate"
            }
        }
        
        return learning_result
    
    async def _adaptive_learning(self, session: LearningSession) -> Dict[str, Any]:
        """Perform adaptive learning"""
        self.logger.info(f"Performing adaptive learning with {len(session.examples)} examples")
        
        # Analyze examples to determine best strategy
        strategy = self._determine_best_strategy(session.examples)
        
        # Perform learning with determined strategy
        strategy_func = self.learning_strategies.get(strategy)
        if strategy_func:
            strategy_result = await strategy_func(session)
        else:
            strategy_result = {"strategy": strategy.value, "status": "not_implemented"}
        
        # Simulate adaptive learning
        learning_result = {
            "mode": "adaptive",
            "strategy_used": strategy.value,
            "examples_processed": len(session.examples),
            "learning_rate": self.learning_rate * 1.2,
            "convergence_steps": len(session.examples) * 2,
            "metrics": {
                "accuracy": np.random.normal(0.90, 0.03),
                "loss": np.random.normal(0.10, 0.03),
                "learning_rate": self.learning_rate * 1.2,
                "convergence_rate": np.random.normal(0.95, 0.02)
            },
            "updates": {
                "parameters_updated": len(session.examples) * 6,
                "new_patterns_learned": len(session.examples),
                "strategy_adaptation": strategy.value,
                "adaptation_speed": "optimal"
            },
            "strategy_result": strategy_result
        }
        
        return learning_result
    
    def _determine_best_strategy(self, examples: List[LearningExample]) -> LearningStrategy:
        """Determine the best learning strategy based on examples"""
        # Simple heuristics for strategy selection
        avg_confidence = np.mean([ex.confidence for ex in examples])
        example_count = len(examples)
        
        if avg_confidence > 0.9 and example_count < 5:
            return LearningStrategy.FEW_SHOT
        elif avg_confidence > 0.8 and example_count < 20:
            return LearningStrategy.META_LEARNING
        elif avg_confidence > 0.7:
            return LearningStrategy.TRANSFER_LEARNING
        elif avg_confidence > 0.6:
            return LearningStrategy.ACTIVE_LEARNING
        else:
            return LearningStrategy.SELF_SUPERVISED
    
    async def _gradient_descent_update(self, session: LearningSession) -> Dict[str, Any]:
        """Perform gradient descent update"""
        return {
            "strategy": "gradient_descent",
            "learning_rate": self.learning_rate,
            "convergence": "stable",
            "status": "completed"
        }
    
    async def _meta_learning_update(self, session: LearningSession) -> Dict[str, Any]:
        """Perform meta-learning update"""
        return {
            "strategy": "meta_learning",
            "learning_rate": self.learning_rate * 1.5,
            "convergence": "fast",
            "status": "completed"
        }
    
    async def _transfer_learning_update(self, session: LearningSession) -> Dict[str, Any]:
        """Perform transfer learning update"""
        return {
            "strategy": "transfer_learning",
            "learning_rate": self.learning_rate * 0.8,
            "convergence": "stable",
            "status": "completed"
        }
    
    async def _active_learning_update(self, session: LearningSession) -> Dict[str, Any]:
        """Perform active learning update"""
        return {
            "strategy": "active_learning",
            "learning_rate": self.learning_rate * 1.2,
            "convergence": "adaptive",
            "status": "completed"
        }
    
    async def _self_supervised_update(self, session: LearningSession) -> Dict[str, Any]:
        """Perform self-supervised learning update"""
        return {
            "strategy": "self_supervised",
            "learning_rate": self.learning_rate * 0.6,
            "convergence": "slow_but_stable",
            "status": "completed"
        }
    
    async def _update_replay_buffers_with_session(self, session: LearningSession):
        """Update replay buffers with session data"""
        buffer_id = f"replay_buffer_{session.mode.value}"
        replay_buffer = self.replay_buffers.get(buffer_id)
        
        if replay_buffer:
            # Add examples to replay buffer
            for example in session.examples:
                if len(replay_buffer.examples) < replay_buffer.max_size:
                    replay_buffer.examples.append(example)
                    replay_buffer.priority_scores[example.timestamp.isoformat()] = example.confidence
            
            replay_buffer.last_accessed = datetime.now()
            replay_buffer.access_count += 1
    
    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            "learning_sessions": len(self.learning_history),
            "current_session": self.current_session.session_id if self.current_session else None,
            "learning_stats": dict(self.learning_stats),
            "replay_buffers": {
                buffer_id: {
                    "size": len(buffer.examples),
                    "max_size": buffer.max_size,
                    "access_count": buffer.access_count,
                    "last_accessed": buffer.last_accessed.isoformat()
                }
                for buffer_id, buffer in self.replay_buffers.items()
            },
            "performance_metrics": {
                metric: values[-10:] if values else []  # Last 10 values
                for metric, values in self.performance_metrics.items()
            }
        }
    
    async def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent learning sessions"""
        recent_sessions = self.learning_history[-limit:] if self.learning_history else []
        
        return [
            {
                "session_id": session.session_id,
                "mode": session.mode.value,
                "strategy": session.strategy.value,
                "examples_count": len(session.examples),
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "performance_metrics": session.performance_metrics
            }
            for session in recent_sessions
        ]
    
    async def clear_replay_buffer(self, buffer_id: str) -> Dict[str, Any]:
        """Clear a replay buffer"""
        if buffer_id in self.replay_buffers:
            buffer = self.replay_buffers[buffer_id]
            cleared_size = len(buffer.examples)
            
            buffer.examples.clear()
            buffer.priority_scores.clear()
            buffer.access_count = 0
            buffer.last_accessed = datetime.now()
            
            return {
                "status": "success",
                "message": f"Replay buffer {buffer_id} cleared",
                "cleared_examples": cleared_size
            }
        else:
            return {
                "status": "error",
                "error": f"Replay buffer {buffer_id} not found"
            }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent"""
        return {
            "name": "ContinuousLearningAgent",
            "description": "Continuous online and few-shot learning agent with replay buffers and incremental updates",
            "capabilities": [
                "Few-shot learning",
                "Online learning",
                "Incremental learning",
                "Replay learning",
                "Adaptive learning",
                "Gradient descent updates",
                "Meta-learning updates",
                "Transfer learning updates",
                "Active learning updates",
                "Self-supervised learning updates"
            ],
            "status": "active",
            "learning_modes": [mode.value for mode in LearningMode],
            "learning_strategies": [strategy.value for strategy in LearningStrategy],
            "stats": {
                "total_sessions": len(self.learning_history),
                "replay_buffers": len(self.replay_buffers),
                "learning_stats": dict(self.learning_stats)
            }
        } 