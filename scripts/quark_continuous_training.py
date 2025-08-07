#!/usr/bin/env python3
"""
Quark Continuous Training Orchestrator
=====================================

Orchestrates Quark's continuous training, dataset discovery, and self-improvement system.
This script enables Quark to search for its own datasets and continuously improve its intelligence.
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.dataset_discovery_agent import DatasetDiscoveryAgent
from agents.continuous_training_agent import ContinuousTrainingAgent
from agents.self_improvement_agent import SelfImprovementAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuarkContinuousTrainingOrchestrator:
    """Orchestrates Quark's continuous training and self-improvement system."""
    
    def __init__(self):
        self.dataset_discovery_agent = None
        self.continuous_training_agent = None
        self.self_improvement_agent = None
        
        # Training configuration
        self.training_config = {
            'auto_discovery_interval': 24 * 60 * 60,  # 24 hours
            'training_interval': 12 * 60 * 60,  # 12 hours
            'improvement_check_interval': 6 * 60 * 60,  # 6 hours
            'max_concurrent_training_sessions': 3,
            'min_dataset_quality_threshold': 0.7,
            'min_dataset_relevance_threshold': 0.6,
            'max_datasets_per_training_session': 5
        }
        
        # Performance tracking
        self.training_history = []
        self.discovery_history = []
        self.improvement_history = []
        
        # Background threads
        self.discovery_thread = None
        self.training_thread = None
        self.improvement_thread = None
        self.running = False
        
        # Initialize agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all training and improvement agents."""
        logger.info("🔧 Initializing Quark Continuous Training Orchestrator...")
        
        try:
            # Initialize dataset discovery agent
            self.dataset_discovery_agent = DatasetDiscoveryAgent()
            self.dataset_discovery_agent.load_model()
            logger.info("✅ Dataset Discovery Agent initialized")
            
            # Initialize continuous training agent
            self.continuous_training_agent = ContinuousTrainingAgent()
            self.continuous_training_agent.load_model()
            logger.info("✅ Continuous Training Agent initialized")
            
            # Initialize self-improvement agent
            self.self_improvement_agent = SelfImprovementAgent()
            self.self_improvement_agent.load_model()
            logger.info("✅ Self-Improvement Agent initialized")
            
            # Link agents together
            self._link_agents()
            
            logger.info("🎯 All agents initialized and linked successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing agents: {e}")
            raise e
    
    def _link_agents(self):
        """Link agents together for coordinated operation."""
        try:
            # Link dataset discovery to continuous training
            if self.dataset_discovery_agent and self.continuous_training_agent:
                self.dataset_discovery_agent.continuous_training_agent = self.continuous_training_agent
            
            # Link continuous training to self-improvement
            if self.continuous_training_agent and self.self_improvement_agent:
                self.continuous_training_agent.self_improvement_agent = self.self_improvement_agent
            
            # Link self-improvement to dataset discovery
            if self.self_improvement_agent and self.dataset_discovery_agent:
                self.self_improvement_agent.dataset_discovery_agent = self.dataset_discovery_agent
                self.self_improvement_agent.continuous_training_agent = self.continuous_training_agent
            
            logger.info("🔗 Agents linked successfully")
            
        except Exception as e:
            logger.error(f"❌ Error linking agents: {e}")
            raise e
    
    def start_continuous_training(self):
        """Start the continuous training system."""
        logger.info("🚀 Starting Quark Continuous Training System...")
        
        self.running = True
        
        # Start background threads
        self._start_discovery_thread()
        self._start_training_thread()
        self._start_improvement_thread()
        
        logger.info("✅ Continuous training system started")
        logger.info("📊 Monitoring training progress...")
        
        # Main monitoring loop
        try:
            while self.running:
                self._monitor_training_progress()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("⏹️ Stopping continuous training system...")
            self.stop_continuous_training()
    
    def _start_discovery_thread(self):
        """Start dataset discovery thread."""
        self.discovery_thread = threading.Thread(
            target=self._discovery_loop,
            daemon=True
        )
        self.discovery_thread.start()
        logger.info("🔍 Dataset discovery thread started")
    
    def _start_training_thread(self):
        """Start continuous training thread."""
        self.training_thread = threading.Thread(
            target=self._training_loop,
            daemon=True
        )
        self.training_thread.start()
        logger.info("🏋️ Continuous training thread started")
    
    def _start_improvement_thread(self):
        """Start self-improvement thread."""
        self.improvement_thread = threading.Thread(
            target=self._improvement_loop,
            daemon=True
        )
        self.improvement_thread.start()
        logger.info("🧠 Self-improvement thread started")
    
    def _discovery_loop(self):
        """Background loop for dataset discovery."""
        while self.running:
            try:
                logger.info("🔍 Starting dataset discovery cycle...")
                
                # Discover datasets for Quark improvement
                discovery_result = self._discover_training_datasets()
                
                if discovery_result and "error" not in discovery_result:
                    self.discovery_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'datasets_found': discovery_result.get('total_discovered', 0),
                        'high_quality_count': discovery_result.get('high_quality_count', 0),
                        'datasets': discovery_result.get('datasets', [])
                    })
                    
                    logger.info(f"✅ Discovery cycle completed: {discovery_result.get('high_quality_count', 0)} high-quality datasets found")
                else:
                    logger.warning("⚠️ Dataset discovery cycle failed")
                
                # Wait for next discovery cycle
                time.sleep(self.training_config['auto_discovery_interval'])
                
            except Exception as e:
                logger.error(f"❌ Error in discovery loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _training_loop(self):
        """Background loop for continuous training."""
        while self.running:
            try:
                logger.info("🏋️ Starting continuous training cycle...")
                
                # Check if we have discovered datasets
                if self.discovery_history:
                    latest_discovery = self.discovery_history[-1]
                    datasets = latest_discovery.get('datasets', [])
                    
                    if datasets:
                        # Start training with discovered datasets
                        training_result = self._start_training_session(datasets)
                        
                        if training_result and "error" not in training_result:
                            self.training_history.append({
                                'timestamp': datetime.now().isoformat(),
                                'session_id': training_result.get('session_id'),
                                'improvement': training_result.get('performance_improvement', 0),
                                'datasets_used': len(datasets)
                            })
                            
                            logger.info(f"✅ Training cycle completed: {training_result.get('performance_improvement', 0):.3f} improvement")
                        else:
                            logger.warning("⚠️ Training cycle failed")
                    else:
                        logger.info("⏳ No datasets available for training")
                else:
                    logger.info("⏳ No discovery history available")
                
                # Wait for next training cycle
                time.sleep(self.training_config['training_interval'])
                
            except Exception as e:
                logger.error(f"❌ Error in training loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _improvement_loop(self):
        """Background loop for self-improvement."""
        while self.running:
            try:
                logger.info("🧠 Starting self-improvement cycle...")
                
                # Run self-reflection
                reflection_result = self.self_improvement_agent.run_self_reflection()
                
                if reflection_result and "error" not in reflection_result:
                    improvement_needed = reflection_result.get('improvement_needed', False)
                    
                    if improvement_needed:
                        # Run automated fine-tuning
                        fine_tuning_result = self.self_improvement_agent.run_automated_fine_tuning()
                        
                        if fine_tuning_result and "error" not in fine_tuning_result:
                            self.improvement_history.append({
                                'timestamp': datetime.now().isoformat(),
                                'improvement_type': 'fine_tuning',
                                'performance_gain': fine_tuning_result.get('performance_gain', 0)
                            })
                            
                            logger.info(f"✅ Self-improvement cycle completed: {fine_tuning_result.get('performance_gain', 0):.3f} gain")
                        else:
                            logger.warning("⚠️ Fine-tuning failed")
                    else:
                        logger.info("✅ No improvement needed at this time")
                else:
                    logger.warning("⚠️ Self-reflection failed")
                
                # Wait for next improvement cycle
                time.sleep(self.training_config['improvement_check_interval'])
                
            except Exception as e:
                logger.error(f"❌ Error in improvement loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _discover_training_datasets(self) -> Dict[str, Any]:
        """Discover training datasets for Quark improvement."""
        try:
            # Search for datasets relevant to Quark's capabilities
            search_queries = [
                "conversation ai training",
                "question answering datasets",
                "reasoning tasks datasets",
                "planning problems datasets",
                "creative writing datasets",
                "code generation datasets",
                "sentiment analysis datasets",
                "entity recognition datasets",
                "social intelligence datasets",
                "emotional intelligence datasets"
            ]
            
            all_datasets = []
            
            for query in search_queries:
                # Use dataset discovery agent to search
                search_result = self.dataset_discovery_agent.generate(
                    f"search for {query}",
                    categories=["conversation", "qa", "reasoning", "planning", "creative_writing"],
                    min_size=500,
                    max_results=5
                )
                
                if "datasets" in search_result:
                    all_datasets.extend(search_result["datasets"])
            
            # Filter high-quality datasets
            high_quality_datasets = [
                d for d in all_datasets
                if d.quality_score >= self.training_config['min_dataset_quality_threshold']
                and d.relevance_score >= self.training_config['min_dataset_relevance_threshold']
            ]
            
            return {
                "total_discovered": len(all_datasets),
                "high_quality_count": len(high_quality_datasets),
                "datasets": high_quality_datasets[:self.training_config['max_datasets_per_training_session']]
            }
            
        except Exception as e:
            logger.error(f"Error discovering datasets: {e}")
            return {"error": str(e)}
    
    def _start_training_session(self, datasets: List) -> Dict[str, Any]:
        """Start a training session with discovered datasets."""
        try:
            # Select best datasets for training
            selected_datasets = sorted(
                datasets,
                key=lambda x: x.quality_score * x.relevance_score,
                reverse=True
            )[:self.training_config['max_datasets_per_training_session']]
            
            dataset_ids = [d.id for d in selected_datasets]
            
            # Start training session
            training_result = self.continuous_training_agent.generate(
                "start training session",
                model_name="quark_core",
                dataset_ids=dataset_ids,
                strategy="incremental"
            )
            
            return training_result
            
        except Exception as e:
            logger.error(f"Error starting training session: {e}")
            return {"error": str(e)}
    
    def _monitor_training_progress(self):
        """Monitor and log training progress."""
        try:
            # Get current statistics
            discovery_stats = self.dataset_discovery_agent.get_discovery_statistics()
            training_stats = self.continuous_training_agent.get_training_statistics()
            improvement_stats = self.self_improvement_agent.get_learning_statistics()
            
            # Log progress
            logger.info("📊 Training Progress Summary:")
            logger.info(f"   🔍 Datasets discovered: {discovery_stats.get('total_searches', 0)}")
            logger.info(f"   🏋️ Training sessions: {training_stats.get('total_sessions', 0)}")
            logger.info(f"   🧠 Learning examples: {improvement_stats.get('total_examples', 0)}")
            
            # Check for recent improvements
            if self.training_history:
                recent_improvements = [
                    h for h in self.training_history[-5:]
                    if h.get('improvement', 0) > 0.01
                ]
                
                if recent_improvements:
                    avg_improvement = sum(h['improvement'] for h in recent_improvements) / len(recent_improvements)
                    logger.info(f"   📈 Recent average improvement: {avg_improvement:.3f}")
            
        except Exception as e:
            logger.error(f"Error monitoring training progress: {e}")
    
    def stop_continuous_training(self):
        """Stop the continuous training system."""
        logger.info("⏹️ Stopping continuous training system...")
        
        self.running = False
        
        # Wait for threads to finish
        if self.discovery_thread and self.discovery_thread.is_alive():
            self.discovery_thread.join(timeout=10)
        
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=10)
        
        if self.improvement_thread and self.improvement_thread.is_alive():
            self.improvement_thread.join(timeout=10)
        
        logger.info("✅ Continuous training system stopped")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of training activities."""
        return {
            "discovery_history": self.discovery_history,
            "training_history": self.training_history,
            "improvement_history": self.improvement_history,
            "total_datasets_discovered": sum(h.get('datasets_found', 0) for h in self.discovery_history),
            "total_training_sessions": len(self.training_history),
            "total_improvements": len(self.improvement_history),
            "average_improvement": sum(h.get('improvement', 0) for h in self.training_history) / max(len(self.training_history), 1),
            "system_status": "running" if self.running else "stopped"
        }


def main():
    """Main function to run the continuous training orchestrator."""
    print("🤖 Quark Continuous Training System")
    print("===================================")
    print("This system enables Quark to:")
    print("  🔍 Search for its own training datasets")
    print("  🏋️ Continuously train and improve")
    print("  🧠 Self-improve through reflection and learning")
    print("  📈 Monitor and optimize performance")
    print()
    
    # Create orchestrator
    orchestrator = QuarkContinuousTrainingOrchestrator()
    
    try:
        # Start continuous training
        orchestrator.start_continuous_training()
        
    except KeyboardInterrupt:
        print("\n⏹️ Stopping continuous training system...")
        orchestrator.stop_continuous_training()
        
        # Print summary
        summary = orchestrator.get_training_summary()
        print("\n📊 Training Summary:")
        print(f"   Datasets discovered: {summary['total_datasets_discovered']}")
        print(f"   Training sessions: {summary['total_training_sessions']}")
        print(f"   Improvements made: {summary['total_improvements']}")
        print(f"   Average improvement: {summary['average_improvement']:.3f}")
        
    except Exception as e:
        logger.error(f"❌ Error in continuous training system: {e}")
        orchestrator.stop_continuous_training()


if __name__ == "__main__":
    main() 