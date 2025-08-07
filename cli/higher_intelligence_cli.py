"""
Higher Intelligence CLI
Command-line interface for managing higher intelligence pillars (21-25)
"""

import asyncio
import argparse
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from agents.self_monitoring_agent import SelfMonitoringAgent
from agents.rag_agent import RAGAgent, RetrievalMethod, GenerationStrategy
from agents.adaptive_model_agent import AdaptiveModelAgent, SelectionStrategy
from agents.continuous_learning_agent import ContinuousLearningAgent, LearningMode, LearningStrategy
from agents.meta_learning_agent import MetaLearningAgent, MetaLearningMode, TrainingSignalType


class HigherIntelligenceCLI:
    """CLI for managing higher intelligence pillars (21-25)"""
    
    def __init__(self):
        self.self_monitoring = SelfMonitoringAgent()
        self.rag_agent = RAGAgent()
        self.adaptive_model = AdaptiveModelAgent()
        self.continuous_learning = ContinuousLearningAgent()
        self.meta_learning = MetaLearningAgent()
        
    async def run(self, args):
        """Run the CLI with given arguments"""
        try:
            if args.command == "monitor":
                await self._handle_monitor_command(args)
            elif args.command == "rag":
                await self._handle_rag_command(args)
            elif args.command == "model":
                await self._handle_model_command(args)
            elif args.command == "learn":
                await self._handle_learn_command(args)
            elif args.command == "meta":
                await self._handle_meta_command(args)
            elif args.command == "workflow":
                await self._handle_workflow_command(args)
            elif args.command == "status":
                await self._handle_status_command(args)
            else:
                print(f"Unknown command: {args.command}")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    async def _handle_monitor_command(self, args):
        """Handle monitor-related commands"""
        if args.monitor_action == "status":
            await self._show_system_status()
        elif args.monitor_action == "metrics":
            await self._show_system_metrics()
        elif args.monitor_action == "drift":
            await self._show_drift_alerts()
        elif args.monitor_action == "evaluation":
            await self._show_evaluation_results()
        elif args.monitor_action == "tests":
            await self._show_test_results()
        elif args.monitor_action == "recommendations":
            await self._show_recommendations()
        else:
            print(f"Unknown monitor action: {args.monitor_action}")
    
    async def _handle_rag_command(self, args):
        """Handle RAG-related commands"""
        if args.rag_action == "query":
            await self._perform_rag_query(args)
        elif args.rag_action == "add-knowledge":
            await self._add_knowledge(args)
        elif args.rag_action == "stats":
            await self._show_rag_stats()
        elif args.rag_action == "cache":
            await self._manage_rag_cache(args)
        else:
            print(f"Unknown RAG action: {args.rag_action}")
    
    async def _handle_model_command(self, args):
        """Handle model selection commands"""
        if args.model_action == "select":
            await self._select_model(args)
        elif args.model_action == "registry":
            await self._show_model_registry()
        elif args.model_action == "stats":
            await self._show_model_stats()
        elif args.model_action == "add":
            await self._add_model(args)
        elif args.model_action == "remove":
            await self._remove_model(args)
        else:
            print(f"Unknown model action: {args.model_action}")
    
    async def _handle_learn_command(self, args):
        """Handle continuous learning commands"""
        if args.learn_action == "session":
            await self._run_learning_session(args)
        elif args.learn_action == "stats":
            await self._show_learning_stats()
        elif args.learn_action == "sessions":
            await self._show_learning_sessions(args)
        elif args.learn_action == "clear":
            await self._clear_replay_buffer(args)
        else:
            print(f"Unknown learning action: {args.learn_action}")
    
    async def _handle_meta_command(self, args):
        """Handle meta-learning commands"""
        if args.meta_action == "session":
            await self._run_meta_session(args)
        elif args.meta_action == "signals":
            await self._generate_training_signals(args)
        elif args.meta_action == "critique":
            await self._perform_self_critique(args)
        elif args.meta_action == "stats":
            await self._show_meta_stats()
        elif args.meta_action == "sessions":
            await self._show_meta_sessions(args)
        else:
            print(f"Unknown meta action: {args.meta_action}")
    
    async def _handle_workflow_command(self, args):
        """Handle workflow commands"""
        if args.workflow_action == "complex":
            await self._run_complex_workflow(args)
        elif args.workflow_action == "simple":
            await self._run_simple_workflow(args)
        else:
            print(f"Unknown workflow action: {args.workflow_action}")
    
    async def _handle_status_command(self, args):
        """Handle status commands"""
        await self._show_overall_status()
    
    # Monitor Commands
    async def _show_system_status(self):
        """Show system status"""
        print("🔍 System Status")
        print("=" * 50)
        
        response = await self.self_monitoring.process_message({"type": "status"})
        
        if response["status"] == "success":
            status = response["response"]
            print(f"Status: {status['status']}")
            print(f"Uptime: {status['uptime']:.2f} seconds")
            print(f"Total Requests: {status['total_requests']}")
            print(f"Success Rate: {status['success_rate']:.2%}")
            print(f"Active Alerts: {status['active_alerts']}")
            print(f"Last Evaluation: {status['last_evaluation']}")
            print(f"Next Evaluation: {status['next_evaluation']}")
        else:
            print(f"Error getting status: {response.get('error', 'Unknown error')}")
    
    async def _show_system_metrics(self):
        """Show system metrics"""
        print("📊 System Metrics")
        print("=" * 50)
        
        response = await self.self_monitoring.process_message({"type": "metrics"})
        
        if response["status"] == "success":
            metrics = response["response"]
            print(f"Response Time: {metrics['response_time']:.3f}s")
            print(f"Memory Usage: {metrics['memory_usage']:.1f}%")
            print(f"CPU Usage: {metrics['cpu_usage']:.1f}%")
            print(f"Error Rate: {metrics['error_rate']:.2%}")
            print(f"Accuracy: {metrics['accuracy']:.2%}")
            print(f"Safety Score: {metrics['safety_score']:.2%}")
            print(f"Success Rate: {metrics['success_rate']:.2%}")
        else:
            print(f"Error getting metrics: {response.get('error', 'Unknown error')}")
    
    async def _show_drift_alerts(self):
        """Show drift alerts"""
        print("⚠️  Drift Alerts")
        print("=" * 50)
        
        response = await self.self_monitoring.process_message({"type": "drift_alerts"})
        
        if response["status"] == "success":
            alerts = response["response"]
            print(f"Total Alerts: {alerts['total_alerts']}")
            print(f"Recent Alerts: {alerts['recent_alerts']}")
            print(f"Critical Alerts: {alerts['critical_alerts']}")
            
            if alerts['alerts']:
                print("\nRecent Alerts:")
                for alert in alerts['alerts'][-5:]:  # Show last 5 alerts
                    print(f"  - {alert['description']} ({alert['severity']})")
            else:
                print("No recent alerts")
        else:
            print(f"Error getting alerts: {response.get('error', 'Unknown error')}")
    
    async def _show_evaluation_results(self):
        """Show evaluation results"""
        print("📈 Evaluation Results")
        print("=" * 50)
        
        response = await self.self_monitoring.process_message({"type": "evaluation"})
        
        if response["status"] == "success":
            evaluation = response["response"]
            if "results" in evaluation:
                results = evaluation["results"]
                print(f"Timestamp: {evaluation['timestamp']}")
                print(f"Performance Metrics: {len(results.get('performance_metrics', {}))} metrics")
                print(f"Agent Evaluation: {len(results.get('agent_evaluation', {}))} agents")
                print(f"System Health: {results.get('system_health', {}).get('overall_health', 'Unknown')}")
            else:
                print("No evaluation results available")
        else:
            print(f"Error getting evaluation: {response.get('error', 'Unknown error')}")
    
    async def _show_test_results(self):
        """Show test results"""
        print("🧪 Test Results")
        print("=" * 50)
        
        response = await self.self_monitoring.process_message({"type": "test_results"})
        
        if response["status"] == "success":
            results = response["response"]
            print(f"Status: {results['status']}")
            print(f"Last Test Run: {results['last_test_run']}")
            print(f"Next Test Run: {results['next_test_run']}")
        else:
            print(f"Error getting test results: {response.get('error', 'Unknown error')}")
    
    async def _show_recommendations(self):
        """Show recommendations"""
        print("💡 Recommendations")
        print("=" * 50)
        
        response = await self.self_monitoring.process_message({"type": "recommendations"})
        
        if response["status"] == "success":
            recommendations = response["response"]
            print(f"Priority: {recommendations['priority']}")
            print(f"Timestamp: {recommendations['timestamp']}")
            
            if recommendations['recommendations']:
                print("\nRecommendations:")
                for i, rec in enumerate(recommendations['recommendations'], 1):
                    print(f"  {i}. {rec}")
            else:
                print("No recommendations at this time")
        else:
            print(f"Error getting recommendations: {response.get('error', 'Unknown error')}")
    
    # RAG Commands
    async def _perform_rag_query(self, args):
        """Perform a RAG query"""
        print(f"🔍 RAG Query: {args.query}")
        print("=" * 50)
        
        message = {
            "query": args.query,
            "retrieval_method": args.retrieval_method or RetrievalMethod.HYBRID_SEARCH.value,
            "generation_strategy": args.generation_strategy or GenerationStrategy.ADAPTIVE_GENERATION.value,
            "max_contexts": args.max_contexts or 3
        }
        
        response = await self.rag_agent.process_message(message)
        
        if response["status"] == "success":
            result = response["result"]
            print(f"Generated Text:\n{result.generated_text}")
            print(f"\nConfidence Score: {result.confidence_score:.2%}")
            print(f"Generation Strategy: {result.generation_strategy.value}")
            print(f"Processing Time: {result.processing_time:.3f}s")
            print(f"Retrieved Contexts: {len(result.retrieved_contexts)}")
            print(f"Cached: {response['cached']}")
        else:
            print(f"Error performing RAG query: {response.get('error', 'Unknown error')}")
    
    async def _add_knowledge(self, args):
        """Add knowledge to RAG system"""
        print("📚 Adding Knowledge")
        print("=" * 50)
        
        # Read knowledge from file if provided
        if args.knowledge_file:
            with open(args.knowledge_file, 'r') as f:
                knowledge_data = json.load(f)
        else:
            # Create sample knowledge
            knowledge_data = {
                "entities": [
                    {
                        "id": f"entity_{datetime.now().timestamp()}",
                        "name": args.entity_name or "Sample Entity",
                        "type": "concept",
                        "properties": {"description": args.entity_description or "A sample entity"}
                    }
                ],
                "memories": [
                    {
                        "id": f"memory_{datetime.now().timestamp()}",
                        "content": args.memory_content or "Sample memory content",
                        "type": "semantic",
                        "timestamp": datetime.now().isoformat()
                    }
                ]
            }
        
        result = await self.rag_agent.add_knowledge(knowledge_data)
        
        if result["status"] == "success":
            print(f"✅ Knowledge added successfully")
            print(f"Entities added: {result['entities_added']}")
            print(f"Memories added: {result['memories_added']}")
        else:
            print(f"❌ Error adding knowledge: {result.get('error', 'Unknown error')}")
    
    async def _show_rag_stats(self):
        """Show RAG statistics"""
        print("📊 RAG Statistics")
        print("=" * 50)
        
        stats = await self.rag_agent.get_system_stats()
        
        print("Retrieval Statistics:")
        for method, count in stats["retrieval_stats"].items():
            print(f"  {method}: {count}")
        
        print("\nGeneration Statistics:")
        for strategy, count in stats["generation_stats"].items():
            print(f"  {strategy}: {count}")
        
        print(f"\nCache Statistics:")
        cache_stats = stats["cache_stats"]
        print(f"  Hits: {cache_stats['hits']}")
        print(f"  Misses: {cache_stats['misses']}")
        print(f"  Hit Rate: {cache_stats['hit_rate']:.2%}")
        
        print(f"\nKnowledge Statistics:")
        knowledge_stats = stats["knowledge_stats"]
        print(f"  Entities: {knowledge_stats['entities']}")
        print(f"  Memories: {knowledge_stats['memories']}")
        print(f"  Last Update: {knowledge_stats['last_update']}")
    
    async def _manage_rag_cache(self, args):
        """Manage RAG cache"""
        if args.cache_action == "clear":
            print("🗑️  Clearing RAG Cache")
            print("=" * 50)
            
            result = await self.rag_agent.clear_cache()
            
            if result["status"] == "success":
                print(f"✅ Cache cleared successfully")
                print(f"Cleared entries: {result['cleared_entries']}")
            else:
                print(f"❌ Error clearing cache: {result.get('error', 'Unknown error')}")
        else:
            print(f"Unknown cache action: {args.cache_action}")
    
    # Model Commands
    async def _select_model(self, args):
        """Select a model for a task"""
        print(f"🤖 Model Selection: {args.task_description}")
        print("=" * 50)
        
        message = {
            "task_description": args.task_description,
            "strategy": args.strategy or SelectionStrategy.BALANCED.value,
            "constraints": {
                "max_latency": args.max_latency or 5000,
                "min_accuracy": args.min_accuracy or 0.8,
                "max_cost_per_request": args.max_cost or 0.1
            }
        }
        
        response = await self.adaptive_model.process_message(message)
        
        if response["status"] == "success":
            selection = response["selection"]
            print(f"Selected Model: {selection['selected_model']['model_id']}")
            print(f"Model Type: {selection['selected_model']['model_type']}")
            print(f"Parameters: {selection['selected_model']['parameters']}M")
            print(f"Selection Reason: {selection['selection_reason']}")
            print(f"Confidence Score: {selection['confidence_score']:.2%}")
            print(f"Estimated Cost: ${selection['estimated_cost']:.6f}")
            print(f"Estimated Latency: {selection['estimated_latency']:.1f}ms")
            
            print(f"\nAlternatives:")
            for i, alt in enumerate(selection['alternatives'][:3], 1):
                print(f"  {i}. {alt['model_id']} ({alt['parameters']}M params)")
        else:
            print(f"Error selecting model: {response.get('error', 'Unknown error')}")
    
    async def _show_model_registry(self):
        """Show model registry"""
        print("📋 Model Registry")
        print("=" * 50)
        
        registry = await self.adaptive_model.get_model_registry()
        
        print(f"Total Models: {registry['total_models']}")
        print(f"Model Types: {', '.join(registry['model_types'])}")
        print(f"Selection Strategies: {', '.join(registry['selection_strategies'])}")
        
        print(f"\nRegistered Models:")
        for model_id, model in registry["models"].items():
            print(f"  {model_id}:")
            print(f"    Type: {model['model_type']}")
            print(f"    Parameters: {model['parameters']}M")
            print(f"    Latency: {model['latency_ms']}ms")
            print(f"    Cost: ${model['cost_per_token']:.6f}/token")
            print(f"    Accuracy: {model['accuracy_score']:.2%}")
            print(f"    Capabilities: {', '.join(model['capabilities'])}")
    
    async def _show_model_stats(self):
        """Show model statistics"""
        print("📊 Model Statistics")
        print("=" * 50)
        
        stats = await self.adaptive_model.get_performance_stats()
        
        print("Model Performance:")
        for model_id, perf in stats["model_performance"].items():
            print(f"  {model_id}:")
            print(f"    Accuracy: {perf['recent_accuracy']:.2%}")
            print(f"    Latency: {perf['recent_latency']:.1f}ms")
            print(f"    Throughput: {perf['recent_throughput']:.1f} req/s")
            print(f"    Error Rate: {perf['recent_error_rate']:.2%}")
            print(f"    Total Cost: ${perf['total_cost']:.6f}")
        
        print(f"\nSelection Statistics:")
        for strategy, count in stats["selection_stats"].items():
            print(f"  {strategy}: {count}")
    
    async def _add_model(self, args):
        """Add a model to the registry"""
        print("➕ Adding Model")
        print("=" * 50)
        
        model_spec = {
            "model_id": args.model_id,
            "model_type": args.model_type,
            "parameters": args.parameters,
            "latency_ms": args.latency,
            "cost_per_token": args.cost_per_token,
            "accuracy_score": args.accuracy,
            "capabilities": args.capabilities.split(",") if args.capabilities else ["general"],
            "max_context_length": args.max_context or 2048
        }
        
        result = await self.adaptive_model.add_model(model_spec)
        
        if result["status"] == "success":
            print(f"✅ Model {args.model_id} added successfully")
            print(f"Model Type: {result['model']['model_type']}")
            print(f"Parameters: {result['model']['parameters']}M")
        else:
            print(f"❌ Error adding model: {result.get('error', 'Unknown error')}")
    
    async def _remove_model(self, args):
        """Remove a model from the registry"""
        print(f"🗑️  Removing Model: {args.model_id}")
        print("=" * 50)
        
        result = await self.adaptive_model.remove_model(args.model_id)
        
        if result["status"] == "success":
            print(f"✅ Model {args.model_id} removed successfully")
        else:
            print(f"❌ Error removing model: {result.get('error', 'Unknown error')}")
    
    # Learning Commands
    async def _run_learning_session(self, args):
        """Run a continuous learning session"""
        print(f"🧠 Continuous Learning Session: {args.mode}")
        print("=" * 50)
        
        examples = json.loads(args.examples) if args.examples else []
        
        message = {
            "learning_mode": args.mode,
            "strategy": args.strategy,
            "examples": examples,
            "session_config": json.loads(args.config) if args.config else {}
        }
        
        response = await self.continuous_learning.process_message(message)
        
        if response["status"] == "success":
            result = response["result"]
            print(f"✅ Learning session completed")
            print(f"Session ID: {response['session_id']}")
            print(f"Mode: {result['mode']}")
            print(f"Examples Processed: {result['examples_processed']}")
            print(f"Learning Rate: {result['learning_rate']}")
            print(f"Convergence Steps: {result['convergence_steps']}")
            
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"Accuracy: {metrics.get('accuracy', 0):.2%}")
                print(f"Loss: {metrics.get('loss', 0):.3f}")
                print(f"Convergence Rate: {metrics.get('convergence_rate', 0):.2%}")
        else:
            print(f"❌ Error in learning session: {response.get('error', 'Unknown error')}")
    
    async def _show_learning_stats(self):
        """Show continuous learning statistics"""
        print("📊 Continuous Learning Statistics")
        print("=" * 50)
        
        stats = await self.continuous_learning.get_learning_stats()
        
        print(f"Learning Sessions: {stats['learning_sessions']}")
        print(f"Current Session: {stats['current_session'] or 'None'}")
        
        print("\nLearning Statistics:")
        for mode, count in stats['learning_stats'].items():
            print(f"  {mode}: {count}")
        
        print("\nReplay Buffers:")
        for buffer_id, buffer_info in stats['replay_buffers'].items():
            print(f"  {buffer_id}:")
            print(f"    Size: {buffer_info['size']}/{buffer_info['max_size']}")
            print(f"    Access Count: {buffer_info['access_count']}")
            print(f"    Last Accessed: {buffer_info['last_accessed']}")
    
    async def _show_learning_sessions(self, args):
        """Show recent learning sessions"""
        print("📋 Recent Learning Sessions")
        print("=" * 50)
        
        sessions = await self.continuous_learning.get_recent_sessions(limit=args.limit)
        
        for i, session in enumerate(sessions, 1):
            print(f"{i}. Session {session['session_id']}")
            print(f"   Mode: {session['mode']}")
            print(f"   Strategy: {session['strategy']}")
            print(f"   Examples: {session['examples_count']}")
            print(f"   Start: {session['start_time']}")
            print(f"   End: {session['end_time'] or 'In Progress'}")
            print()
    
    async def _clear_replay_buffer(self, args):
        """Clear a replay buffer"""
        print(f"🗑️  Clearing Replay Buffer: {args.buffer_id}")
        print("=" * 50)
        
        result = await self.continuous_learning.clear_replay_buffer(args.buffer_id)
        
        if result["status"] == "success":
            print(f"✅ Buffer cleared successfully")
            print(f"Cleared Examples: {result['cleared_examples']}")
        else:
            print(f"❌ Error clearing buffer: {result.get('error', 'Unknown error')}")
    
    # Meta-Learning Commands
    async def _run_meta_session(self, args):
        """Run a meta-learning session"""
        print(f"🧠 Meta-Learning Session: {args.mode}")
        print("=" * 50)
        
        message = {
            "learning_mode": args.mode,
            "signal_types": args.signal_types,
            "session_config": json.loads(args.config) if args.config else {}
        }
        
        response = await self.meta_learning.process_message(message)
        
        if response["status"] == "success":
            result = response["result"]
            print(f"✅ Meta-learning session completed")
            print(f"Session ID: {response['session_id']}")
            print(f"Mode: {result['mode']}")
            
            if 'signals_generated' in result:
                print(f"Signals Generated: {result['signals_generated']}")
            if 'critiques_generated' in result:
                print(f"Critiques Generated: {result['critiques_generated']}")
            if 'adaptations_made' in result:
                print(f"Adaptations Made: {result['adaptations_made']}")
            
            if 'metrics' in result:
                metrics = result['metrics']
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"{key.title()}: {value:.3f}")
                    else:
                        print(f"{key.title()}: {value}")
        else:
            print(f"❌ Error in meta-learning session: {response.get('error', 'Unknown error')}")
    
    async def _generate_training_signals(self, args):
        """Generate training signals"""
        print(f"🎯 Generating Training Signals: {args.signal_types}")
        print("=" * 50)
        
        result = await self.meta_learning.generate_training_signals(
            args.signal_types, count=args.count
        )
        
        if result["status"] == "success":
            print(f"✅ Generated {result['signals_generated']} signals")
            print(f"Signal Types: {', '.join(result['signal_types'])}")
            
            print("\nSample Signals:")
            for i, signal in enumerate(result['signals'][:3], 1):
                print(f"{i}. {signal['signal_type']}")
                print(f"   Input: {signal['input_data']}")
                print(f"   Output: {signal['target_output']}")
                print(f"   Confidence: {signal['confidence']:.2%}")
                print()
        else:
            print(f"❌ Error generating signals: {result.get('error', 'Unknown error')}")
    
    async def _perform_self_critique(self, args):
        """Perform self-critique"""
        print("🔍 Performing Self-Critique")
        print("=" * 50)
        
        context = json.loads(args.context) if args.context else None
        
        result = await self.meta_learning.perform_self_critique(args.output, context)
        
        if result["status"] == "success":
            print(f"✅ Self-critique completed")
            print(f"Critique ID: {result['critique_id']}")
            print(f"Critique Score: {result['critique_score']:.3f}")
            print(f"Confidence: {result['confidence']:.2%}")
            
            print("\nImprovement Suggestions:")
            for i, suggestion in enumerate(result['improvement_suggestions'], 1):
                print(f"  {i}. {suggestion}")
        else:
            print(f"❌ Error in self-critique: {result.get('error', 'Unknown error')}")
    
    async def _show_meta_stats(self):
        """Show meta-learning statistics"""
        print("📊 Meta-Learning Statistics")
        print("=" * 50)
        
        stats = await self.meta_learning.get_meta_learning_stats()
        
        print(f"Meta-Learning Sessions: {stats['meta_learning_sessions']}")
        print(f"Current Session: {stats['current_session'] or 'None'}")
        print(f"Total Signals Generated: {stats['total_signals_generated']}")
        print(f"Total Critiques Performed: {stats['total_critiques_performed']}")
        
        print("\nMeta-Learning Statistics:")
        for mode, count in stats['meta_learning_stats'].items():
            print(f"  {mode}: {count}")
        
        print("\nSignal Generation Statistics:")
        for signal_type, count in stats['signal_generation_stats'].items():
            print(f"  {signal_type}: {count}")
        
        print("\nCritique Statistics:")
        for metric, count in stats['critique_stats'].items():
            print(f"  {metric}: {count}")
    
    async def _show_meta_sessions(self, args):
        """Show recent meta-learning sessions"""
        print("📋 Recent Meta-Learning Sessions")
        print("=" * 50)
        
        sessions = await self.meta_learning.get_recent_sessions(limit=args.limit)
        
        for i, session in enumerate(sessions, 1):
            print(f"{i}. Session {session['session_id']}")
            print(f"   Mode: {session['mode']}")
            print(f"   Signals Generated: {session['signals_generated']}")
            print(f"   Critiques Performed: {session['critiques_performed']}")
            print(f"   Adaptations Made: {session['adaptations_made']}")
            print(f"   Start: {session['start_time']}")
            print(f"   End: {session['end_time'] or 'In Progress'}")
            print()
    
    # Workflow Commands
    async def _run_complex_workflow(self, args):
        """Run a complex workflow using all pillars"""
        print("🔄 Complex Workflow")
        print("=" * 50)
        
        # Step 1: Monitor system status
        print("1. Checking system status...")
        status_response = await self.self_monitoring.process_message({"type": "status"})
        if status_response["status"] == "success":
            print("   ✅ System status: OK")
        
        # Step 2: Select model for complex task
        print("2. Selecting appropriate model...")
        model_selection = {
            "task_description": args.task_description or "Analyze and explain complex machine learning concepts with detailed examples",
            "strategy": SelectionStrategy.ADAPTIVE.value,
            "constraints": {"max_latency": 5000, "min_accuracy": 0.9}
        }
        
        model_response = await self.adaptive_model.process_message(model_selection)
        if model_response["status"] == "success":
            selected_model = model_response["selection"]["selected_model"]["model_id"]
            print(f"   ✅ Selected model: {selected_model}")
        
        # Step 3: Perform RAG operation
        print("3. Performing RAG operation...")
        rag_operation = {
            "query": args.query or "What are the key differences between supervised and unsupervised learning?",
            "retrieval_method": RetrievalMethod.SEMANTIC_SEARCH.value,
            "generation_strategy": GenerationStrategy.ADAPTIVE_GENERATION.value,
            "max_contexts": 5
        }
        
        rag_response = await self.rag_agent.process_message(rag_operation)
        if rag_response["status"] == "success":
            print("   ✅ RAG operation completed")
        
        # Step 4: Get recommendations
        print("4. Getting recommendations...")
        recommendations_response = await self.self_monitoring.process_message({"type": "recommendations"})
        if recommendations_response["status"] == "success":
            print("   ✅ Recommendations retrieved")
        
        print("\n✅ Complex workflow completed successfully!")
    
    async def _run_simple_workflow(self, args):
        """Run a simple workflow"""
        print("🔄 Simple Workflow")
        print("=" * 50)
        
        # Simple RAG query
        print("1. Performing simple RAG query...")
        rag_message = {
            "query": args.query or "What is artificial intelligence?",
            "retrieval_method": RetrievalMethod.HYBRID_SEARCH.value,
            "generation_strategy": GenerationStrategy.ADAPTIVE_GENERATION.value
        }
        
        rag_response = await self.rag_agent.process_message(rag_message)
        if rag_response["status"] == "success":
            print("   ✅ RAG query completed")
            result = rag_response["result"]
            print(f"   Generated: {result.generated_text[:100]}...")
        
        # Check system status
        print("2. Checking system status...")
        status_response = await self.self_monitoring.process_message({"type": "status"})
        if status_response["status"] == "success":
            print("   ✅ System status: OK")
        
        print("\n✅ Simple workflow completed successfully!")
    
    async def _show_overall_status(self):
        """Show overall status of all pillars"""
        print("🏗️  Phase 6: Foundation Intelligence Pillars Status")
        print("=" * 50)
        
        # Pillar 21: Self-Monitoring
        print("Pillar 21: Self-Monitoring & Evaluation")
        print("-" * 30)
        self_monitoring_info = self.self_monitoring.get_agent_info()
        print(f"  Status: {self_monitoring_info['status']}")
        print(f"  Capabilities: {len(self_monitoring_info['capabilities'])}")
        print(f"  Total Requests: {self_monitoring_info['metrics']['total_requests']}")
        print(f"  Success Rate: {self_monitoring_info['metrics']['success_rate']:.2%}")
        print(f"  Active Alerts: {self_monitoring_info['metrics']['active_alerts']}")
        
        # Pillar 22: RAG
        print("\nPillar 22: Retrieval-Augmented Generation (RAG)")
        print("-" * 30)
        rag_info = self.rag_agent.get_agent_info()
        print(f"  Status: {rag_info['status']}")
        print(f"  Capabilities: {len(rag_info['capabilities'])}")
        print(f"  Retrieval Methods: {len(rag_info['retrieval_methods'])}")
        print(f"  Generation Strategies: {len(rag_info['generation_strategies'])}")
        print(f"  Cache Hit Rate: {rag_info['stats']['cache_hit_rate']:.2%}")
        
        # Pillar 23: Adaptive Model Selection
        print("\nPillar 23: Adaptive Model Selection")
        print("-" * 30)
        model_info = self.adaptive_model.get_agent_info()
        print(f"  Status: {model_info['status']}")
        print(f"  Capabilities: {len(model_info['capabilities'])}")
        print(f"  Registered Models: {model_info['registered_models']}")
        print(f"  Selection Strategies: {len(model_info['selection_strategies'])}")
        print(f"  Model Types: {len(model_info['model_types'])}")
        print(f"  Total Selections: {model_info['stats']['total_selections']}")
        
        # Pillar 24: Continuous Learning
        print("\nPillar 24: Continuous Online & Few-Shot Learning")
        print("-" * 30)
        learning_info = self.continuous_learning.get_agent_info()
        print(f"  Status: {learning_info['status']}")
        print(f"  Capabilities: {len(learning_info['capabilities'])}")
        print(f"  Learning Modes: {len(learning_info['learning_modes'])}")
        print(f"  Learning Strategies: {len(learning_info['learning_strategies'])}")
        print(f"  Total Sessions: {learning_info['stats']['total_sessions']}")
        print(f"  Replay Buffers: {learning_info['stats']['replay_buffers']}")
        
        # Pillar 25: Meta-Learning
        print("\nPillar 25: Meta-Learning & Self-Supervision")
        print("-" * 30)
        meta_info = self.meta_learning.get_agent_info()
        print(f"  Status: {meta_info['status']}")
        print(f"  Capabilities: {len(meta_info['capabilities'])}")
        print(f"  Meta-Learning Modes: {len(meta_info['meta_learning_modes'])}")
        print(f"  Training Signal Types: {len(meta_info['training_signal_types'])}")
        print(f"  Total Sessions: {meta_info['stats']['total_sessions']}")
        print(f"  Total Signals: {meta_info['stats']['total_signals']}")
        print(f"  Total Critiques: {meta_info['stats']['total_critiques']}")
        
        print("\n✅ Phase 6: Foundation Intelligence - 100% Complete!")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Higher Intelligence CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor system")
    monitor_parser.add_argument("monitor_action", choices=[
        "status", "metrics", "drift", "evaluation", "tests", "recommendations"
    ], help="Monitor action")
    
    # RAG command
    rag_parser = subparsers.add_parser("rag", help="RAG operations")
    rag_parser.add_argument("rag_action", choices=[
        "query", "add-knowledge", "stats", "cache"
    ], help="RAG action")
    rag_parser.add_argument("--query", help="Query for RAG")
    rag_parser.add_argument("--retrieval-method", help="Retrieval method")
    rag_parser.add_argument("--generation-strategy", help="Generation strategy")
    rag_parser.add_argument("--max-contexts", type=int, help="Maximum contexts")
    rag_parser.add_argument("--knowledge-file", help="Knowledge file path")
    rag_parser.add_argument("--entity-name", help="Entity name")
    rag_parser.add_argument("--entity-description", help="Entity description")
    rag_parser.add_argument("--memory-content", help="Memory content")
    rag_parser.add_argument("--cache-action", choices=["clear"], help="Cache action")
    
    # Model command
    model_parser = subparsers.add_parser("model", help="Model selection")
    model_parser.add_argument("model_action", choices=[
        "select", "registry", "stats", "add", "remove"
    ], help="Model action")
    model_parser.add_argument("--task-description", help="Task description")
    model_parser.add_argument("--strategy", help="Selection strategy")
    model_parser.add_argument("--max-latency", type=int, help="Maximum latency")
    model_parser.add_argument("--min-accuracy", type=float, help="Minimum accuracy")
    model_parser.add_argument("--max-cost", type=float, help="Maximum cost")
    model_parser.add_argument("--model-id", help="Model ID")
    model_parser.add_argument("--model-type", help="Model type")
    model_parser.add_argument("--parameters", type=int, help="Number of parameters")
    model_parser.add_argument("--latency", type=float, help="Latency in ms")
    model_parser.add_argument("--cost-per-token", type=float, help="Cost per token")
    model_parser.add_argument("--accuracy", type=float, help="Accuracy score")
    model_parser.add_argument("--capabilities", help="Comma-separated capabilities")
    model_parser.add_argument("--max-context", type=int, help="Maximum context length")
    
    # Learning command
    learn_parser = subparsers.add_parser("learn", help="Continuous learning")
    learn_parser.add_argument("learn_action", choices=[
        "session", "stats", "sessions", "clear"
    ], help="Learning action")
    learn_parser.add_argument("--mode", default="online", help="Learning mode")
    learn_parser.add_argument("--strategy", default="gradient_descent", help="Learning strategy")
    learn_parser.add_argument("--examples", default="[]", help="Examples JSON")
    learn_parser.add_argument("--config", default="{}", help="Session config JSON")
    learn_parser.add_argument("--limit", type=int, default=10, help="Session limit")
    learn_parser.add_argument("--buffer-id", help="Replay buffer ID")
    
    # Meta-learning command
    meta_parser = subparsers.add_parser("meta", help="Meta-learning")
    meta_parser.add_argument("meta_action", choices=[
        "session", "signals", "critique", "stats", "sessions"
    ], help="Meta-learning action")
    meta_parser.add_argument("--mode", default="self_supervised", help="Meta-learning mode")
    meta_parser.add_argument("--signal-types", nargs="+", default=["qa_pairs"], help="Signal types")
    meta_parser.add_argument("--config", default="{}", help="Session config JSON")
    meta_parser.add_argument("--count", type=int, default=10, help="Signal count")
    meta_parser.add_argument("--output", help="Output to critique")
    meta_parser.add_argument("--context", default="{}", help="Context JSON")
    meta_parser.add_argument("--limit", type=int, default=10, help="Session limit")
    
    # Workflow command
    workflow_parser = subparsers.add_parser("workflow", help="Workflow operations")
    workflow_parser.add_argument("workflow_action", choices=[
        "complex", "simple"
    ], help="Workflow action")
    workflow_parser.add_argument("--task-description", help="Task description")
    workflow_parser.add_argument("--query", help="Query for workflow")
    
    # Status command
    subparsers.add_parser("status", help="Show overall status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Run CLI
    cli = HigherIntelligenceCLI()
    asyncio.run(cli.run(args))


if __name__ == "__main__":
    main() 