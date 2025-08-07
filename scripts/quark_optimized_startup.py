#!/usr/bin/env python3
"""
Quark Optimized Startup Script
Pre-loads all components efficiently and provides immediate terminal access.
"""

import os
import sys
import asyncio
import threading
import time
import signal
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import psutil

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import core components that are definitely available
from core.orchestrator import Orchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/quark_optimized_startup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QuarkOptimizedStartup:
    def __init__(self):
        self.orchestrator = None
        self.agents = {}
        self.startup_time = None
        self.is_ready = False
        self.shutdown_event = threading.Event()
        
    def preload_models_async(self):
        """Preload all models asynchronously in parallel"""
        logger.info("🚀 Starting optimized Quark startup...")
        self.startup_time = time.time()
        
        # Initialize core components
        try:
            logger.info("📦 Initializing core orchestrator...")
            self.orchestrator = Orchestrator()
            
            # Define agent classes to try loading (only essential ones for fast startup)
            agent_classes = [
                ("ResponseGenerationAgent", "response_generation"),
                ("NLUAgent", "nlu"),
                ("RetrievalAgent", "retrieval"),
                ("ReasoningAgent", "reasoning"),
                ("MemoryAgent", "memory"),
                ("MetricsAgent", "metrics")
            ]
            
            # Load agents in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for agent_name, name in agent_classes:
                    future = executor.submit(self._load_agent, agent_name, name)
                    futures.append(future)
                
                # Wait for all agents to load with shorter timeout
                for future in futures:
                    try:
                        future.result(timeout=10)  # 10 second timeout per agent
                    except Exception as e:
                        logger.warning(f"Agent loading timeout or error: {e}")
            
            # Note: Orchestrator already has agents registered in its __init__
            # No need to register agents here
            
            self.is_ready = True
            startup_duration = time.time() - self.startup_time
            logger.info(f"✅ Quark startup completed in {startup_duration:.2f} seconds")
            
            # Write ready flag
            with open('logs/quark_ready.flag', 'w') as f:
                f.write(str(int(time.time())))
                
        except Exception as e:
            logger.error(f"❌ Startup failed: {e}")
            raise
    
    def _load_agent(self, agent_name, name):
        """Load a single agent with error handling"""
        try:
            logger.info(f"🔄 Loading {name} agent...")
            
            # Dynamic import to handle missing agents gracefully
            try:
                module = __import__(f"agents.{name}_agent", fromlist=[agent_name])
                agent_class = getattr(module, agent_name)
                agent = agent_class()
                self.agents[name] = agent
                logger.info(f"✅ {name} agent loaded successfully")
                return agent
            except ImportError as e:
                logger.warning(f"⚠️ {name} agent not available: {e}")
                self.agents[name] = None
                return None
            except Exception as e:
                logger.error(f"❌ Failed to load {name} agent: {e}")
                self.agents[name] = None
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to load {name} agent: {e}")
            self.agents[name] = None
            return None
    
    def get_startup_status(self):
        """Get current startup status"""
        if not self.startup_time:
            return {"status": "not_started", "progress": 0}
        
        loaded_agents = sum(1 for agent in self.agents.values() if agent is not None)
        total_agents = len(self.agents)
        progress = (loaded_agents / total_agents) * 100 if total_agents > 0 else 0
        
        return {
            "status": "ready" if self.is_ready else "loading",
            "progress": progress,
            "loaded_agents": loaded_agents,
            "total_agents": total_agents,
            "startup_time": time.time() - self.startup_time if self.startup_time else 0
        }
    
    def start_interactive_shell(self):
        """Start the interactive shell immediately"""
        if not self.is_ready:
            logger.warning("⚠️ Quark not fully ready, starting shell anyway...")
        
        logger.info("🎯 Starting Quark interactive shell...")
        
        # Simple interactive loop
        print("\n🤖 Welcome to Quark AI System!")
        print("=" * 50)
        print("Quark is ready for interaction!")
        print("Type 'help' for commands, 'quit' to exit")
        print("=" * 50)
        
        while not self.shutdown_event.is_set():
            try:
                user_input = input("\nQuark> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Quark shutting down...")
                    break
                elif user_input.lower() == 'help':
                    print("\n📋 Available Commands:")
                    print("  help - Show this help")
                    print("  status - Show system status")
                    print("  agents - List loaded agents")
                    print("  memory - Show memory stats")
                    print("  metrics - Show performance metrics")
                    print("  quit/exit - Exit Quark")
                    print("  <any text> - Chat with Quark")
                elif user_input.lower() == 'status':
                    status = self.get_startup_status()
                    print(f"\n📊 System Status: {status['status']}")
                    print(f"Progress: {status['progress']:.1f}%")
                    print(f"Loaded Agents: {status['loaded_agents']}/{status['total_agents']}")
                    print(f"Startup Time: {status['startup_time']:.2f}s")
                elif user_input.lower() == 'agents':
                    print("\n🤖 Loaded Agents:")
                    for name, agent in self.agents.items():
                        status = "✅" if agent else "❌"
                        print(f"  {status} {name}")
                elif user_input.lower() == 'memory':
                    if 'memory' in self.agents and self.agents['memory']:
                        try:
                            stats = self.agents['memory'].get_memory_stats()
                            print(f"\n🧠 Memory Stats: {stats}")
                        except:
                            print("❌ Memory agent not available")
                    else:
                        print("❌ Memory agent not loaded")
                elif user_input.lower() == 'metrics':
                    if 'metrics' in self.agents and self.agents['metrics']:
                        try:
                            metrics = self.agents['metrics'].get_performance_summary()
                            print(f"\n📈 Performance Metrics: {metrics}")
                        except:
                            print("❌ Metrics agent not available")
                    else:
                        print("❌ Metrics agent not loaded")
                elif user_input:
                    # Process user input through orchestrator
                    if self.orchestrator:
                        try:
                            result = self.orchestrator.handle(user_input)
                            if hasattr(result, 'final_response'):
                                print(f"\n🤖 Quark: {result.final_response}")
                            elif hasattr(result, 'response'):
                                print(f"\n🤖 Quark: {result.response}")
                            else:
                                print(f"\n🤖 Quark: {result}")
                        except Exception as e:
                            print(f"\n❌ Error processing input: {e}")
                    else:
                        print("\n🤖 Quark: I'm still starting up, please wait...")
                        
            except KeyboardInterrupt:
                print("\n👋 Interrupted by user")
                break
            except EOFError:
                print("\n👋 End of input")
                break
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down Quark...")
        self.shutdown_event.set()
        
        # Clean up agents
        for name, agent in self.agents.items():
            if agent and hasattr(agent, 'shutdown'):
                try:
                    agent.shutdown()
                except:
                    pass
        
        # Remove ready flag
        try:
            os.remove('logs/quark_ready.flag')
        except:
            pass
        
        logger.info("✅ Quark shutdown complete")

def main():
    """Main entry point"""
    startup = QuarkOptimizedStartup()
    
    # Set up signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        startup.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start preloading in background thread
        preload_thread = threading.Thread(target=startup.preload_models_async)
        preload_thread.daemon = True
        preload_thread.start()
        
        # Start interactive shell immediately
        startup.start_interactive_shell()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        startup.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main() 