#!/usr/bin/env python3
"""
Quark Fast Startup
==================

Ultra-fast startup that provides immediate interactive access
while loading agents in the background.
"""

import os
import sys
import time
import signal
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.orchestrator import Orchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/quark_fast_startup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QuarkFastStartup:
    """Ultra-fast Quark startup with immediate shell access."""
    
    def __init__(self):
        self.startup_time = time.time()
        self.is_ready = False
        self.shutdown_event = threading.Event()
        self.orchestrator = None
        self.agents = {}
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
    def start(self):
        """Start Quark with immediate shell access."""
        logger.info("🚀 Starting Quark Fast Startup...")
        
        try:
            # Initialize orchestrator immediately
            logger.info("📦 Initializing orchestrator...")
            self.orchestrator = Orchestrator()
            
            # Mark as ready immediately
            self.is_ready = True
            startup_duration = time.time() - self.startup_time
            logger.info(f"✅ Quark ready in {startup_duration:.2f} seconds!")
            
            # Write ready flag
            with open('logs/quark_ready.flag', 'w') as f:
                f.write(str(int(time.time())))
            
            # Start background agent loading
            self._start_background_loading()
            
            # Start interactive shell immediately
            self.start_interactive_shell()
            
        except Exception as e:
            logger.error(f"❌ Fast startup failed: {e}")
            raise
    
    def _start_background_loading(self):
        """Load agents in background thread."""
        def load_agents():
            try:
                logger.info("🔄 Loading agents in background...")
                
                # Only load essential agents
                agent_classes = [
                    ("ResponseGenerationAgent", "response_generation"),
                    ("NLUAgent", "nlu"),
                    ("RetrievalAgent", "retrieval"),
                    ("ReasoningAgent", "reasoning")
                ]
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = []
                    for agent_name, name in agent_classes:
                        future = executor.submit(self._load_agent, agent_name, name)
                        futures.append(future)
                    
                    # Wait for agents with short timeout
                    for future in futures:
                        try:
                            future.result(timeout=5)  # 5 second timeout
                        except Exception as e:
                            logger.warning(f"Background agent loading: {e}")
                
                logger.info("✅ Background agent loading completed")
                
            except Exception as e:
                logger.error(f"❌ Background loading failed: {e}")
        
        # Start background loading thread
        thread = threading.Thread(target=load_agents, daemon=True)
        thread.start()
    
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
    
    def start_interactive_shell(self):
        """Start the interactive shell immediately"""
        logger.info("🎯 Starting Quark interactive shell...")
        
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
                    print("  quit/exit - Exit Quark")
                    print("  <any text> - Chat with Quark")
                elif user_input.lower() == 'status':
                    loaded_agents = sum(1 for agent in self.agents.values() if agent is not None)
                    total_agents = len(self.agents)
                    print(f"\n📊 System Status: {'Ready' if self.is_ready else 'Loading'}")
                    print(f"Loaded Agents: {loaded_agents}/{total_agents}")
                    print(f"Startup Time: {time.time() - self.startup_time:.2f}s")
                elif user_input.lower() == 'agents':
                    print("\n🤖 Loaded Agents:")
                    for name, agent in self.agents.items():
                        status = "✅" if agent else "❌"
                        print(f"  {status} {name}")
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
        
        # Remove ready flag
        try:
            os.remove('logs/quark_ready.flag')
        except:
            pass
        
        logger.info("✅ Quark shutdown complete")

def main():
    """Main entry point"""
    startup = QuarkFastStartup()
    
    # Set up signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        startup.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        startup.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        startup.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main() 