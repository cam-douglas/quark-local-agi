#!/usr/bin/env python3
"""
Quark Instant Startup
====================

Instant startup that provides immediate interactive access
without waiting for complex initialization.
"""

import os
import sys
import time
import signal
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.response_generation_agent import ResponseGenerationAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/quark_instant_startup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QuarkInstantStartup:
    """Instant Quark startup with immediate shell access."""
    
    def __init__(self):
        self.startup_time = time.time()
        self.is_ready = False
        self.response_agent = None
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
    def start(self):
        """Start Quark with instant access."""
        logger.info("🚀 Starting Quark Instant Startup...")
        
        try:
            # Initialize only the response generation agent
            logger.info("📦 Initializing response agent...")
            self.response_agent = ResponseGenerationAgent()
            
            # Mark as ready immediately
            self.is_ready = True
            startup_duration = time.time() - self.startup_time
            logger.info(f"✅ Quark ready in {startup_duration:.2f} seconds!")
            
            # Write ready flag
            with open('logs/quark_ready.flag', 'w') as f:
                f.write(str(int(time.time())))
            
            # Start interactive shell immediately
            self.start_interactive_shell()
            
        except Exception as e:
            logger.error(f"❌ Instant startup failed: {e}")
            raise
    
    def start_interactive_shell(self):
        """Start the interactive shell immediately"""
        logger.info("🎯 Starting Quark interactive shell...")
        
        print("\n🤖 Welcome to Quark AI System!")
        print("=" * 50)
        print("Quark is ready for interaction!")
        print("Type 'help' for commands, 'quit' to exit")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nQuark> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Quark shutting down...")
                    break
                elif user_input.lower() == 'help':
                    print("\n📋 Available Commands:")
                    print("  help - Show this help")
                    print("  status - Show system status")
                    print("  quit/exit - Exit Quark")
                    print("  <any text> - Chat with Quark")
                elif user_input.lower() == 'status':
                    startup_duration = time.time() - self.startup_time
                    print(f"\n📊 System Status: Ready")
                    print(f"Startup Time: {startup_duration:.2f}s")
                    print(f"Response Agent: {'✅' if self.response_agent else '❌'}")
                elif user_input:
                    # Process user input through response agent
                    if self.response_agent:
                        try:
                            result = self.response_agent.generate(user_input)
                            if isinstance(result, dict) and "response" in result:
                                print(f"\n🤖 Quark: {result['response']}")
                            else:
                                print(f"\n🤖 Quark: {result}")
                        except Exception as e:
                            print(f"\n❌ Error processing input: {e}")
                    else:
                        print("\n🤖 Quark: Response agent not available")
                        
            except KeyboardInterrupt:
                print("\n👋 Interrupted by user")
                break
            except EOFError:
                print("\n👋 End of input")
                break
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down Quark...")
        
        # Remove ready flag
        try:
            os.remove('logs/quark_ready.flag')
        except:
            pass
        
        logger.info("✅ Quark shutdown complete")

def main():
    """Main entry point"""
    startup = QuarkInstantStartup()
    
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