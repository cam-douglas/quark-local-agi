#!/usr/bin/env python3
"""
Quark Chat Interface
===================

Simple chat interface using the instant startup system.
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
        logging.FileHandler('logs/quark_chat.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QuarkChat:
    """Simple Quark chat interface."""
    
    def __init__(self):
        self.startup_time = time.time()
        self.response_agent = None
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
    def start(self):
        """Start the chat interface."""
        logger.info("🚀 Starting Quark Chat Interface...")
        
        try:
            # Initialize response agent
            logger.info("📦 Initializing response agent...")
            self.response_agent = ResponseGenerationAgent()
            
            startup_duration = time.time() - self.startup_time
            logger.info(f"✅ Quark ready in {startup_duration:.2f} seconds!")
            
            # Start interactive chat
            self.start_chat()
            
        except Exception as e:
            logger.error(f"❌ Chat startup failed: {e}")
            raise
    
    def start_chat(self):
        """Start the interactive chat."""
        logger.info("🎯 Starting Quark Chat Interface...")
        
        print("\n🤖 Welcome to Quark AI System!")
        print("=" * 50)
        print("Quark: Hello! I'm Quark, your AI assistant. I'm ready to help you with any task!")
        print("Quark: Type 'help' to see available commands, or just start chatting!")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nUser: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nQuark: 👋 Goodbye! Have a great day!")
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
                                print(f"\nQuark: {result['response']}")
                            else:
                                print(f"\nQuark: {result}")
                        except Exception as e:
                            print(f"\nQuark: Sorry, I encountered an error: {e}")
                    else:
                        print("\nQuark: Response agent not available")
                        
            except KeyboardInterrupt:
                print("\nQuark: 👋 Interrupted by user")
                break
            except EOFError:
                print("\nQuark: 👋 End of input")
                break
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down Quark Chat...")
        logger.info("✅ Quark Chat shutdown complete")

def main():
    """Main entry point"""
    chat = QuarkChat()
    
    # Set up signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        chat.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        chat.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        chat.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main() 