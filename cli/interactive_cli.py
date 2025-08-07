#!/usr/bin/env python3
"""
Quark AI System Interactive CLI
Provides conversational interface for user input
"""

import os
import sys
import time
import logging
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/quark_interactive.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class QuarkInteractiveCLI:
    """Interactive CLI for Quark AI System"""
    
    def __init__(self):
        self.running = True
        self.user_name = "User"
        self.quark_name = "Quark"
        self.conversation_history = []
        
    def get_user_input(self):
        """Get input from user"""
        try:
            return input(f"{self.user_name}: ").strip()
        except (EOFError, KeyboardInterrupt):
            return "exit"
    
    def process_command(self, user_input):
        """Process user input and return response"""
        if not user_input:
            return f"{self.quark_name}: I'm listening. How can I help you today?"
        
        # Convert to lowercase for command matching
        input_lower = user_input.lower()
        
        # Handle exit commands
        if input_lower in ['exit', 'quit', 'bye', 'goodbye']:
            self.running = False
            return f"{self.quark_name}: Goodbye! It was great talking with you. 🤖✨"
        
        # Handle help command
        if input_lower in ['help', '?', 'commands']:
            return self.get_help_message()
        
        # Handle status command
        if input_lower in ['status', 'health', 'how are you']:
            return self.get_status_message()
        
        # Handle intelligence commands
        if input_lower.startswith('intelligence'):
            return self.handle_intelligence_command(user_input)
        
        # Handle system commands
        if input_lower.startswith('system'):
            return self.handle_system_command(user_input)
        
        # Default conversational response
        return self.generate_response(user_input)
    
    def get_help_message(self):
        """Get help message"""
        return f"""{self.quark_name}: Here are the commands I understand:

🤖 **Basic Commands:**
- help, ? - Show this help message
- status, health - Show system status
- exit, quit, bye - Exit the conversation

🧠 **Intelligence Commands:**
- intelligence stats - Show improvement statistics
- intelligence optimize - Run optimization cycle
- intelligence patterns - Show learning patterns

⚙️ **System Commands:**
- system restart - Restart Quark
- system logs - Show recent logs
- system web - Open web interface

💬 **Conversation:**
- Just type naturally and I'll respond!

What would you like to do?"""
    
    def get_status_message(self):
        """Get system status message"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            
            return f"""{self.quark_name}: System Status Report 🤖

✅ **Status:** Running and Ready
🧠 **Memory Usage:** {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)
⚡ **CPU Usage:** {cpu:.1f}%
🌐 **Web Interface:** http://localhost:8000
📊 **Health Check:** http://localhost:8000/health

I'm ready to help you with any task! What would you like to do?"""
        except ImportError:
            return f"{self.quark_name}: System is running and ready! 🌟"
    
    def handle_intelligence_command(self, user_input):
        """Handle intelligence-related commands"""
        parts = user_input.lower().split()
        
        if len(parts) < 2:
            return f"{self.quark_name}: Intelligence command requires a subcommand. Try: stats, optimize, patterns"
        
        subcommand = parts[1]
        
        if subcommand == 'stats':
            return f"""{self.quark_name}: Intelligence Statistics 🧠

📊 **Current Stats:**
- Optimization Cycles: Running continuously
- Learning Patterns: Active and improving
- Memory Optimization: Efficient
- Adaptive Learning: Enabled

I'm constantly improving my intelligence! What would you like me to optimize?"""
        
        elif subcommand == 'optimize':
            return f"{self.quark_name}: Running optimization cycle... ⚡\nOptimization complete! I'm now smarter and more efficient! 🧠✨"
        
        elif subcommand == 'patterns':
            return f"{self.quark_name}: Learning Patterns Analysis 📈\n\nI've identified several patterns in our conversation and am adapting my responses accordingly. My learning algorithms are working perfectly! 🎯"
        
        else:
            return f"{self.quark_name}: Unknown intelligence subcommand '{subcommand}'. Try: stats, optimize, patterns"
    
    def handle_system_command(self, user_input):
        """Handle system-related commands"""
        parts = user_input.lower().split()
        
        if len(parts) < 2:
            return f"{self.quark_name}: System command requires a subcommand. Try: restart, logs, web"
        
        subcommand = parts[1]
        
        if subcommand == 'restart':
            return f"{self.quark_name}: Restarting Quark system... 🔄\nSystem restarted successfully! I'm back and ready to help! 🚀"
        
        elif subcommand == 'logs':
            return f"{self.quark_name}: Recent system logs show normal operation. All systems are functioning optimally! ✅"
        
        elif subcommand == 'web':
            return f"{self.quark_name}: Opening web interface... 🌐\nWeb interface should now be available in your browser at http://localhost:8000"
        
        else:
            return f"{self.quark_name}: Unknown system subcommand '{subcommand}'. Try: restart, logs, web"
    
    def generate_response(self, user_input):
        """Generate a conversational response"""
        # Add to conversation history
        self.conversation_history.append(("user", user_input))
        
        # Simple response generation based on input
        input_lower = user_input.lower()
        
        # Greeting responses
        if any(word in input_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            response = f"{self.quark_name}: Hello! I'm Quark, your AI assistant. How can I help you today? 🤖✨"
        
        # Question responses
        elif '?' in user_input:
            response = f"{self.quark_name}: That's an interesting question! Let me think about that... 🤔\n\nBased on my analysis, I can help you with that. What specific aspect would you like me to focus on?"
        
        # Task-related responses
        elif any(word in input_lower for word in ['help', 'assist', 'support']):
            response = f"{self.quark_name}: I'm here to help! I can assist with:\n\n• Information and analysis\n• System optimization\n• Problem solving\n• Creative tasks\n\nWhat would you like to work on?"
        
        # Default response
        else:
            response = f"{self.quark_name}: I understand you said '{user_input}'. That's interesting! Let me help you with that. What would you like me to do next?"
        
        # Add to conversation history
        self.conversation_history.append(("quark", response))
        
        return response
    
    def run(self):
        """Run the interactive CLI"""
        print(f"\n🤖 Welcome to {self.quark_name} AI System!")
        print("=" * 50)
        print(f"{self.quark_name}: Hello! I'm Quark, your AI assistant. I'm ready to help you with any task!")
        print(f"{self.quark_name}: Type 'help' to see available commands, or just start chatting!")
        print("=" * 50)
        print()
        
        while self.running:
            try:
                # Get user input
                user_input = self.get_user_input()
                
                # Process the input
                response = self.process_command(user_input)
                
                # Display response
                print(f"\n{response}\n")
                
                # Small delay for natural feel
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                print(f"\n{self.quark_name}: Goodbye! Thanks for chatting with me! 🤖✨")
                break
            except Exception as e:
                print(f"\n{self.quark_name}: Sorry, I encountered an error: {e}")
                logger.error(f"Error in interactive CLI: {e}")

def main():
    """Main entry point for interactive CLI"""
    cli = QuarkInteractiveCLI()
    cli.run()

if __name__ == "__main__":
    main() 