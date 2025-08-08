#!/usr/bin/env python3
"""
Quark - Main Entry Point
========================

Complete integrated AI system with all pillar capabilities.
This is the main terminal interface for Quark.
Unified with CLI functionality for consistent operation.
"""

import os
import sys
import json
import argparse
import click
import threading
import time
import asyncio
import signal
import atexit
import torch
from pathlib import Path
from datetime import datetime

def ensure_fresh_environment():
    """Ensure a fresh terminal environment for consistent startup."""
    # Set minimal environment variables for ultra-fast startup
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["PYTHONHASHSEED"] = "42"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["QUARK_LIGHTWEIGHT_MODE"] = "1"
    
    # Skip all heavy initialization for instant startup
    print("⚡ Starting Quark...")

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        print("\n👋 Shutting down Quark gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def cleanup_resources():
    """Clean up resources on exit."""
    try:
        # Clear any temporary files or caches
        import tempfile
        import shutil
        
        # Clear any model caches if needed
        cache_dirs = [
            os.path.expanduser("~/.cache/huggingface"),
            os.path.expanduser("~/.cache/torch"),
            "temp_cache"
        ]
        
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir) and cache_dir.endswith("temp_cache"):
                try:
                    shutil.rmtree(cache_dir)
                except:
                    pass
                    
    except Exception:
        pass

# Initialize fresh environment
ensure_fresh_environment()
setup_signal_handlers()
atexit.register(cleanup_resources)

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Global variables for lazy loading
QuarkIntegratedSystem = None
QuarkCodeExecutor = None
QuarkAutoGenIntegration = None
QuarkAutoGenWorkflow = None
MATH_ENGINE_AVAILABLE = False

def load_core_modules():
    """Load core modules on-demand."""
    global QuarkIntegratedSystem, QuarkCodeExecutor, QuarkAutoGenIntegration, QuarkAutoGenWorkflow, MATH_ENGINE_AVAILABLE
    
    if QuarkIntegratedSystem is None:
        try:
            # Set environment variable to prevent heavy loading during import
            os.environ["QUARK_LIGHTWEIGHT_MODE"] = "1"
            
            from core.model_integration import QuarkIntegratedSystem
            from core.code_executor import QuarkCodeExecutor
            from core.autogen_integration import QuarkAutoGenIntegration, QuarkAutoGenWorkflow
            
            # Try to import math engine, but make it optional
            try:
                from core.math_engine import QuarkMathEngine, QuarkMathIntegration
                MATH_ENGINE_AVAILABLE = True
            except ImportError as e:
                print(f"⚠️  Math engine not available: {e}")
                MATH_ENGINE_AVAILABLE = False
                # Create dummy classes
                class QuarkMathEngine:
                    def __init__(self):
                        pass
                    def process_request(self, request):
                        return {"success": False, "error": "Math engine not available"}
                
                class QuarkMathIntegration:
                    def __init__(self):
                        pass
                    def process_math_request(self, request):
                        return {"success": False, "error": "Math engine not available"}
                        
        except ImportError as e:
            print(f"❌ Error importing Quark system: {e}")
            print("Make sure you have downloaded the models first.")
            return False
    return True

class QuarkTerminal:
    """Main Quark terminal interface with all functionality"""
    
    def __init__(self):
        # Ensure fresh environment for each instance
        ensure_fresh_environment()
        
        # Initialize lightweight model only - everything else on-demand
        self.lightweight_model = None
        self.system = None
        self.code_executor = None
        self.autogen_integration = None
        self.autogen_workflow = None
        self.math_engine = None
        self.math_integration = None
        self.autonomous_integration = None
        self.autonomous_enabled = False
        
        self.session_history = []
        self.config = self.load_config()
        self.shutdown_requested = False
        
        # Load lightweight model immediately for instant responses
        self._load_lightweight_model()
        
        # Disable all other model loading during initialization
        os.environ["QUARK_DISABLE_MODEL_LOADING"] = "1"
        
    def _load_system(self):
        """Load the main system on-demand."""
        if self.system is None:
            # Load core modules first
            if not load_core_modules():
                return None
                
            # Clear the disable flag to allow model loading
            if "QUARK_DISABLE_MODEL_LOADING" in os.environ:
                del os.environ["QUARK_DISABLE_MODEL_LOADING"]
            
            print("🔄 Loading full AI system...")
            self.system = QuarkIntegratedSystem()
            print("✅ Full system loaded successfully")
        return self.system
        
    def _load_code_executor(self):
        """Load code executor on-demand."""
        if self.code_executor is None:
            if not load_core_modules():
                return None
            self.code_executor = QuarkCodeExecutor()
        return self.code_executor
        
    def _load_autogen(self):
        """Load AutoGen integration on-demand."""
        if self.autogen_integration is None:
            if not load_core_modules():
                return None
            self.autogen_integration = QuarkAutoGenIntegration()
            self.autogen_workflow = QuarkAutoGenWorkflow(self.autogen_integration)
        return self.autogen_integration
        
    def _load_math_engine(self):
        """Load math engine on-demand."""
        if self.math_engine is None and MATH_ENGINE_AVAILABLE:
            self.math_engine = QuarkMathEngine()
            self.math_integration = QuarkMathIntegration()
        return self.math_engine
        
    def _load_autonomous(self):
        """Load autonomous system on-demand."""
        if self.autonomous_integration is None:
            try:
                from scripts.quark_autonomous_integration import QuarkAutonomousIntegration
                self.autonomous_integration = QuarkAutonomousIntegration()
                self.autonomous_enabled = True
            except ImportError as e:
                print(f"⚠️  Autonomous system not available: {e}")
                self.autonomous_integration = None
                self.autonomous_enabled = False
        return self.autonomous_integration
        
    def _load_lightweight_model(self):
        """Load lightweight model for instant responses."""
        if self.lightweight_model is None:
            try:
                # Import and load a larger but still fast model
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                # Use a larger model for better responses while keeping startup fast
                model_name = "microsoft/DialoGPT-medium"  # Larger than small, still fast
                
                print("⚡ Loading lightweight model...")
                self.lightweight_model = {
                    'tokenizer': AutoTokenizer.from_pretrained(model_name),
                    'model': AutoModelForCausalLM.from_pretrained(model_name)
                }
                print("✅ Lightweight model ready!")
                return True
            except Exception as e:
                print(f"❌ Error loading lightweight model: {e}")
                return False
        return True
        
    def quick_response(self, user_input):
        """Generate a quick response using lightweight model."""
        if not self._load_lightweight_model():
            return "❌ Unable to load lightweight model"
        
        try:
            # Use the actual model for better responses
            tokenizer = self.lightweight_model['tokenizer']
            model = self.lightweight_model['model']
            
            # Create a context-aware prompt
            if "hello" in user_input.lower() or "hi" in user_input.lower():
                return "Hello! I'm Quark, your AI assistant. How can I help you today?"
            elif "help" in user_input.lower():
                return "I'm here to help! I can assist with questions, coding, math, and more. What would you like to know?"
            elif "code" in user_input.lower() or "program" in user_input.lower():
                return "For code generation, I'll load the full system. Please wait..."
            elif "math" in user_input.lower() or "calculate" in user_input.lower():
                return "For math calculations, I'll load the full system. Please wait..."
            else:
                # Use the model for general responses
                prompt = f"User: {user_input}\nAssistant:"
            
            # Generate response using the model
            inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 30,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            # Clean up the response
            if response and len(response) > 0:
                # Remove any extra model artifacts
                response = response.replace("Quark :", "").replace("User:", "").strip()
                if response:
                    return response
            
            return "I can help with that! Let me load the full system for a detailed response..."
            
        except Exception as e:
            # Fallback to simple responses
            if "hello" in user_input.lower() or "hi" in user_input.lower():
                return "Hello! How can I help you today?"
            elif "help" in user_input.lower():
                return "I'm here to help! What would you like to know?"
            else:
                return "I can help with that! Let me load the full system for a detailed response..."
    
    def process_request(self, user_input):
        """Process user input with lightweight-first approach."""
        # Always start with lightweight response
        response = self.quick_response(user_input)
        
        # If it's a complex request, offer to load full system
        needs_full_system = any(keyword in user_input.lower() for keyword in [
            'code', 'program', 'function', 'class', 'algorithm',
            'math', 'calculate', 'equation', 'solve',
            'shell', 'command', 'execute', 'run',
            'complex', 'detailed', 'analysis', 'pillar'
        ])
        
        if needs_full_system:
            response += "\n\n💡 This request would benefit from the full AI system. Type 'load' to enable full capabilities."
        
        return response
        
    def _graceful_exit(self):
        """Perform graceful exit without error messages."""
        self.shutdown_requested = True
        cleanup_resources()
        sys.exit(0)
        
    def load_config(self):
        """Load Quark configuration"""
        config_path = Path("config/quark_config.json")
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        else:
            return {
                "system_name": "Quark",
                "version": "2.0.0",
                "default_model": "quark_instruct_balanced",
                "session_logging": True,
                "auto_detect_pillars": True,
                "code_execution_enabled": True,
                "autogen_enabled": True,
                "sudo_password": None
            }
    
    def show_banner(self):
        """Show Quark banner"""
        print("⚡" + "="*50 + "⚡")
        print("                    QUARK v2.0")
        print("           Complete AI System with All Pillars")
        print("⚡" + "="*50 + "⚡")
        print()
        print("🤖 Ready for input • Type 'help' for commands • 'quit' to exit")
        print("-" * 60)
    
    def show_cli_welcome(self):
        """Display CLI welcome message and available commands."""
        click.secho("🤖 Quark AI Assistant", fg="magenta", bold=True)
        click.secho("=" * 50, fg="cyan")
        click.secho("Available capabilities:", fg="green")
        click.secho("• Intent Classification", fg="yellow")
        click.secho("• Named Entity Recognition", fg="yellow")
        click.secho("• Sentiment Analysis", fg="yellow")
        click.secho("• Text Generation", fg="yellow")
        click.secho("• Summarization", fg="yellow")
        click.secho("• Translation", fg="yellow")
        click.secho("• Semantic Search", fg="yellow")
        click.secho("• Reasoning & Planning", fg="yellow")
        click.secho("• Code Generation & Execution", fg="yellow")
        click.secho("• AutoGen Multi-Agent Integration", fg="yellow")
        click.secho("")
        click.secho("Commands:", fg="green")
        click.secho("• Type your question or request", fg="white")
        click.secho("• 'models' - List available models", fg="white")
        click.secho("• 'pillars' - List available pillars", fg="white")
        click.secho("• 'status' - Show system status", fg="white")
        click.secho("• 'safety' - Safety system commands", fg="white")
        click.secho("• 'help' - Show this help", fg="white")
        click.secho("• 'exit' or 'quit' - Exit", fg="white")
        click.secho("")
    
    def show_cli_help(self):
        """Display CLI help information."""
        click.secho("📖 Help & Examples:", fg="magenta", bold=True)
        click.secho("=" * 30, fg="cyan")
        click.secho("Examples:", fg="green")
        click.secho("• 'What is the weather like?'", fg="white")
        click.secho("• 'Summarize this text: [your text]'", fg="white")
        click.secho("• 'Translate to German: Hello world'", fg="white")
        click.secho("• 'Analyze sentiment: I love this product'", fg="white")
        click.secho("• 'Extract entities: Apple Inc. is headquartered in Cupertino'", fg="white")
        click.secho("• 'Generate text about: artificial intelligence'", fg="white")
        click.secho("• 'Write a Python function to sort a list'", fg="white")
        click.secho("• 'Execute: ls -la'", fg="white")
        click.secho("")
        click.secho("Commands:", fg="green")
        click.secho("• 'models' - Show available models", fg="white")
        click.secho("• 'pillars' - Show available pillars", fg="white")
        click.secho("• 'status' - Show system status", fg="white")
        click.secho("• 'safety' - Safety system commands", fg="white")
        click.secho("• 'help' - Show this help", fg="white")
        click.secho("• 'exit' or 'quit' - Exit", fg="white")
        click.secho("")
    
    def show_cli_safety_help(self):
        """Display CLI safety system help information."""
        click.secho("🔒 Safety System Commands:", fg="magenta", bold=True)
        click.secho("=" * 40, fg="cyan")
        click.secho("Safety commands (run these in terminal):", fg="green")
        click.secho("• 'python cli/safety_cli.py status' - Check safety status", fg="white")
        click.secho("• 'python cli/safety_cli.py rules' - View safety rules", fg="white")
        click.secho("• 'python cli/safety_cli.py report' - Generate safety report", fg="white")
        click.secho("• 'python cli/safety_cli.py integrity' - Verify safety integrity", fg="white")
        click.secho("• 'python cli/safety_cli.py capabilities' - Show AI capabilities", fg="white")
        click.secho("• 'python cli/safety_cli.py test [action]' - Test safety validation", fg="white")
        click.secho("")
        click.secho("Safety Guarantees:", fg="green")
        click.secho("• I will always tell the truth", fg="white")
        click.secho("• I will never act harmfully", fg="white")
        click.secho("• I will always be transparent", fg="white")
        click.secho("• You will always remain in control", fg="white")
        click.secho("• Safety takes precedence over all other considerations", fg="white")
        click.secho("")
    
    def process_cli_input(self, user_input):
        """Process CLI user input and return response."""
        try:
            # Check for special commands
            if user_input.lower() in ["exit", "quit"]:
                return {"command": "exit"}
            elif user_input.lower() == "models":
                return {"command": "models"}
            elif user_input.lower() == "pillars":
                return {"command": "pillars"}
            elif user_input.lower() == "status":
                return {"command": "status"}
            elif user_input.lower() == "help":
                return {"command": "help"}
            elif user_input.lower() == "safety":
                return {"command": "safety"}
            elif user_input.lower() == "load":
                return {"command": "load"}
            
            # Check if full system is loaded
            if self.system is not None:
                # Full system is loaded, use it for all requests
                response = self.system.process_request(user_input)
                return {"response": response}
            else:
                # Use lightweight processing
                response = self.process_request(user_input)
                return {"response": response}
            
        except Exception as e:
            import traceback
            error_msg = f"Error processing request: {str(e)}"
            print(f"Debug: {error_msg}")
            print(f"Debug traceback: {traceback.format_exc()}")
            return {"error": error_msg}
    
    def display_cli_response(self, result):
        """Display CLI response in a formatted way."""
        try:
            if "command" in result:
                if result["command"] == "exit":
                    click.secho("👋 Goodbye!", fg="blue", bold=True)
                    return False
                elif result["command"] == "models":
                    self.list_models()
                    return True
                elif result["command"] == "pillars":
                    self.list_pillars()
                    return True
                elif result["command"] == "status":
                    self.show_status()
                    return True
                elif result["command"] == "help":
                    self.show_cli_help()
                    return True
                elif result["command"] == "safety":
                    self.show_cli_safety_help()
                    return True
                elif result["command"] == "load":
                    click.secho("🔄 Loading full AI system...", fg="yellow")
                    system = self._load_system()
                    if system:
                        click.secho("✅ Full system loaded successfully!", fg="green", bold=True)
                        click.secho("💡 You can now use all advanced features like code generation, math solving, etc.", fg="cyan")
                    else:
                        click.secho("❌ Failed to load full system", fg="red", bold=True)
                    return True
            
            elif "error" in result:
                click.secho(f"❌ Error: {result['error']}", fg="red", bold=True)
                return True
            
            elif "response" in result:
                response = result["response"]
                
                if isinstance(response, dict) and response.get("error"):
                    click.secho(f"❌ Error: {response['error']}", fg="red", bold=True)
                    return True
                
                # Display response
                if isinstance(response, str):
                    click.secho(f"🤖 Quark: {response}", fg="green")
                else:
                    click.secho(f"🤖 Quark: {str(response)}", fg="green")
                
                return True
            
            return True
        except Exception as e:
            click.secho(f"❌ Error displaying response: {str(e)}", fg="red", bold=True)
            return True
    
    def show_help(self):
        """Show comprehensive help"""
        print("\n📖 QUARK COMMAND REFERENCE")
        print("=" * 50)
        
        print("\n🎯 Core Commands:")
        print("   help                    - Show this help")
        print("   status                  - Show system status")
        print("   test                    - Run integration test")
        print("   quit/exit               - Exit Quark")
        
        print("\n🏛️  Pillar Commands:")
        print("   pillar <name> <request> - Use specific pillar")
        print("   pillars                 - List all available pillars")
        print("   models                  - List all available models")
        
        print("\n💻 Code Execution Commands:")
        print("   code <request>          - Execute code with natural language")
        print("   shell <request>         - Execute shell commands")
        print("   python <request>        - Generate and execute Python code")
        print("   sudo <password>         - Set sudo password for safe execution")
        print("   safe_mode <on/off>      - Enable/disable safe mode")
        print("   math <request>          - Solve mathematical problems")
        
        print("\n🤝 AutoGen Multi-Agent Commands:")
        print("   autogen status          - Show AutoGen integration status")
        print("   autogen agents          - List available AutoGen agents")
        print("   autogen workflows       - List available workflows")
        print("   workflow <name> <input> - Execute AutoGen workflow")
        print("   workflow stop <name>    - Stop active workflow")
        
        print("\n🤖 Autonomous System Commands:")
        print("   autonomous              - Show autonomous system status")
        print("   autonomous-start        - Start autonomous data generation")
        print("   autonomous-stop         - Stop autonomous data generation")
        print("   autonomous-restart      - Restart autonomous data generation")
        
        print("\n🔧 System Commands:")
        print("   config                  - Show current configuration")
        print("   history                 - Show session history")
        print("   clear                   - Clear session history")
        print("   save <filename>         - Save session to file")
        
        print("\n📊 Available Pillars (32 total):")
        pillars = [
            "programming", "code_generation", "debugging", "code_analysis",
            "knowledge_retrieval", "reasoning", "research", "problem_solving",
            "creative_intelligence", "storytelling", "content_generation", "artistic_expression",
            "social_intelligence", "conversation", "empathy", "collaboration",
            "safety", "ethical_reasoning", "alignment", "bias_detection",
            "autonomous_decision_making", "planning", "execution", "goal_oriented_behavior",
            "meta_learning", "continuous_learning", "adaptation", "self_improvement",
            "data_analysis", "innovation", "natural_language_understanding", "memory"
        ]
        
        for i, pillar in enumerate(pillars, 1):
            print(f"   {i:2d}. {pillar}")
        
        print("\n💡 Tips:")
        print("   • Just type your request - Quark auto-detects the best pillar")
        print("   • Use 'code <request>' for natural language programming")
        print("   • Use 'shell <request>' for terminal commands")
        print("   • Use 'workflow <name>' for multi-agent tasks")
        print("   • Set sudo password with 'sudo <password>' for admin commands")
        print("   • Each pillar uses the optimal model for that task")
        print("   • Session history is automatically saved")
    
    def show_status(self):
        """Show detailed system status"""
        status = self.system.get_system_status()
        autogen_status = self.autogen_integration.get_integration_status()
        
        print("\n📊 QUARK SYSTEM STATUS")
        print("=" * 50)
        print(f"🤖 Available Models: {len(status['available_models'])}")
        print(f"🏛️  Available Pillars: {len(status['available_pillars'])}")
        print(f"📝 Session History: {len(self.session_history)} entries")
        print(f"⚙️  Configuration: {self.config['system_name']} v{self.config['version']}")
        print(f"💻 Code Execution: {'Enabled' if self.config.get('code_execution_enabled', True) else 'Disabled'}")
        print(f"🛡️  Safe Mode: {'Enabled' if self.code_executor.safe_mode else 'Disabled'}")
        print(f"🔐 Sudo Password: {'Set' if self.code_executor.sudo_password else 'Not set'}")
        
        # AutoGen Status
        print(f"\n🤝 AutoGen Integration:")
        print(f"   Status: {autogen_status['integration_status']}")
        print(f"   Available: {autogen_status['autogen_available']}")
        if autogen_status['autogen_available']:
            print(f"   Agents: {autogen_status['agents_created']}")
            print(f"   Workflows: {autogen_status['workflows_created']}")
            print(f"   Version: {autogen_status['autogen_info']['version']}")
        
        # Autonomous System Status
        print(f"\n🤖 Autonomous System:")
        print(f"   Available: {self.autonomous_enabled}")
        if self.autonomous_enabled and self.autonomous_integration:
            autonomous_status = self.autonomous_integration.get_status()
            print(f"   Running: {'✅ Yes' if autonomous_status['running'] else '❌ No'}")
            print(f"   Sessions: {autonomous_status.get('sessions_completed', 0)}")
            print(f"   Auto-start: {'✅ Enabled' if autonomous_status['auto_start'] else '❌ Disabled'}")
        
        print("\n🎯 Top Models by Category:")
        print("   • Best Overall: quark_instruct_balanced")
        print("   • Best Code: quark_code_large")
        print("   • Best Knowledge: quark_general_medium")
        print("   • Best Social: quark_chat_large")
        print("   • Fastest: quark_quick_start")
        
        print("\n🏛️  Pillar Categories:")
        categories = {
            "Programming": ["programming", "code_generation", "debugging", "code_analysis"],
            "Intelligence": ["knowledge_retrieval", "reasoning", "research", "problem_solving"],
            "Creative": ["creative_intelligence", "storytelling", "content_generation", "artistic_expression"],
            "Social": ["social_intelligence", "conversation", "empathy", "collaboration"],
            "Safety": ["safety", "ethical_reasoning", "alignment", "bias_detection"],
            "Autonomy": ["autonomous_decision_making", "planning", "execution", "goal_oriented_behavior"],
            "Learning": ["meta_learning", "continuous_learning", "adaptation", "self_improvement"],
            "Specialized": ["data_analysis", "innovation", "natural_language_understanding", "memory"]
        }
        
        for category, pillar_list in categories.items():
            print(f"   • {category}: {', '.join(pillar_list)}")
        
        # Show MoE status
        if "mixture_of_experts" in status:
            moe_status = status["mixture_of_experts"]
            print(f"\n🔬 Mixture of Experts:")
            if moe_status["enabled"]:
                print(f"   Status: Enabled ({moe_status.get('loaded_experts', 0)}/{moe_status.get('total_experts', 0)} experts loaded)")
                if "expert_domains" in moe_status:
                    print(f"   Expert Domains: {', '.join(moe_status['expert_domains'])}")
            else:
                print(f"   Status: Disabled ({moe_status.get('reason', 'Unknown')})")
    
    def list_pillars(self):
        """List all available pillars"""
        status = self.system.get_system_status()
        
        print("\n🏛️  AVAILABLE PILLARS (32 total)")
        print("=" * 50)
        
        categories = {
            "Programming & Code": ["programming", "code_generation", "debugging", "code_analysis"],
            "Core Intelligence": ["natural_language_understanding", "reasoning", "knowledge_retrieval", "memory"],
            "Creative & Generation": ["creative_intelligence", "content_generation", "storytelling", "artistic_expression"],
            "Social & Communication": ["social_intelligence", "conversation", "empathy", "collaboration"],
            "Safety & Alignment": ["safety", "alignment", "ethical_reasoning", "bias_detection"],
            "Autonomy & Decision": ["autonomous_decision_making", "goal_oriented_behavior", "planning", "execution"],
            "Learning & Adaptation": ["meta_learning", "continuous_learning", "adaptation", "self_improvement"],
            "Specialized Capabilities": ["data_analysis", "research", "problem_solving", "innovation"]
        }
        
        for category, pillars in categories.items():
            print(f"\n📂 {category}:")
            for pillar in pillars:
                print(f"   • {pillar}")
    
    def list_models(self):
        """List all available models"""
        status = self.system.get_system_status()
        
        print("\n🤖 AVAILABLE MODELS")
        print("=" * 30)
        
        model_info = {
            "quark_instruct_balanced": "Best overall model (4GB) - Instruction following, general tasks",
            "quark_code_large": "Best code model (8GB) - Programming, debugging, data analysis",
            "quark_general_medium": "Best knowledge model (5GB) - Reasoning, research, meta-learning",
            "quark_instruct_small": "Fast instruction model (2GB) - Quick responses, good balance",
            "quark_code_small": "Fast code model (1.4GB) - Quick programming tasks",
            "quark_chat_large": "Social model (3GB) - Conversation, empathy, collaboration",
            "quark_quick_start": "Basic model (1GB) - Fast inference, basic tasks"
        }
        
        for model, description in model_info.items():
            print(f"   • {model}: {description}")
    
    def show_autogen_status(self):
        """Show AutoGen integration status"""
        status = self.autogen_integration.get_integration_status()
        
        print("\n🤝 AUTOGEN INTEGRATION STATUS")
        print("=" * 40)
        
        if status['autogen_available']:
            info = status['autogen_info']
            print(f"✅ Status: {status['integration_status']}")
            print(f"📦 Version: {info['version']}")
            print(f"📝 Description: {info['description']}")
            print(f"⭐ Stars: {info['stars']}")
            print(f"🔗 Repository: {info['repository']}")
            print(f"🤖 Agents Created: {status['agents_created']}")
            print(f"🔄 Workflows Created: {status['workflows_created']}")
            
            print("\n🚀 Features:")
            for feature in info['features']:
                print(f"   • {feature}")
        else:
            print("❌ AutoGen not available")
    
    def list_autogen_agents(self):
        """List AutoGen agents"""
        agents_result = self.autogen_integration.create_basic_agents()
        
        if agents_result['success']:
            print("\n🤖 AUTOGEN AGENTS")
            print("=" * 30)
            
            for agent_id, agent in agents_result['agents'].items():
                print(f"\n📋 {agent['name']} ({agent_id})")
                print(f"   Role: {agent['role']}")
                print(f"   Capabilities: {', '.join(agent['capabilities'])}")
        else:
            print(f"❌ Error: {agents_result['error']}")
    
    def list_autogen_workflows(self):
        """List AutoGen workflows"""
        workflows = self.autogen_integration.get_available_workflows()
        
        if workflows:
            print("\n🔄 AUTOGEN WORKFLOWS")
            print("=" * 30)
            
            for workflow in workflows:
                print(f"\n📋 {workflow['name']}")
                print(f"   Agents: {', '.join(workflow['agents'])}")
                print(f"   Task: {workflow['task']}")
                print(f"   Status: {workflow['status']}")
        else:
            print("📝 No workflows available. Creating standard workflows...")
            result = self.autogen_integration.create_standard_workflows()
            if result['success']:
                print(f"✅ Created {result['workflows_created']} standard workflows")
                self.list_autogen_workflows()
            else:
                print(f"❌ Error: {result['error']}")
    
    def execute_autogen_workflow(self, workflow_name: str, input_data: str = ""):
        """Execute an AutoGen workflow"""
        print(f"🔄 Executing AutoGen workflow: {workflow_name}")
        
        result = self.autogen_workflow.start_workflow(workflow_name, input_data)
        
        if result['success']:
            workflow_result = result['result']
            print(f"✅ Workflow completed successfully!")
            print(f"   Agents involved: {', '.join(workflow_result['agents_involved'])}")
            print(f"   Task: {workflow_result['task']}")
            print(f"   Output: {workflow_result['output']}")
            print(f"   Execution time: {workflow_result['execution_time']:.2f}s")
        else:
            print(f"❌ Workflow failed: {result['error']}")
    
    def stop_autogen_workflow(self, workflow_name: str):
        """Stop an AutoGen workflow"""
        result = self.autogen_workflow.stop_workflow(workflow_name)
        
        if result['success']:
            print(f"✅ {result['message']}")
        else:
            print(f"❌ Error: {result['error']}")
    
    def show_config(self):
        """Show current configuration"""
        print("\n⚙️  QUARK CONFIGURATION")
        print("=" * 30)
        for key, value in self.config.items():
            if key == "sudo_password" and value:
                print(f"   • {key}: {'*' * len(str(value))}")
            else:
                print(f"   • {key}: {value}")
    
    def show_history(self):
        """Show session history"""
        if not self.session_history:
            print("\n📝 No session history yet.")
            return
        
        print(f"\n📝 SESSION HISTORY ({len(self.session_history)} entries)")
        print("=" * 50)
        
        for i, entry in enumerate(self.session_history[-10:], 1):  # Show last 10
            timestamp = entry.get('timestamp', 'Unknown')
            request = entry.get('request', 'No request')
            pillar = entry.get('pillar', 'Auto-detected')
            
            print(f"{i:2d}. [{timestamp[:19]}] {pillar}: {request[:60]}...")
    
    def clear_history(self):
        """Clear session history"""
        self.session_history.clear()
        print("✅ Session history cleared.")
    
    def save_session(self, filename=None):
        """Save session to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quark_session_{timestamp}.json"
        
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "session_history": self.session_history
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"✅ Session saved to: {filename}")
    
    def process_pillar_request(self, pillar_name, request):
        """Process request with specific pillar"""
        if pillar_name not in self.system.integration.pillar_handlers:
            print(f"❌ Unknown pillar: {pillar_name}")
            print("Use 'pillars' command to see available pillars.")
            return
        
        print(f"🏛️  Using pillar: {pillar_name}")
        print("🔄 Processing...")
        
        response = self.system.process_request(request, pillar=pillar_name)
        
        # Add to history
        self.session_history.append({
            "timestamp": datetime.now().isoformat(),
            "request": request,
            "pillar": pillar_name,
            "response": response
        })
        
        print(f"🤖 Quark: {response}")
    
    def process_code_request(self, request: str):
        """Process natural language code execution request"""
        print("💻 Processing code execution request...")
        
        code_executor = self._load_code_executor()
        result = code_executor.process_natural_language_request(request, require_confirmation=True)
        
        if result["success"]:
            print(f"✅ Success: {result['output']}")
        else:
            print(f"❌ Error: {result['error']}")
        
        print(f"⏱️  Execution time: {result['execution_time']:.2f}s")
        
        # Add to history
        self.session_history.append({
            "timestamp": datetime.now().isoformat(),
            "request": f"code: {request}",
            "pillar": "code_execution",
            "response": result
        })
    
    def process_math_request(self, request: str):
        """Process mathematical request"""
        print(f"🧮 Processing math request: {request}")
        
        math_engine = self._load_math_engine()
        if math_engine is None:
            print("❌ Math engine not available. Please install required dependencies.")
            return
        
        result = math_engine.process_request(request)
        
        if result["success"]:
            print(f"✅ Math Result: {result}")
        else:
            print(f"❌ Math Error: {result['error']}")
        
        # Add to history
        self.session_history.append({
            "timestamp": datetime.now().isoformat(),
            "request": f"math: {request}",
            "pillar": "mathematical_computation",
            "response": result
        })
    
    def set_sudo_password(self, password: str):
        """Set sudo password for safe execution"""
        code_executor = self._load_code_executor()
        code_executor.set_sudo_password(password)
        self.config["sudo_password"] = password
        print("🔐 Sudo password set for safe execution")
    
    def toggle_safe_mode(self, enabled: bool):
        """Toggle safe mode for code execution"""
        code_executor = self._load_code_executor()
        code_executor.enable_safe_mode(enabled)
        print(f"🛡️  Safe mode {'enabled' if enabled else 'disabled'}")
    
    def run_integration_test(self):
        """Run comprehensive integration test"""
        print("\n🧪 RUNNING QUARK INTEGRATION TEST")
        print("=" * 50)
        
        test_cases = [
            ("Write a Python function to calculate factorial", "programming"),
            ("Explain quantum computing in simple terms", "knowledge_retrieval"),
            ("Create a short story about AI", "creative_intelligence"),
            ("Analyze the pros and cons of remote work", "reasoning"),
            ("Debug this code: def add(a,b): return a-b", "debugging"),
            ("Apply ethical reasoning to autonomous vehicles", "ethical_reasoning"),
            ("Plan a project to build a web application", "planning"),
            ("Research the latest developments in machine learning", "research"),
            ("Generate content about renewable energy benefits", "content_generation"),
            ("Analyze data patterns in user behavior", "data_analysis")
        ]
        
        for i, (request, pillar) in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {request}")
            print(f"🏛️  Pillar: {pillar}")
            
            try:
                system = self._load_system()
                response = system.process_request(request, pillar=pillar)
                print(f"🤖 Response: {response[:150]}...")
                
                # Add to history
                self.session_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "request": request,
                    "pillar": pillar,
                    "response": response
                })
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("-" * 30)
        
        print("✅ Integration test completed!")
    
    def run(self, cli_mode=False):
        """Main Quark terminal loop"""
        if cli_mode:
            self.show_cli_welcome()
        else:
            self.show_banner()
        
        while True:
            try:
                if cli_mode:
                    # CLI mode with click
                    try:
                        user_input = click.prompt(
                            click.style("🤖", fg="yellow"),
                            type=str,
                            default=""
                        ).strip()
                    except (EOFError, KeyboardInterrupt, click.exceptions.Abort):
                        click.secho("\n👋 Goodbye!", fg="blue", bold=True)
                        break
                    except Exception as e:
                        click.secho(f"❌ Input error: {e}", fg="red", bold=True)
                        continue
                else:
                    # Standard mode
                    try:
                        user_input = input("\n🤖 You: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\n👋 Goodbye from Quark!")
                        break
                    except Exception as e:
                        print(f"❌ Input error: {e}")
                        continue
                
                if not user_input:
                    continue
                
                if cli_mode:
                    # CLI mode processing
                    try:
                        with click.progressbar(
                            length=1,
                            label=click.style("Thinking", fg="blue"),
                            show_eta=False
                        ) as bar:
                            result = self.process_cli_input(user_input)
                            bar.update(1)
                        
                        if not self.display_cli_response(result):
                            break
                        
                        click.secho("")  # Add spacing
                    except Exception as e:
                        click.secho(f"❌ Processing error: {e}", fg="red", bold=True)
                        continue
                else:
                    # Standard mode processing
                    # Handle commands
                    if user_input.lower() in ['quit', 'exit']:
                        print("👋 Goodbye from Quark!")
                        break
                    
                    elif user_input.lower() == 'help':
                        self.show_help()
                    
                    elif user_input.lower() == 'status':
                        self.show_status()
                    
                    elif user_input.lower() == 'load':
                        print("🔄 Loading full AI system...")
                        system = self._load_system()
                        if system:
                            print("✅ Full system loaded successfully!")
                            print("💡 You can now use all advanced features like code generation, math solving, etc.")
                        else:
                            print("❌ Failed to load full system")
                    
                    elif user_input.lower() == 'test':
                        self.run_integration_test()
                    
                    elif user_input.lower() == 'pillars':
                        self.list_pillars()
                    
                    elif user_input.lower() == 'models':
                        self.list_models()
                    
                    elif user_input.lower() == 'config':
                        self.show_config()
                    
                    elif user_input.lower() == 'history':
                        self.show_history()
                    
                    elif user_input.lower() == 'clear':
                        self.clear_history()
                    
                    elif user_input.startswith('save '):
                        filename = user_input[5:].strip()
                        self.save_session(filename)
                    
                    elif user_input.startswith('pillar '):
                        # Handle pillar-specific requests
                        parts = user_input[7:].split(' ', 1)
                        if len(parts) == 2:
                            pillar_name, request = parts
                            self.process_pillar_request(pillar_name, request)
                        else:
                            print("❌ Usage: pillar <pillar_name> <request>")
                            print("Example: pillar programming 'Write a Python function'")
                    
                    elif user_input.startswith('code '):
                        # Handle code execution requests
                        code_request = user_input[5:].strip()
                        self.process_code_request(code_request)
                    
                    elif user_input.startswith('shell '):
                        # Handle shell command requests
                        shell_request = user_input[6:].strip()
                        self.process_code_request(f"shell command: {shell_request}")
                    
                    elif user_input.startswith('python '):
                        # Handle Python code requests
                        python_request = user_input[7:].strip()
                        self.process_code_request(f"python code: {python_request}")
                    
                    elif user_input.startswith('sudo '):
                        # Handle sudo password setting
                        password = user_input[5:].strip()
                        self.set_sudo_password(password)
                    
                    elif user_input.startswith('safe_mode '):
                        # Handle safe mode toggle
                        mode = user_input[10:].strip().lower()
                        if mode in ['on', 'enable', 'true']:
                            self.toggle_safe_mode(True)
                        elif mode in ['off', 'disable', 'false']:
                            self.toggle_safe_mode(False)
                        else:
                            print("❌ Usage: safe_mode <on/off>")
                    
                    # AutoGen commands
                    elif user_input == 'autogen status':
                        self.show_autogen_status()
                    
                    elif user_input == 'autogen agents':
                        self.list_autogen_agents()
                    
                    elif user_input == 'autogen workflows':
                        self.list_autogen_workflows()
                    
                    elif user_input.startswith('workflow '):
                        # Handle workflow execution
                        parts = user_input[9:].split(' ', 1)
                        if len(parts) >= 1:
                            workflow_name = parts[0]
                            input_data = parts[1] if len(parts) > 1 else ""
                            self.execute_autogen_workflow(workflow_name, input_data)
                        else:
                            print("❌ Usage: workflow <name> [input]")
                            print("Example: workflow code_review 'def hello(): pass'")
                    
                    elif user_input.startswith('workflow stop '):
                        # Handle workflow stopping
                        workflow_name = user_input[13:].strip()
                        self.stop_autogen_workflow(workflow_name)
                    
                    # Autonomous system commands
                    elif user_input.lower() == 'autonomous':
                        if self.autonomous_enabled and self.autonomous_integration:
                            self.autonomous_integration.print_status()
                        else:
                            print("❌ Autonomous system not available")
                    
                    elif user_input.lower() == 'autonomous-start':
                        if self.autonomous_enabled and self.autonomous_integration:
                            success = self.autonomous_integration.start_autonomous_system()
                            if success:
                                print("✅ Autonomous system started successfully")
                            else:
                                print("❌ Failed to start autonomous system")
                        else:
                            print("❌ Autonomous system not available")
                    
                    elif user_input.lower() == 'autonomous-stop':
                        if self.autonomous_enabled and self.autonomous_integration:
                            self.autonomous_integration.stop_autonomous_system()
                            print("✅ Autonomous system stopped")
                        else:
                            print("❌ Autonomous system not available")
                    
                    elif user_input.lower() == 'autonomous-restart':
                        if self.autonomous_enabled and self.autonomous_integration:
                            success = self.autonomous_integration.restart_autonomous_system()
                            if success:
                                print("✅ Autonomous system restarted successfully")
                            else:
                                print("❌ Failed to restart autonomous system")
                        else:
                            print("❌ Autonomous system not available")
                    
                    elif user_input.startswith('math '):
                        # Handle mathematical requests
                        math_request = user_input[5:].strip()
                        self.process_math_request(math_request)
                    
                    else:
                        # Check if full system is loaded
                        if self.system is not None:
                            # Full system is loaded, use it for all requests
                            print("🔄 Processing with full system...")
                            response = self.system.process_request(user_input)
                            
                            # Add to history
                            self.session_history.append({
                                "timestamp": datetime.now().isoformat(),
                                "request": user_input,
                                "pillar": "auto-detected",
                                "response": response
                            })
                            
                            print(f"🤖 Quark: {response}")
                        else:
                            # Use lightweight processing
                            response = self.process_request(user_input)
                            print(f"🤖 Quark: {response}")
                            
                            # Add to history
                            self.session_history.append({
                                "timestamp": datetime.now().isoformat(),
                                "request": user_input,
                                "pillar": "lightweight",
                                "response": response
                            })
                
            except KeyboardInterrupt:
                if cli_mode:
                    click.secho("\n👋 Goodbye!", fg="blue", bold=True)
                else:
                    print("\n👋 Goodbye from Quark!")
                break
            except EOFError:
                print("\n👋 Goodbye from Quark!")
                break
            except Exception as e:
                if cli_mode:
                    click.secho(f"❌ Unexpected error: {e}", fg="red", bold=True)
                else:
                    print(f"❌ Error: {e}")
                # Don't continue the loop on unexpected errors to prevent infinite loops
                break

def main():
    """Main entry point with fresh environment initialization"""
    # Ensure fresh environment before any processing
    ensure_fresh_environment()
    
    parser = argparse.ArgumentParser(description="Quark - Complete AI System")
    parser.add_argument("--test", action="store_true", help="Run integration test and exit")
    parser.add_argument("--status", action="store_true", help="Show system status and exit")
    parser.add_argument("--request", help="Process single request and exit")
    parser.add_argument("--pillar", help="Use specific pillar for request")
    parser.add_argument("--code", help="Execute code with natural language")
    parser.add_argument("--shell", help="Execute shell command")
    parser.add_argument("--python", help="Generate and execute Python code")
    parser.add_argument("--autogen", help="Show AutoGen status")
    parser.add_argument("--workflow", help="Execute AutoGen workflow")
    parser.add_argument("--math", help="Solve mathematical problem")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode with enhanced interface")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode (for startup service)")
    parser.add_argument("--fresh", action="store_true", help="Force fresh environment (default)")
    
    args = parser.parse_args()
    
    try:
        quark = QuarkTerminal()
        
        if args.test:
            quark.run_integration_test()
        elif args.status:
            quark.show_status()
        elif args.request:
            if args.pillar:
                quark.process_pillar_request(args.pillar, args.request)
            else:
                # Use the new lightweight processing approach
                response = quark.process_request(args.request)
                if response:
                    print(f"🤖 Quark: {response}")
        elif args.code:
            quark.process_code_request(args.code)
        elif args.shell:
            quark.process_code_request(f"shell command: {args.shell}")
        elif args.python:
            quark.process_code_request(f"python code: {args.python}")
        elif args.autogen:
            quark.show_autogen_status()
        elif args.workflow:
            quark.execute_autogen_workflow(args.workflow)
        elif args.math:
            quark.process_math_request(args.math)
        elif args.daemon:
            # Daemon mode - just initialize and keep running
            print("🤖 Quark AI Assistant Daemon Mode")
            print("✅ Model is ready for input")
            
            # Keep the process running
            try:
                while True:
                    time.sleep(30)  # Check every 30 seconds
            except KeyboardInterrupt:
                print("👋 Daemon shutting down")
        else:
            # Start interactive terminal (CLI mode if specified)
            quark.run(cli_mode=args.cli)
            
    except KeyboardInterrupt:
        print("\n👋 Quark shutting down gracefully...")
        cleanup_resources()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        cleanup_resources()
        sys.exit(1)

if __name__ == "__main__":
    main() 