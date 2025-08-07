#!/usr/bin/env python3
"""
Code Generation CLI
Interactive command-line interface for Quark's code generation capabilities

This provides a CLI for easy interaction with the code generation agent,
similar to how you would interact with Claude for code assistance.
"""

import asyncio
import sys
import os
from pathlib import Path
import argparse
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from agents.code_generation_agent import CodeGenerationAgent, CodeGenerationRequest
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
except ImportError as e:
    print(f"Error: Missing dependencies. Please install: pip install rich")
    print(f"Import error: {e}")
    sys.exit(1)

console = Console()

class CodeGenerationCLI:
    """Interactive CLI for code generation"""
    
    def __init__(self):
        self.agent = CodeGenerationAgent()
        self.session_history = []
        
    def print_banner(self):
        """Print the CLI banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🤖 Quark Code Assistant                   ║
║                  Claude-like Code Generation                 ║
╚══════════════════════════════════════════════════════════════╝
        """
        console.print(banner, style="bold cyan")
        console.print("Type 'help' for commands, 'quit' to exit\n", style="dim")
    
    def print_help(self):
        """Print help information"""
        help_text = """
Available Commands:
─────────────────────────────────────────────────────────────

• complete <code>        - Complete partial code
• generate <description> - Generate code from description  
• explain <code>         - Explain existing code
• refactor <code>        - Refactor code for better quality
• file <path>            - Process code from file
• lang <language>        - Set programming language (default: python)
• models                 - Show available models
• history                - Show session history
• clear                  - Clear session history
• help                   - Show this help
• quit                   - Exit the CLI

Interactive Mode:
─────────────────────────────────────────────────────────────

You can also just type your request naturally:
• "Write a function to sort a list"
• "Complete this: def fibonacci(n):"
• "Explain this code: [paste code]"
• "Refactor this function to be more efficient"

Supported Languages:
─────────────────────────────────────────────────────────────
Python, JavaScript, TypeScript, Java, C++, Go, Rust, and more!
        """
        console.print(Panel(help_text, title="Help", border_style="blue"))
    
    async def handle_command(self, user_input: str) -> bool:
        """Handle user commands"""
        if not user_input.strip():
            return True
            
        parts = user_input.strip().split(None, 1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        try:
            if command == "quit" or command == "exit":
                return False
            elif command == "help":
                self.print_help()
            elif command == "clear":
                self.session_history.clear()
                console.print("✅ Session history cleared", style="green")
            elif command == "history":
                self.show_history()
            elif command == "models":
                self.show_models()
            elif command == "complete":
                await self.complete_code(args)
            elif command == "generate":
                await self.generate_code(args)
            elif command == "explain":
                await self.explain_code(args)
            elif command == "refactor":
                await self.refactor_code(args)
            elif command == "file":
                await self.process_file(args)
            elif command == "lang":
                self.set_language(args)
            else:
                # Treat as natural language request
                await self.handle_natural_request(user_input)
                
        except Exception as e:
            console.print(f"❌ Error: {e}", style="red")
            
        return True
    
    async def handle_natural_request(self, request: str):
        """Handle natural language requests"""
        request_lower = request.lower()
        
        # Determine intent
        if any(word in request_lower for word in ["complete", "finish"]):
            await self.complete_code(request)
        elif any(word in request_lower for word in ["generate", "create", "write", "make"]):
            await self.generate_code(request)
        elif any(word in request_lower for word in ["explain", "what does", "how does"]):
            await self.explain_code(request)
        elif any(word in request_lower for word in ["refactor", "improve", "optimize"]):
            await self.refactor_code(request)
        else:
            # Default to generation
            await self.generate_code(request)
    
    async def complete_code(self, partial_code: str):
        """Complete partial code"""
        if not partial_code.strip():
            partial_code = Prompt.ask("Enter partial code to complete")
        
        console.print("🔄 Completing code...", style="yellow")
        
        try:
            result = await self.agent.complete_code(partial_code, self.current_language)
            
            self.display_code_result("Code Completion", partial_code, result)
            self.session_history.append({
                "type": "completion",
                "input": partial_code,
                "output": result,
                "language": self.current_language
            })
            
        except Exception as e:
            console.print(f"❌ Error completing code: {e}", style="red")
    
    async def generate_code(self, description: str):
        """Generate code from description"""
        if not description.strip():
            description = Prompt.ask("Enter description of what you want to generate")
        
        console.print("🔄 Generating code...", style="yellow")
        
        try:
            result = await self.agent.generate_from_description(description, self.current_language)
            
            self.display_code_result("Generated Code", description, result)
            self.session_history.append({
                "type": "generation",
                "input": description,
                "output": result,
                "language": self.current_language
            })
            
        except Exception as e:
            console.print(f"❌ Error generating code: {e}", style="red")
    
    async def explain_code(self, code: str):
        """Explain existing code"""
        if not code.strip():
            code = Prompt.ask("Enter code to explain")
        
        console.print("🔄 Explaining code...", style="yellow")
        
        try:
            explanation = await self.agent.get_code_explanation(code, self.current_language)
            
            console.print(Panel(
                Syntax(code, self.current_language, theme="monokai", line_numbers=True),
                title="Code to Explain",
                border_style="blue"
            ))
            
            console.print(Panel(
                explanation,
                title="Explanation",
                border_style="green"
            ))
            
            self.session_history.append({
                "type": "explanation",
                "input": code,
                "output": explanation,
                "language": self.current_language
            })
            
        except Exception as e:
            console.print(f"❌ Error explaining code: {e}", style="red")
    
    async def refactor_code(self, code: str):
        """Refactor existing code"""
        if not code.strip():
            code = Prompt.ask("Enter code to refactor")
        
        console.print("🔄 Refactoring code...", style="yellow")
        
        try:
            refactored = await self.agent.refactor_code(code, self.current_language)
            
            console.print(Panel(
                Syntax(code, self.current_language, theme="monokai", line_numbers=True),
                title="Original Code",
                border_style="red"
            ))
            
            console.print(Panel(
                Syntax(refactored, self.current_language, theme="monokai", line_numbers=True),
                title="Refactored Code",
                border_style="green"
            ))
            
            self.session_history.append({
                "type": "refactoring",
                "input": code,
                "output": refactored,
                "language": self.current_language
            })
            
        except Exception as e:
            console.print(f"❌ Error refactoring code: {e}", style="red")
    
    async def process_file(self, file_path: str):
        """Process code from a file"""
        if not file_path.strip():
            file_path = Prompt.ask("Enter file path")
        
        try:
            path = Path(file_path)
            if not path.exists():
                console.print(f"❌ File not found: {file_path}", style="red")
                return
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Detect language from file extension
            extension_map = {
                '.py': 'python',
                '.js': 'javascript',
                '.ts': 'typescript',
                '.java': 'java',
                '.cpp': 'cpp',
                '.c': 'c',
                '.go': 'go',
                '.rs': 'rust',
                '.php': 'php',
                '.rb': 'ruby'
            }
            
            detected_lang = extension_map.get(path.suffix.lower(), self.current_language)
            
            console.print(f"📁 Processing file: {file_path} ({detected_lang})", style="cyan")
            
            action = Prompt.ask(
                "What would you like to do?", 
                choices=["explain", "refactor", "complete"],
                default="explain"
            )
            
            if action == "explain":
                await self.explain_code(content)
            elif action == "refactor":
                await self.refactor_code(content)
            elif action == "complete":
                await self.complete_code(content)
                
        except Exception as e:
            console.print(f"❌ Error processing file: {e}", style="red")
    
    def set_language(self, language: str):
        """Set the current programming language"""
        if not language.strip():
            language = Prompt.ask("Enter programming language", default="python")
        
        self.current_language = language.lower()
        console.print(f"✅ Language set to: {self.current_language}", style="green")
    
    def show_models(self):
        """Show available models"""
        model_info = self.agent.get_model_info()
        
        info_text = f"""
Loaded Models: {', '.join(model_info['loaded_models']) or 'None (using fallback)'}
Device: {model_info['device']}
Transformers Available: {model_info['transformers_available']}

Available Model Configurations:
"""
        for name, config in model_info['available_models'].items():
            info_text += f"\n• {name}: {config['description']}"
            info_text += f"\n  Specialties: {', '.join(config['specialties'])}"
            info_text += f"\n  Max Length: {config['max_length']}\n"
        
        console.print(Panel(info_text, title="Model Information", border_style="cyan"))
    
    def show_history(self):
        """Show session history"""
        if not self.session_history:
            console.print("No history available", style="dim")
            return
        
        console.print(Panel("Session History", border_style="cyan"))
        
        for i, item in enumerate(self.session_history[-10:], 1):  # Show last 10
            console.print(f"\n{i}. {item['type'].title()} ({item['language']})")
            console.print(f"   Input: {item['input'][:100]}...")
            console.print(f"   Output: {item['output'][:100]}...")
    
    def display_code_result(self, title: str, input_text: str, output_code: str):
        """Display code generation results"""
        if input_text != output_code:  # Don't show input if it's the same as output
            console.print(Panel(
                input_text,
                title="Input",
                border_style="blue"
            ))
        
        console.print(Panel(
            Syntax(output_code, self.current_language, theme="monokai", line_numbers=True),
            title=title,
            border_style="green"
        ))
    
    async def run_interactive(self):
        """Run the interactive CLI"""
        self.current_language = "python"
        self.print_banner()
        
        # Show model status
        model_info = self.agent.get_model_info()
        if model_info['loaded_models']:
            console.print(f"✅ Models loaded: {', '.join(model_info['loaded_models'])}", style="green")
        else:
            console.print("⚠️  Using fallback completion (no models loaded)", style="yellow")
        
        console.print()
        
        while True:
            try:
                user_input = Prompt.ask(
                    f"[bold cyan]Quark[/bold cyan] ({self.current_language})",
                    default=""
                )
                
                if not await self.handle_command(user_input):
                    break
                    
            except KeyboardInterrupt:
                console.print("\n\n👋 Goodbye!", style="cyan")
                break
            except EOFError:
                console.print("\n\n👋 Goodbye!", style="cyan")
                break

async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Quark Code Generation CLI")
    parser.add_argument("--complete", help="Complete code from string")
    parser.add_argument("--generate", help="Generate code from description")
    parser.add_argument("--explain", help="Explain code")
    parser.add_argument("--refactor", help="Refactor code")
    parser.add_argument("--file", help="Process code from file")
    parser.add_argument("--language", default="python", help="Programming language")
    
    args = parser.parse_args()
    
    cli = CodeGenerationCLI()
    cli.current_language = args.language
    
    # Non-interactive mode
    if any([args.complete, args.generate, args.explain, args.refactor, args.file]):
        if args.complete:
            await cli.complete_code(args.complete)
        elif args.generate:
            await cli.generate_code(args.generate)
        elif args.explain:
            await cli.explain_code(args.explain)
        elif args.refactor:
            await cli.refactor_code(args.refactor)
        elif args.file:
            await cli.process_file(args.file)
    else:
        # Interactive mode
        await cli.run_interactive()

if __name__ == "__main__":
    asyncio.run(main())