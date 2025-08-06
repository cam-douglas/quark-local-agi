#!/usr/bin/env python3
import os
import click
import logging
import json
import threading
import time
import sys
from pathlib import Path

# Silence HF device logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Import using absolute paths (project root is already in sys.path)
from core.orchestrator import Orchestrator
from config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class MetaModelCLI:
    def __init__(self):
        self.orchestrator = Orchestrator()
        self.available_models = {
            "intent": "facebook/bart-large-mnli",
            "ner": "elastic/distilbert-base-cased-finetuned-conll03-english",
            "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
            "embeddings": "sentence-transformers/all-distilroberta-v1",
            "generation": "google/flan-t5-small",
            "summarization": "sshleifer/distilbart-cnn-12-6",
            "translation": "Helsinki-NLP/opus-mt-en-de"
        }
    
    def show_welcome(self):
        """Display welcome message and available commands."""
        click.secho("🤖 Meta-Model AI Assistant", fg="magenta", bold=True)
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
        click.secho("")
        click.secho("Commands:", fg="green")
        click.secho("• Type your question or request", fg="white")
        click.secho("• 'models' - List available models", fg="white")
        click.secho("• 'safety' - Safety system commands", fg="white")
        click.secho("• 'help' - Show this help", fg="white")
        click.secho("• 'exit' or 'quit' - Exit", fg="white")
        click.secho("")
    
    def show_models(self):
        """Display available models."""
        click.secho("🤖 Available Models:", fg="magenta", bold=True)
        click.secho("=" * 30, fg="cyan")
        
        for model_type, model_name in self.available_models.items():
            click.secho(f"• {model_type.title()}: {model_name}", fg="yellow")
        click.secho("")
    
    def show_help(self):
        """Display help information."""
        click.secho("📖 Help & Examples:", fg="magenta", bold=True)
        click.secho("=" * 30, fg="cyan")
        click.secho("Examples:", fg="green")
        click.secho("• 'What is the weather like?'", fg="white")
        click.secho("• 'Summarize this text: [your text]'", fg="white")
        click.secho("• 'Translate to German: Hello world'", fg="white")
        click.secho("• 'Analyze sentiment: I love this product'", fg="white")
        click.secho("• 'Extract entities: Apple Inc. is headquartered in Cupertino'", fg="white")
        click.secho("• 'Generate text about: artificial intelligence'", fg="white")
        click.secho("")
        click.secho("Commands:", fg="green")
        click.secho("• 'models' - Show available models", fg="white")
        click.secho("• 'safety' - Safety system commands", fg="white")
        click.secho("• 'help' - Show this help", fg="white")
        click.secho("• 'exit' or 'quit' - Exit", fg="white")
        click.secho("")
    
    def show_safety_help(self):
        """Display safety system help information."""
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
    
    def process_input(self, user_input):
        """Process user input and return response."""
        try:
            # Check for special commands
            if user_input.lower() in ["exit", "quit"]:
                return {"command": "exit"}
            elif user_input.lower() == "models":
                return {"command": "models"}
            elif user_input.lower() == "help":
                return {"command": "help"}
            elif user_input.lower() == "safety":
                return {"command": "safety"}
            
            # Process with orchestrator
            response = self.orchestrator.handle(user_input)
            return {"response": response}
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return {"error": f"Error processing request: {str(e)}"}
    
    def display_response(self, result):
        """Display the response in a formatted way."""
        if "command" in result:
            if result["command"] == "exit":
                click.secho("👋 Goodbye!", fg="blue", bold=True)
                return False
            elif result["command"] == "models":
                self.show_models()
                return True
            elif result["command"] == "help":
                self.show_help()
                return True
            elif result["command"] == "safety":
                self.show_safety_help()
                return True
        
        elif "error" in result:
            click.secho(f"❌ Error: {result['error']}", fg="red", bold=True)
            return True
        
        elif "response" in result:
            response = result["response"]
            
            if response.get("error"):
                click.secho(f"❌ Error: {response['error']}", fg="red", bold=True)
                return True
            
            # Display category
            category = response.get("category", "Unknown")
            click.secho(f"📋 Category: {category}", fg="cyan", bold=True)
            
            # Display results
            results = response.get("results", {})
            for task, output in results.items():
                if isinstance(output, dict) and output.get("error"):
                    click.secho(f"  ❌ [{task}] Error: {output['error']}", fg="red")
                else:
                    # Format output for better display
                    if isinstance(output, str):
                        display_output = output[:200] + "..." if len(output) > 200 else output
                    elif isinstance(output, (list, tuple)):
                        display_output = str(output[:3]) + "..." if len(output) > 3 else str(output)
                    else:
                        display_output = str(output)
                    
                    click.secho(f"  ✅ [{task}] {display_output}", fg="green")
            
            return True
        
        return True
    
    def run(self):
        """Main CLI loop."""
        self.show_welcome()
        
        while True:
            try:
                # Get user input
                user_input = click.prompt(
                    click.style("🤖", fg="yellow"),
                    type=str,
                    default=""
                ).strip()
                
                if not user_input:
                    continue
                
                # Show thinking indicator
                with click.progressbar(
                    length=1,
                    label=click.style("Thinking", fg="blue"),
                    show_eta=False
                ) as bar:
                    result = self.process_input(user_input)
                    bar.update(1)
                
                # Display response
                if not self.display_response(result):
                    break
                
                click.secho("")  # Add spacing
                
            except (EOFError, KeyboardInterrupt):
                click.secho("\n👋 Goodbye!", fg="blue", bold=True)
                break
            except Exception as e:
                click.secho(f"❌ Unexpected error: {e}", fg="red", bold=True)
                logger.error(f"Unexpected error: {e}")

@click.command()
@click.option('--daemon', is_flag=True, help='Run in daemon mode (for startup service)')
def cli(daemon):
    """Meta-Model AI Assistant CLI."""
    cli_instance = MetaModelCLI()
    
    if daemon:
        # Daemon mode - just initialize and keep running
        click.secho("🤖 Meta-Model AI Assistant Daemon Mode", fg="magenta", bold=True)
        click.secho("✅ Model is ready for input", fg="green", bold=True)
        
        # Keep the process running
        try:
            while True:
                time.sleep(30)  # Check every 30 seconds
        except KeyboardInterrupt:
            click.secho("👋 Daemon shutting down", fg="blue", bold=True)
    else:
        # Interactive mode
        cli_instance.run()

if __name__ == "__main__":
    cli()

