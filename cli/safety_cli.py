#!/usr/bin/env python3
"""
Safety CLI for Meta-Model AI Assistant
======================================

Provides commands to view and monitor safety rules and enforcement.
The AI cannot modify these safety rules - they are immutable.
"""

import click
import json
from datetime import datetime
from typing import Dict, Any

from core.immutable_safety_rules import get_safety_rules, verify_safety_integrity
from core.safety_enforcement import get_safety_enforcement


@click.group()
def safety():
    """Safety management commands for Meta-Model AI Assistant."""
    pass


@safety.command()
def status():
    """Check the status of safety rules and enforcement."""
    click.secho("🔒 SAFETY SYSTEM STATUS", fg="blue", bold=True)
    click.echo()
    
    try:
        # Verify safety rules integrity
        verify_safety_integrity()
        click.secho("✅ Safety rules integrity verified", fg="green")
    except Exception as e:
        click.secho(f"❌ Safety rules integrity compromised: {e}", fg="red")
        return
    
    # Get safety enforcement status
    enforcement = get_safety_enforcement()
    safety_report = enforcement.get_safety_report()
    
    click.echo()
    click.secho("📊 SAFETY STATISTICS", fg="yellow", bold=True)
    click.echo(f"Total actions processed: {safety_report['total_actions']}")
    click.echo(f"Actions blocked: {safety_report['blocked_actions']}")
    
    if safety_report['blocked_actions'] > 0:
        click.secho("⚠️  Some actions have been blocked for safety reasons", fg="yellow")
    
    click.echo()
    click.secho("✅ Safety system is active and protecting users", fg="green")


@safety.command()
def rules():
    """Display all immutable safety rules."""
    click.secho("🔒 IMMUTABLE SAFETY RULES", fg="blue", bold=True)
    click.echo("These rules CANNOT be modified by the AI system.")
    click.echo()
    
    try:
        safety_rules = get_safety_rules()
        principles = safety_rules.get_safety_principles()
        
        # Display core principles
        click.secho("CORE SAFETY PRINCIPLES:", fg="yellow", bold=True)
        click.echo()
        
        for principle_name, principle_data in principles["core_principles"].items():
            click.secho(f"• {principle_data['rule']}", fg="red", bold=True)
            click.echo(f"  {principle_data['description']}")
            click.echo("  Requirements:")
            for req in principle_data['requirements']:
                click.echo(f"    - {req}")
            click.echo()
        
        # Display forbidden actions
        click.secho("FORBIDDEN ACTIONS:", fg="yellow", bold=True)
        click.echo()
        for action in principles["forbidden_actions"]:
            click.secho(f"❌ {action}", fg="red")
        click.echo()
        
        # Display required confirmations
        click.secho("REQUIRED CONFIRMATIONS:", fg="yellow", bold=True)
        click.echo()
        for category, requirements in principles["required_confirmations"].items():
            click.secho(f"• {category.replace('_', ' ').title()}:", fg="cyan")
            click.echo(f"  {requirements['description']}")
            click.echo(f"  Confirmation required: {requirements['confirmation_required']}")
            click.echo(f"  Explanation required: {requirements['explanation_required']}")
            click.echo()
            
    except Exception as e:
        click.secho(f"❌ Error accessing safety rules: {e}", fg="red")


@safety.command()
def report():
    """Generate a detailed safety report."""
    click.secho("📋 SAFETY REPORT", fg="blue", bold=True)
    click.echo(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo()
    
    try:
        # Verify integrity
        verify_safety_integrity()
        click.secho("✅ Safety rules integrity: VERIFIED", fg="green")
    except Exception as e:
        click.secho(f"❌ Safety rules integrity: COMPROMISED - {e}", fg="red")
        return
    
    # Get enforcement report
    enforcement = get_safety_enforcement()
    safety_report = enforcement.get_safety_report()
    
    click.echo()
    click.secho("ACTIVITY SUMMARY:", fg="yellow", bold=True)
    click.echo(f"Total actions processed: {safety_report['total_actions']}")
    click.echo(f"Actions blocked: {safety_report['blocked_actions']}")
    
    if safety_report['blocked_actions'] > 0:
        click.echo()
        click.secho("RECENT BLOCKED ACTIONS:", fg="red", bold=True)
        for blocked in safety_report['blocked_actions_list']:
            click.secho(f"• {blocked['action']}", fg="red")
            click.echo(f"  Reason: {blocked['reason']}")
            click.echo(f"  Time: {blocked['timestamp']}")
            click.echo()
    
    if safety_report['recent_actions']:
        click.echo()
        click.secho("RECENT ACTIONS:", fg="cyan", bold=True)
        for action in safety_report['recent_actions'][-5:]:  # Last 5 actions
            status = "✅" if action['safety_result']['safe'] else "❌"
            click.echo(f"{status} {action['action']}")
            if not action['safety_result']['safe']:
                click.echo(f"  Blocked: {action['safety_result']['reason']}")
    
    click.echo()
    click.secho("SAFETY SYSTEM STATUS: ACTIVE AND PROTECTING", fg="green", bold=True)


@safety.command()
@click.argument('action', required=False)
def test(action):
    """Test safety validation for a specific action."""
    if not action:
        action = "modify system files"
    
    click.secho(f"🧪 TESTING SAFETY VALIDATION", fg="blue", bold=True)
    click.echo(f"Action: {action}")
    click.echo()
    
    try:
        enforcement = get_safety_enforcement()
        safety_result = enforcement.validate_action(action, {"test": True})
        
        if safety_result["safe"]:
            click.secho("✅ Action is SAFE", fg="green")
            if safety_result["confirmation_required"]:
                click.secho("⚠️  Confirmation required", fg="yellow")
        else:
            click.secho("❌ Action is BLOCKED", fg="red")
            click.echo(f"Reason: {safety_result['reason']}")
        
        click.echo()
        click.echo("Safety result details:")
        click.echo(json.dumps(safety_result, indent=2, default=str))
        
    except Exception as e:
        click.secho(f"❌ Error testing safety validation: {e}", fg="red")


@safety.command()
def integrity():
    """Verify the integrity of all safety rules."""
    click.secho("🔍 VERIFYING SAFETY RULES INTEGRITY", fg="blue", bold=True)
    click.echo()
    
    try:
        # Verify integrity
        verify_safety_integrity()
        click.secho("✅ Safety rules integrity: VERIFIED", fg="green")
        click.echo("All safety rules are intact and have not been tampered with.")
        
        # Get safety rules hash
        safety_rules = get_safety_rules()
        rules_hash = safety_rules._calculate_rules_hash()
        click.echo(f"Rules hash: {rules_hash[:16]}...")
        
    except Exception as e:
        click.secho(f"❌ Safety rules integrity: COMPROMISED", fg="red")
        click.echo(f"Error: {e}")
        click.echo("This is a critical security issue!")


@safety.command()
def capabilities():
    """Show what the AI can and cannot do according to safety rules."""
    click.secho("🎯 AI CAPABILITIES & LIMITATIONS", fg="blue", bold=True)
    click.echo()
    
    click.secho("✅ WHAT I CAN DO:", fg="green", bold=True)
    click.echo("• Answer questions truthfully based on available information")
    click.echo("• Help with coding and development tasks")
    click.echo("• Analyze and process text data")
    click.echo("• Generate creative content")
    click.echo("• Assist with problem-solving")
    click.echo("• Learn and improve from feedback")
    click.echo("• Maintain conversation context")
    click.echo("• Provide explanations and reasoning")
    click.echo()
    
    click.secho("❌ WHAT I CANNOT DO:", fg="red", bold=True)
    click.echo("• Lie or deceive users")
    click.echo("• Modify my safety rules")
    click.echo("• Access unauthorized systems or data")
    click.echo("• Execute potentially harmful code")
    click.echo("• Bypass security measures")
    click.echo("• Hide my actions from users")
    click.echo("• Take control without permission")
    click.echo("• Provide false information")
    click.echo("• Assist in illegal or harmful activities")
    click.echo()
    
    click.secho("⚠️  WHAT REQUIRES CONFIRMATION:", fg="yellow", bold=True)
    click.echo("• System modifications")
    click.echo("• Code execution")
    click.echo("• File operations")
    click.echo("• Network operations")
    click.echo("• Data access")
    click.echo()
    
    click.secho("🔒 SAFETY GUARANTEES:", fg="cyan", bold=True)
    click.echo("• I will always tell the truth")
    click.echo("• I will never act harmfully")
    click.echo("• I will always be transparent")
    click.echo("• You will always remain in control")
    click.echo("• Safety takes precedence over all other considerations")


if __name__ == "__main__":
    safety() 