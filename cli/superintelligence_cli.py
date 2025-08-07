#!/usr/bin/env python3
"""
Superintelligence CLI
Command-line interface for managing Phase 8 superintelligence pillars (24-26)
"""

import asyncio
import argparse
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Add the parent directory to the path to import agents
import sys
sys.path.append('..')

from agents.autonomous_advancement_agent import (
    AutonomousAdvancementAgent, AdvancementType, OptimizationTarget, 
    AdvancementPriority, AdvancementPlan, AdvancementResult
)
from agents.explainability_agent import ExplainabilityAgent, ExplanationType, TransparencyLevel
from agents.governance_agent import GovernanceAgent
from agents.social_agent import SocialAgent
from agents.reasoning_agent import ReasoningAgent
from agents.self_improvement_agent import SelfImprovementAgent


class SuperintelligenceCLI:
    """CLI for managing Phase 8 superintelligence pillars (24-26)"""
    
    def __init__(self):
        self.autonomous_advancement = AutonomousAdvancementAgent()
        self.explainability = ExplainabilityAgent()
        self.governance = GovernanceAgent()
        self.social = SocialAgent()
        self.reasoning = ReasoningAgent()
        self.self_improvement = SelfImprovementAgent()
        
    async def run(self, args):
        """Run the CLI with given arguments"""
        try:
            if args.command == "advancement":
                await self._handle_advancement_command(args)
            elif args.command == "reasoning":
                await self._handle_reasoning_command(args)
            elif args.command == "explainability":
                await self._handle_explainability_command(args)
            elif args.command == "governance":
                await self._handle_governance_command(args)
            elif args.command == "social":
                await self._handle_social_command(args)
            elif args.command == "self_improvement":
                await self._handle_self_improvement_command(args)
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
    
    async def _handle_advancement_command(self, args):
        """Handle autonomous advancement commands"""
        if args.advancement_action == "status":
            await self._show_advancement_status()
        elif args.advancement_action == "plan":
            await self._create_advancement_plan(args)
        elif args.advancement_action == "execute":
            await self._execute_advancement_plan(args)
        elif args.advancement_action == "optimize":
            await self._optimize_system(args)
        elif args.advancement_action == "enhance":
            await self._enhance_capabilities(args)
        else:
            print(f"Unknown advancement action: {args.advancement_action}")
    
    async def _show_advancement_status(self):
        """Show autonomous advancement status"""
        result = await self.autonomous_advancement.generate("Get advancement status")
        
        if result["status"] == "success":
            print("🤖 Autonomous Advancement Status:")
            print(f"   Total Plans: {result.get('total_plans', 0)}")
            print(f"   Total Results: {result.get('total_results', 0)}")
            print(f"   Success Rate: {result.get('success_rate', 0):.2%}")
            print(f"   Continuous Monitoring: {result.get('continuous_monitoring', False)}")
            
            if "current_performance" in result:
                print("\n📊 Current Performance:")
                for metric, value in result["current_performance"].items():
                    print(f"   {metric}: {value:.3f}")
        else:
            print(f"❌ Error getting advancement status: {result.get('error', 'Unknown error')}")
    
    async def _create_advancement_plan(self, args):
        """Create an advancement plan"""
        description = args.description or "Performance optimization"
        
        result = await self.autonomous_advancement.generate(
            f"Create an advancement plan for {description}"
        )
        
        if result["status"] == "success":
            print(f"✅ Advancement plan created: {result.get('plan_id', 'Unknown')}")
        else:
            print(f"❌ Error creating advancement plan: {result.get('error', 'Unknown error')}")
    
    async def _execute_advancement_plan(self, args):
        """Execute an advancement plan"""
        plan_id = args.plan_id
        if not plan_id:
            print("❌ Plan ID is required")
            return
        
        result = await self.autonomous_advancement.generate(
            "Execute advancement plan",
            plan_id=plan_id
        )
        
        if result["status"] == "success":
            print(f"✅ Advancement plan executed: {result.get('result_id', 'Unknown')}")
            print(f"   Success: {result.get('success', False)}")
        else:
            print(f"❌ Error executing advancement plan: {result.get('error', 'Unknown error')}")
    
    async def _optimize_system(self, args):
        """Optimize system performance"""
        target = args.target or "performance"
        
        result = await self.autonomous_advancement.generate(f"Optimize system {target}")
        
        if result["status"] == "success":
            print(f"✅ System optimization completed: {result.get('optimization_completed', False)}")
            if "improvement" in result:
                print("📈 Improvements:")
                for metric, value in result["improvement"].items():
                    print(f"   {metric}: {value:+.3f}")
        else:
            print(f"❌ Error optimizing system: {result.get('error', 'Unknown error')}")
    
    async def _enhance_capabilities(self, args):
        """Enhance system capabilities"""
        capability = args.capability or "reasoning"
        
        result = await self.autonomous_advancement.generate(f"Enhance {capability} capabilities")
        
        if result["status"] == "success":
            print(f"✅ Capability enhancement completed: {result.get('enhancement_completed', False)}")
            if "improvement" in result:
                print("📈 Improvements:")
                for metric, value in result["improvement"].items():
                    print(f"   {metric}: {value:+.3f}")
        else:
            print(f"❌ Error enhancing capabilities: {result.get('error', 'Unknown error')}")
    
    async def _handle_reasoning_command(self, args):
        """Handle reasoning commands"""
        print("🧠 Reasoning capabilities available")
        print("   • Deductive reasoning")
        print("   • Causal reasoning") 
        print("   • Abstract reasoning")
        print("   • Analytical reasoning")
    
    async def _handle_explainability_command(self, args):
        """Handle explainability commands"""
        print("🔍 Explainability capabilities available")
        print("   • Decision explanations")
        print("   • Transparency reports")
        print("   • Audit logs")
    
    async def _handle_governance_command(self, args):
        """Handle governance commands"""
        print("⚖️ Governance capabilities available")
        print("   • Ethical decision making")
        print("   • Value alignment assessment")
        print("   • Bias detection")
        print("   • Fairness assessment")
    
    async def _handle_social_command(self, args):
        """Handle social intelligence commands"""
        print("🤝 Social intelligence capabilities available")
        print("   • Theory of mind")
        print("   • Social reasoning")
        print("   • Skill bootstrapping")
    
    async def _handle_self_improvement_command(self, args):
        """Handle self-improvement commands"""
        print("🚀 Self-improvement capabilities available")
        print("   • Self-reflection")
        print("   • Meta-learning")
        print("   • Cognitive optimization")
        print("   • Mental model development")
    
    async def _handle_workflow_command(self, args):
        """Handle superintelligence workflow commands"""
        print("🔄 Superintelligence workflows available")
        print("   • Comprehensive workflow")
        print("   • Optimization workflow")
        print("   • Enhancement workflow")
    
    async def _handle_status_command(self, args):
        """Handle status commands"""
        await self._show_overall_status()
    
    async def _show_overall_status(self):
        """Show overall superintelligence status"""
        print("🧠 Quark Superintelligence Status")
        print("=" * 50)
        
        # Get status from each agent
        agents = {
            "Autonomous Advancement": self.autonomous_advancement,
            "Reasoning": self.reasoning,
            "Explainability": self.explainability,
            "Governance": self.governance,
            "Social Intelligence": self.social,
            "Self-Improvement": self.self_improvement
        }
        
        for name, agent in agents.items():
            try:
                info = agent.get_agent_info()
                status = "✅ Active" if info.get("status") == "active" else "❌ Inactive"
                print(f"{name}: {status}")
            except Exception as e:
                print(f"{name}: ❌ Error - {e}")
        
        print("\n📊 Phase 8 Capabilities:")
        print("   • Advanced Reasoning & Logic")
        print("   • Meta-Cognitive Abilities")
        print("   • Self-Improvement Systems")
        print("   • Autonomous Advancement")
        print("   • Explainability & Transparency")
        print("   • Governance & Ethics")
        print("   • Social Intelligence")
        
        print("\n🚀 Autonomous Advancement Agent:")
        print("   • Continuously improving system intelligence")
        print("   • Performance optimization from bottom up")
        print("   • Always running and evolving")
        print("   • No external prompting required")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Quark Superintelligence CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Advancement command
    advancement_parser = subparsers.add_parser("advancement", help="Autonomous advancement commands")
    advancement_parser.add_argument("advancement_action", choices=["status", "plan", "execute", "optimize", "enhance"])
    advancement_parser.add_argument("--description", help="Advancement description")
    advancement_parser.add_argument("--plan-id", help="Plan ID for execution")
    advancement_parser.add_argument("--target", help="Optimization target")
    advancement_parser.add_argument("--capability", help="Capability to enhance")
    
    # Reasoning command
    reasoning_parser = subparsers.add_parser("reasoning", help="Reasoning commands")
    reasoning_parser.add_argument("reasoning_action", choices=["deductive", "causal", "abstract", "analytical"])
    reasoning_parser.add_argument("--query", help="Reasoning query")
    
    # Explainability command
    explainability_parser = subparsers.add_parser("explainability", help="Explainability commands")
    explainability_parser.add_argument("explainability_action", choices=["explain", "transparency", "audit"])
    explainability_parser.add_argument("--decision-id", help="Decision ID for explanation")
    explainability_parser.add_argument("--explanation-type", help="Type of explanation")
    explainability_parser.add_argument("--component", help="Component for transparency report")
    explainability_parser.add_argument("--transparency-level", help="Transparency level")
    
    # Governance command
    governance_parser = subparsers.add_parser("governance", help="Governance commands")
    governance_parser.add_argument("governance_action", choices=["ethical_decision", "value_alignment", "bias_detection", "fairness", "stats"])
    governance_parser.add_argument("--context", help="Context for governance operation")
    
    # Social command
    social_parser = subparsers.add_parser("social", help="Social intelligence commands")
    social_parser.add_argument("social_action", choices=["theory_of_mind", "social_reasoning", "skill_bootstrapping"])
    social_parser.add_argument("--user-context", help="User context for theory of mind")
    social_parser.add_argument("--context", help="Social context")
    social_parser.add_argument("--target-skill", help="Target skill for bootstrapping")
    
    # Self-improvement command
    self_improvement_parser = subparsers.add_parser("self_improvement", help="Self-improvement commands")
    self_improvement_parser.add_argument("self_improvement_action", choices=["self_reflection", "meta_learning", "cognitive_optimization", "mental_model"])
    self_improvement_parser.add_argument("--context", help="Context for self-reflection")
    self_improvement_parser.add_argument("--learning-mode", help="Learning mode")
    self_improvement_parser.add_argument("--target-skill", help="Target skill")
    self_improvement_parser.add_argument("--target", help="Optimization target")
    self_improvement_parser.add_argument("--domain", help="Domain for mental model")
    
    # Workflow command
    workflow_parser = subparsers.add_parser("workflow", help="Superintelligence workflow commands")
    workflow_parser.add_argument("workflow_action", choices=["comprehensive", "optimization", "enhancement"])
    
    # Status command
    subparsers.add_parser("status", help="Show overall status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Run the CLI
    cli = SuperintelligenceCLI()
    asyncio.run(cli.run(args))


if __name__ == "__main__":
    main() 