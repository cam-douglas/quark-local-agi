"""
CLI Interface for Multi-Agent Negotiation & Coordination
Pillar 28: Command-line interface for negotiation agent operations
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import argparse
import uuid

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.negotiation_agent import NegotiationAgent


class NegotiationCLI:
    """Command-line interface for negotiation agent operations"""
    
    def __init__(self):
        self.agent = NegotiationAgent()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for the CLI"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    async def run(self):
        """Run the negotiation CLI"""
        parser = argparse.ArgumentParser(
            description="Multi-Agent Negotiation & Coordination CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python negotiation_cli.py register-agent agent_001 --capabilities text_generation,reasoning
  python negotiation_cli.py propose-task "Generate Report" --capabilities text_generation,reasoning
  python negotiation_cli.py form-group "Test Group" --members agent_001,agent_002
  python negotiation_cli.py make-decision --type consensus --participants agent_001,agent_002
  python negotiation_cli.py stats
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Register agent command
        register_parser = subparsers.add_parser('register-agent', help='Register an agent')
        register_parser.add_argument('agent_id', help='Agent ID')
        register_parser.add_argument('--capabilities', help='Comma-separated list of capabilities')
        register_parser.add_argument('--performance', help='Performance metrics JSON')
        register_parser.add_argument('--resources', help='Resource requirements JSON')
        
        # Propose task command
        propose_parser = subparsers.add_parser('propose-task', help='Propose a task for negotiation')
        propose_parser.add_argument('title', help='Task title')
        propose_parser.add_argument('--description', help='Task description')
        propose_parser.add_argument('--priority', choices=['critical', 'high', 'medium', 'low'], default='medium')
        propose_parser.add_argument('--complexity', type=float, default=0.5, help='Task complexity (0-1)')
        propose_parser.add_argument('--duration', type=float, default=60.0, help='Estimated duration in seconds')
        propose_parser.add_argument('--capabilities', help='Required capabilities (comma-separated)')
        propose_parser.add_argument('--deadline', help='Deadline (ISO format)')
        
        # Form working group command
        group_parser = subparsers.add_parser('form-group', help='Form a working group')
        group_parser.add_argument('name', help='Group name')
        group_parser.add_argument('--purpose', help='Group purpose')
        group_parser.add_argument('--members', help='Comma-separated list of member IDs')
        group_parser.add_argument('--coordinator', help='Coordinator agent ID')
        group_parser.add_argument('--completion', help='Expected completion time (ISO format)')
        
        # Make decision command
        decision_parser = subparsers.add_parser('make-decision', help='Make a collaborative decision')
        decision_parser.add_argument('--type', choices=['consensus', 'majority', 'leader', 'weighted', 'autonomous'], default='consensus')
        decision_parser.add_argument('--participants', help='Comma-separated list of participant IDs')
        decision_parser.add_argument('--options', help='JSON array of options')
        
        # Get status commands
        subparsers.add_parser('stats', help='Show negotiation statistics')
        subparsers.add_parser('agents', help='List registered agents')
        subparsers.add_parser('groups', help='List working groups')
        subparsers.add_parser('negotiations', help='List active negotiations')
        
        # Get specific information
        get_parser = subparsers.add_parser('get', help='Get specific information')
        get_parser.add_argument('type', choices=['agent', 'group', 'negotiation'], help='Type of information to get')
        get_parser.add_argument('id', help='ID of the item to get')
        
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return
        
        try:
            if args.command == 'register-agent':
                await self.register_agent(args)
            elif args.command == 'propose-task':
                await self.propose_task(args)
            elif args.command == 'form-group':
                await self.form_working_group(args)
            elif args.command == 'make-decision':
                await self.make_decision(args)
            elif args.command == 'stats':
                await self.show_stats()
            elif args.command == 'agents':
                await self.list_agents()
            elif args.command == 'groups':
                await self.list_groups()
            elif args.command == 'negotiations':
                await self.list_negotiations()
            elif args.command == 'get':
                await self.get_information(args)
            else:
                print(f"Unknown command: {args.command}")
                
        except Exception as e:
            self.logger.error(f"Error: {str(e)}")
            sys.exit(1)
    
    async def register_agent(self, args):
        """Register an agent"""
        capabilities = []
        if args.capabilities:
            for cap_name in args.capabilities.split(','):
                capabilities.append({
                    "name": cap_name.strip(),
                    "description": f"Capability: {cap_name}",
                    "confidence": 0.8,
                    "performance_metrics": {"accuracy": 0.85},
                    "resource_requirements": {"memory": "1GB"}
                })
        
        performance_metrics = {}
        if args.performance:
            try:
                performance_metrics = json.loads(args.performance)
            except json.JSONDecodeError:
                print("Error: Invalid performance metrics JSON")
                return
        
        resource_requirements = {}
        if args.resources:
            try:
                resource_requirements = json.loads(args.resources)
            except json.JSONDecodeError:
                print("Error: Invalid resource requirements JSON")
                return
        
        result = await self.agent.process_message({
            "type": "register_agent",
            "agent_id": args.agent_id,
            "capabilities": capabilities,
            "performance_metrics": performance_metrics,
            "resource_requirements": resource_requirements
        })
        
        if result["status"] == "success":
            print(f"✅ Agent {args.agent_id} registered successfully with {result['capabilities_count']} capabilities")
        else:
            print(f"❌ Failed to register agent: {result['message']}")
    
    async def propose_task(self, args):
        """Propose a task for negotiation"""
        required_capabilities = []
        if args.capabilities:
            required_capabilities = [cap.strip() for cap in args.capabilities.split(',')]
        
        deadline = None
        if args.deadline:
            try:
                deadline = datetime.fromisoformat(args.deadline)
            except ValueError:
                print("Error: Invalid deadline format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")
                return
        
        result = await self.agent.process_message({
            "type": "propose_task",
            "task_id": str(uuid.uuid4()),
            "title": args.title,
            "description": args.description or f"Task: {args.title}",
            "priority": args.priority,
            "complexity": args.complexity,
            "estimated_duration": args.duration,
            "required_capabilities": required_capabilities,
            "constraints": {},
            "deadline": deadline,
            "dependencies": []
        })
        
        if result["status"] == "success":
            print(f"✅ Task '{args.title}' proposed successfully")
            print(f"   Negotiation ID: {result['negotiation_id']}")
            print(f"   Suitable agents: {len(result['suitable_agents'])}")
            print(f"   Confidence: {result['proposal']['confidence']:.2f}")
        else:
            print(f"❌ Failed to propose task: {result['message']}")
    
    async def form_working_group(self, args):
        """Form a working group"""
        members = []
        if args.members:
            members = [member.strip() for member in args.members.split(',')]
        
        expected_completion = None
        if args.completion:
            try:
                expected_completion = datetime.fromisoformat(args.completion)
            except ValueError:
                print("Error: Invalid completion time format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")
                return
        
        result = await self.agent.process_message({
            "type": "form_working_group",
            "group_id": str(uuid.uuid4()),
            "name": args.name,
            "purpose": args.purpose or "Collaborative problem solving",
            "members": members,
            "coordinator": args.coordinator,
            "expected_completion": expected_completion
        })
        
        if result["status"] == "success":
            working_group = result["working_group"]
            print(f"✅ Working group '{args.name}' formed successfully")
            print(f"   Group ID: {working_group['group_id']}")
            print(f"   Members: {len(working_group['members'])}")
            print(f"   Coordinator: {working_group['coordinator']}")
            print(f"   Communication channels: {len(working_group['communication_channels'])}")
        else:
            print(f"❌ Failed to form working group: {result['message']}")
    
    async def make_decision(self, args):
        """Make a collaborative decision"""
        participants = []
        if args.participants:
            participants = [p.strip() for p in args.participants.split(',')]
        
        options = []
        if args.options:
            try:
                options = json.loads(args.options)
            except json.JSONDecodeError:
                print("Error: Invalid options JSON")
                return
        else:
            # Default options if none provided
            options = [
                {"id": "option_1", "name": "Option 1", "description": "First option"},
                {"id": "option_2", "name": "Option 2", "description": "Second option"}
            ]
        
        result = await self.agent.process_message({
            "type": "collaborative_decision",
            "decision_id": str(uuid.uuid4()),
            "decision_type": args.type,
            "participants": participants,
            "options": options
        })
        
        if result["status"] == "success":
            decision = result["decision"]
            print(f"✅ Collaborative decision made successfully")
            print(f"   Decision ID: {decision['decision_id']}")
            print(f"   Type: {decision['decision_type']}")
            print(f"   Participants: {len(decision['participants'])}")
            print(f"   Selected option: {decision['selected_option']['name']}")
            print(f"   Confidence: {decision['confidence']:.2f}")
            print(f"   Reasoning: {decision['reasoning']}")
        else:
            print(f"❌ Failed to make decision: {result['message']}")
    
    async def show_stats(self):
        """Show negotiation statistics"""
        stats = await self.agent.get_negotiation_stats()
        
        print("📊 Negotiation Statistics")
        print("=" * 40)
        
        negotiation_stats = stats["stats"]
        print(f"Total negotiations: {negotiation_stats['total_negotiations']}")
        print(f"Successful negotiations: {negotiation_stats['successful_negotiations']}")
        print(f"Average negotiation time: {negotiation_stats['average_negotiation_time']:.2f}s")
        print(f"Working groups formed: {negotiation_stats['working_groups_formed']}")
        print(f"Collaborative decisions: {negotiation_stats['collaborative_decisions']}")
        print()
        print(f"Active negotiations: {stats['active_negotiations']}")
        print(f"Working groups: {stats['working_groups']}")
        print(f"Registered agents: {stats['registered_agents']}")
    
    async def list_agents(self):
        """List registered agents"""
        result = await self.agent.process_message({
            "type": "get_agent_capabilities"
        })
        
        if result["status"] == "success":
            print("🤖 Registered Agents")
            print("=" * 40)
            
            if result["registered_agents"] == 0:
                print("No agents registered")
                return
            
            for agent_id, agent_info in result["agent_registry"].items():
                print(f"Agent: {agent_id}")
                print(f"  Capabilities: {len(agent_info['capabilities'])}")
                print(f"  Status: {agent_info['status']}")
                print(f"  Registered: {agent_info['registration_time']}")
                print()
        else:
            print(f"❌ Failed to get agents: {result['message']}")
    
    async def list_groups(self):
        """List working groups"""
        result = await self.agent.process_message({
            "type": "get_working_groups"
        })
        
        if result["status"] == "success":
            print("👥 Working Groups")
            print("=" * 40)
            
            if result["working_groups_count"] == 0:
                print("No working groups formed")
                return
            
            for group in result["working_groups"]:
                print(f"Group: {group['name']}")
                print(f"  ID: {group['group_id']}")
                print(f"  Purpose: {group['purpose']}")
                print(f"  Members: {len(group['members'])}")
                print(f"  Coordinator: {group['coordinator']}")
                print(f"  Status: {group['status']}")
                print(f"  Progress: {group['progress']:.1%}")
                print()
        else:
            print(f"❌ Failed to get groups: {result['message']}")
    
    async def list_negotiations(self):
        """List active negotiations"""
        result = await self.agent.process_message({
            "type": "get_negotiation_status"
        })
        
        if result["status"] == "success":
            print("🤝 Active Negotiations")
            print("=" * 40)
            
            if result["active_negotiations"] == 0:
                print("No active negotiations")
                return
            
            print(f"Total active negotiations: {result['active_negotiations']}")
            print(f"Statistics: {json.dumps(result['negotiation_stats'], indent=2)}")
        else:
            print(f"❌ Failed to get negotiations: {result['message']}")
    
    async def get_information(self, args):
        """Get specific information"""
        if args.type == "agent":
            result = await self.agent.process_message({
                "type": "get_agent_capabilities",
                "agent_id": args.id
            })
            
            if result["status"] == "success":
                print(f"🤖 Agent: {args.id}")
                print("=" * 40)
                print(f"Capabilities: {len(result['agent_capabilities'])}")
                for cap in result["agent_capabilities"]:
                    print(f"  - {cap['name']}: {cap['description']}")
            else:
                print(f"❌ Failed to get agent: {result['message']}")
        
        elif args.type == "group":
            result = await self.agent.process_message({
                "type": "get_working_groups",
                "group_id": args.id
            })
            
            if result["status"] == "success":
                group = result["working_group"]
                print(f"👥 Group: {group['name']}")
                print("=" * 40)
                print(f"ID: {group['group_id']}")
                print(f"Purpose: {group['purpose']}")
                print(f"Members: {', '.join(group['members'])}")
                print(f"Coordinator: {group['coordinator']}")
                print(f"Status: {group['status']}")
                print(f"Progress: {group['progress']:.1%}")
                print(f"Channels: {len(group['communication_channels'])}")
            else:
                print(f"❌ Failed to get group: {result['message']}")
        
        elif args.type == "negotiation":
            result = await self.agent.process_message({
                "type": "get_negotiation_status",
                "negotiation_id": args.id
            })
            
            if result["status"] == "success":
                negotiation = result["negotiation"]
                print(f"🤝 Negotiation: {args.id}")
                print("=" * 40)
                print(f"Status: {negotiation['status']}")
                print(f"Participants: {len(negotiation['participants'])}")
                print(f"Responses: {negotiation['responses_count']}")
                print(f"Start time: {negotiation['start_time']}")
            else:
                print(f"❌ Failed to get negotiation: {result['message']}")


async def main():
    """Main entry point"""
    cli = NegotiationCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main()) 