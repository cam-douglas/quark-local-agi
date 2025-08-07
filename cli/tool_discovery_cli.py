"""
CLI Interface for Advanced Tool Discovery & Integration
Pillar 29: Command-line interface for tool discovery agent operations
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.tool_discovery_agent import ToolDiscoveryAgent

class ToolDiscoveryCLI:
    """CLI interface for tool discovery and integration"""
    
    def __init__(self):
        self.agent = ToolDiscoveryAgent()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    async def run(self):
        """Main CLI entry point"""
        parser = argparse.ArgumentParser(
            description="Quark Tool Discovery & Integration CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python cli/tool_discovery_cli.py discover --capabilities "machine learning,data visualization"
  python cli/tool_discovery_cli.py evaluate --tool "pandas"
  python cli/tool_discovery_cli.py integrate --tool "numpy"
  python cli/tool_discovery_cli.py stats
  python cli/tool_discovery_cli.py list --type discovered
  python cli/tool_discovery_cli.py info --tool "requests"
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Discover command
        discover_parser = subparsers.add_parser('discover', help='Discover new tools')
        discover_parser.add_argument('--capabilities', type=str, required=True,
                                   help='Comma-separated list of capabilities')
        discover_parser.add_argument('--priority', choices=['critical', 'high', 'medium', 'low'],
                                   default='medium', help='Tool priority level')
        discover_parser.add_argument('--category', choices=['utility', 'ai_model', 'data_processing',
                                                          'communication', 'visualization', 'automation',
                                                          'security', 'monitoring', 'integration', 'custom'],
                                   help='Tool category')
        
        # Evaluate command
        evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate a specific tool')
        evaluate_parser.add_argument('--tool', type=str, required=True,
                                   help='Tool name to evaluate')
        
        # Integrate command
        integrate_parser = subparsers.add_parser('integrate', help='Integrate a specific tool')
        integrate_parser.add_argument('--tool', type=str, required=True,
                                    help='Tool name to integrate')
        
        # Stats command
        stats_parser = subparsers.add_parser('stats', help='Show discovery statistics')
        
        # List command
        list_parser = subparsers.add_parser('list', help='List tools by type')
        list_parser.add_argument('--type', choices=['discovered', 'evaluated', 'integrated'],
                               required=True, help='Type of tools to list')
        list_parser.add_argument('--limit', type=int, default=10,
                               help='Maximum number of tools to show')
        
        # Info command
        info_parser = subparsers.add_parser('info', help='Get detailed information about a tool')
        info_parser.add_argument('--tool', type=str, required=True,
                               help='Tool name to get information about')
        
        # Search command
        search_parser = subparsers.add_parser('search', help='Search for tools')
        search_parser.add_argument('--query', type=str, required=True,
                                 help='Search query')
        search_parser.add_argument('--source', choices=['pypi', 'github', 'huggingface', 'all'],
                                 default='all', help='Search source')
        
        # Continuous discovery command
        continuous_parser = subparsers.add_parser('continuous', help='Start continuous discovery')
        continuous_parser.add_argument('--interval', type=int, default=3600,
                                    help='Discovery interval in seconds')
        
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return
        
        try:
            if args.command == 'discover':
                await self.discover_tools(args)
            elif args.command == 'evaluate':
                await self.evaluate_tool(args)
            elif args.command == 'integrate':
                await self.integrate_tool(args)
            elif args.command == 'stats':
                await self.show_stats()
            elif args.command == 'list':
                await self.list_tools(args)
            elif args.command == 'info':
                await self.show_tool_info(args)
            elif args.command == 'search':
                await self.search_tools(args)
            elif args.command == 'continuous':
                await self.start_continuous_discovery(args)
            else:
                parser.print_help()
        
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            sys.exit(1)
    
    async def discover_tools(self, args):
        """Discover tools based on capabilities"""
        print("🔍 Discovering tools...")
        
        # Parse capabilities
        capabilities = [cap.strip() for cap in args.capabilities.split(',')]
        
        # Create requirements
        requirements = {
            'capabilities': capabilities,
            'priority': args.priority,
            'category': args.category
        }
        
        # Discover tools
        discovered_tools = await self.agent._discover_tools(requirements)
        
        if discovered_tools:
            print(f"✅ Discovered {len(discovered_tools)} tools:")
            for tool in discovered_tools[:5]:  # Show top 5
                print(f"  • {tool.name} ({tool.source}) - {tool.description[:100]}...")
            
            # Evaluate discovered tools
            print("\n📊 Evaluating tools...")
            evaluations = await self.agent._evaluate_tools(discovered_tools)
            
            if evaluations:
                print(f"✅ Evaluated {len(evaluations)} tools:")
                for eval in evaluations[:3]:  # Show top 3
                    print(f"  • {eval.tool_name} - Score: {eval.overall_score:.2f}")
            
            # Generate recommendations
            recommendations = self.agent._generate_recommendations(evaluations)
            
            if recommendations:
                print(f"\n⭐ Top Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"  {i}. {rec['tool_name']} (Score: {rec['overall_score']:.2f})")
                    print(f"     Effort: {rec['integration_effort']}, Time: {rec['estimated_time']}")
        else:
            print("❌ No tools discovered for the specified capabilities.")
    
    async def evaluate_tool(self, args):
        """Evaluate a specific tool"""
        tool_name = args.tool
        print(f"📊 Evaluating tool: {tool_name}")
        
        # Check if tool is discovered
        if tool_name not in self.agent.discovered_tools:
            print(f"❌ Tool '{tool_name}' not found in discovered tools.")
            print("Use 'discover' command first to find tools.")
            return
        
        tool = self.agent.discovered_tools[tool_name]
        evaluation = await self.agent._evaluate_single_tool(tool)
        
        print(f"✅ Evaluation complete for {tool_name}:")
        print(f"  • Overall Score: {evaluation.overall_score:.2f}")
        print(f"  • Functionality: {evaluation.functionality_score:.2f}")
        print(f"  • Performance: {evaluation.performance_score:.2f}")
        print(f"  • Security: {evaluation.security_score:.2f}")
        print(f"  • Compatibility: {evaluation.compatibility_score:.2f}")
        print(f"  • Documentation: {evaluation.documentation_score:.2f}")
        print(f"  • Community: {evaluation.community_score:.2f}")
        print(f"  • Integration Effort: {evaluation.integration_effort}")
        print(f"  • Estimated Time: {evaluation.estimated_time}")
        
        if evaluation.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in evaluation.recommendations:
                print(f"  • {rec}")
        
        if evaluation.risks:
            print(f"\n⚠️  Risks:")
            for risk in evaluation.risks:
                print(f"  • {risk}")
    
    async def integrate_tool(self, args):
        """Integrate a specific tool"""
        tool_name = args.tool
        print(f"🔧 Integrating tool: {tool_name}")
        
        result = await self.agent.integrate_tool(tool_name)
        print(result)
    
    async def show_stats(self):
        """Show discovery statistics"""
        stats = self.agent.get_discovery_stats()
        
        print("📊 Tool Discovery Statistics:")
        print(f"  • Total Discovered: {stats['total_discovered']}")
        print(f"  • Total Evaluated: {stats['total_evaluated']}")
        print(f"  • Total Integrated: {stats['total_integrated']}")
        print(f"  • Success Rate: {stats['success_rate']:.2f}")
        print(f"  • Average Evaluation Time: {stats['average_evaluation_time']:.2f}s")
        print(f"  • Average Integration Time: {stats['average_integration_time']:.2f}s")
        
        # Show capability coverage
        current_capabilities = self.agent._get_current_capabilities()
        missing_capabilities = self.agent._identify_missing_capabilities(current_capabilities)
        
        print(f"\n🎯 Capability Coverage:")
        print(f"  • Current Capabilities: {len(current_capabilities)}")
        print(f"  • Missing Capabilities: {len(missing_capabilities)}")
        
        if current_capabilities:
            print(f"  • Current: {', '.join(current_capabilities[:5])}")
        
        if missing_capabilities:
            print(f"  • Missing: {', '.join(missing_capabilities)}")
    
    async def list_tools(self, args):
        """List tools by type"""
        tool_type = args.type
        limit = args.limit
        
        if tool_type == 'discovered':
            tools = list(self.agent.discovered_tools.values())
            print(f"📦 Discovered Tools ({len(tools)}):")
        elif tool_type == 'evaluated':
            tools = list(self.agent.evaluated_tools.values())
            print(f"📊 Evaluated Tools ({len(tools)}):")
        elif tool_type == 'integrated':
            tools = list(self.agent.integrated_tools.values())
            print(f"🔧 Integrated Tools ({len(tools)}):")
        
        for i, tool in enumerate(tools[:limit], 1):
            if tool_type == 'discovered':
                print(f"  {i}. {tool.name} ({tool.source}) - {tool.description[:80]}...")
            elif tool_type == 'evaluated':
                print(f"  {i}. {tool.tool_name} - Score: {tool.overall_score:.2f}")
            elif tool_type == 'integrated':
                print(f"  {i}. {tool.tool_name} - Method: {tool.integration_method}")
    
    async def show_tool_info(self, args):
        """Show detailed information about a tool"""
        tool_name = args.tool
        info = self.agent.get_tool_info(tool_name)
        
        if not info:
            print(f"❌ Tool '{tool_name}' not found.")
            return
        
        print(f"📋 Tool Information: {tool_name}")
        print("=" * 50)
        
        # Specification
        if info['specification']:
            spec = info['specification']
            print(f"📦 Specification:")
            print(f"  • Name: {spec['name']}")
            print(f"  • Description: {spec['description']}")
            print(f"  • Category: {spec['category']}")
            print(f"  • Priority: {spec['priority']}")
            print(f"  • Source: {spec['source']}")
            print(f"  • Version: {spec['version']}")
            print(f"  • Author: {spec['author']}")
            print(f"  • License: {spec['license']}")
            print(f"  • Status: {spec['status']}")
            print(f"  • Capabilities: {', '.join(spec['capabilities'])}")
            print(f"  • Dependencies: {', '.join(spec['dependencies'])}")
        
        # Evaluation
        if info['evaluation']:
            eval_info = info['evaluation']
            print(f"\n📊 Evaluation:")
            print(f"  • Overall Score: {eval_info['overall_score']:.2f}")
            print(f"  • Functionality: {eval_info['functionality_score']:.2f}")
            print(f"  • Performance: {eval_info['performance_score']:.2f}")
            print(f"  • Security: {eval_info['security_score']:.2f}")
            print(f"  • Compatibility: {eval_info['compatibility_score']:.2f}")
            print(f"  • Documentation: {eval_info['documentation_score']:.2f}")
            print(f"  • Community: {eval_info['community_score']:.2f}")
            print(f"  • Integration Effort: {eval_info['integration_effort']}")
            print(f"  • Estimated Time: {eval_info['estimated_time']}")
        
        # Integration
        if info['integration']:
            int_info = info['integration']
            print(f"\n🔧 Integration:")
            print(f"  • Method: {int_info['integration_method']}")
            print(f"  • Date: {int_info['integration_date']}")
            print(f"  • Endpoints: {', '.join(int_info['endpoints'])}")
            print(f"  • API Keys: {', '.join(int_info['api_keys'])}")
            print(f"  • Environment Variables: {', '.join(int_info['environment_variables'])}")
            print(f"  • Dependencies Installed: {', '.join(int_info['dependencies_installed'])}")
    
    async def search_tools(self, args):
        """Search for tools"""
        query = args.query
        source = args.source
        
        print(f"🔍 Searching for tools: '{query}'")
        
        # Create requirements based on query
        requirements = {
            'capabilities': [query.lower()],
            'priority': 'medium'
        }
        
        discovered_tools = []
        
        if source in ['pypi', 'all']:
            pypi_tools = await self.agent._search_pypi(requirements)
            discovered_tools.extend(pypi_tools)
            print(f"  • PyPI: {len(pypi_tools)} tools found")
        
        if source in ['github', 'all']:
            github_tools = await self.agent._search_github(requirements)
            discovered_tools.extend(github_tools)
            print(f"  • GitHub: {len(github_tools)} tools found")
        
        if source in ['huggingface', 'all']:
            hf_tools = await self.agent._search_huggingface(requirements)
            discovered_tools.extend(hf_tools)
            print(f"  • Hugging Face: {len(hf_tools)} tools found")
        
        if discovered_tools:
            print(f"\n✅ Found {len(discovered_tools)} tools:")
            for tool in discovered_tools[:10]:  # Show top 10
                print(f"  • {tool.name} ({tool.source}) - {tool.description[:80]}...")
        else:
            print("❌ No tools found for the query.")
    
    async def start_continuous_discovery(self, args):
        """Start continuous discovery process"""
        interval = args.interval
        print(f"🔄 Starting continuous discovery (interval: {interval}s)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                print(f"\n⏰ Running discovery cycle at {datetime.now()}")
                
                # Run discovery
                current_capabilities = self.agent._get_current_capabilities()
                missing_capabilities = self.agent._identify_missing_capabilities(current_capabilities)
                
                if missing_capabilities:
                    print(f"🎯 Looking for tools for missing capabilities: {missing_capabilities}")
                    
                    requirements = {
                        'capabilities': missing_capabilities,
                        'priority': 'medium'
                    }
                    
                    discovered_tools = await self.agent._discover_tools(requirements)
                    
                    if discovered_tools:
                        print(f"✅ Discovered {len(discovered_tools)} new tools")
                        
                        # Evaluate top tools
                        evaluations = await self.agent._evaluate_tools(discovered_tools[:3])
                        
                        if evaluations:
                            print(f"📊 Evaluated {len(evaluations)} tools")
                            
                            # Show top recommendations
                            recommendations = self.agent._generate_recommendations(evaluations)
                            if recommendations:
                                print("⭐ Top recommendations:")
                                for rec in recommendations[:2]:
                                    print(f"  • {rec['tool_name']} (Score: {rec['overall_score']:.2f})")
                    else:
                        print("❌ No new tools discovered")
                else:
                    print("✅ All capabilities covered")
                
                print(f"⏳ Waiting {interval} seconds until next cycle...")
                await asyncio.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n🛑 Continuous discovery stopped")

async def main():
    """Main entry point"""
    cli = ToolDiscoveryCLI()
    await cli.run()

if __name__ == "__main__":
    asyncio.run(main()) 