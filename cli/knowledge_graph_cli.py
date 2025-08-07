#!/usr/bin/env python3
"""
Knowledge Graph CLI for Quark AI Assistant

This CLI provides an interface for testing Pillar 17: Long-Term Memory & Knowledge Graphs
"""

import argparse
import json
import sys
from typing import Dict, Any
from agents.knowledge_graph_agent import KnowledgeGraphAgent


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph CLI for Pillar 17")
    parser.add_argument("operation", choices=[
        "process_document", "query_knowledge", "extract_entities", 
        "find_relationships", "reason", "store_memory", "retrieve_memory", 
        "get_statistics"
    ], help="Operation to perform")
    
    parser.add_argument("--input", "-i", type=str, help="Input text or document")
    parser.add_argument("--file", "-f", type=str, help="Input file path")
    parser.add_argument("--document-id", type=str, help="Document ID")
    parser.add_argument("--memory-type", type=str, default="general", help="Memory type")
    parser.add_argument("--importance", type=float, default=0.5, help="Importance score")
    parser.add_argument("--reasoning-type", type=str, default="connections", 
                       choices=["connections", "similarity", "communities", "centrality", "inference"],
                       help="Reasoning type")
    parser.add_argument("--entities", nargs="+", help="Entities for reasoning")
    parser.add_argument("--limit", type=int, default=10, help="Result limit")
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = KnowledgeGraphAgent()
    
    # Get input
    input_text = args.input
    if args.file:
        try:
            with open(args.file, 'r') as f:
                input_text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    
    if not input_text:
        print("Error: No input provided. Use --input or --file")
        sys.exit(1)
    
    # Prepare kwargs
    kwargs = {
        'operation': args.operation,
        'document_id': args.document_id,
        'memory_type': args.memory_type,
        'importance': args.importance,
        'reasoning_type': args.reasoning_type,
        'limit': args.limit
    }
    
    if args.entities:
        kwargs['entities'] = args.entities
    
    # Perform operation
    try:
        result = agent.generate(input_text, **kwargs)
        
        # Format output
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"Error performing operation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 