#!/usr/bin/env python3
"""
Test Pillar 17: Long-Term Memory & Knowledge Graphs
==================================================

Tests the implementation of persistent memory systems, knowledge graph
construction, memory consolidation, and intelligent retrieval mechanisms.

Part of Phase 5: AGI Capabilities
"""

import os
import sys
import time
import tempfile
import shutil
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.long_term_memory import LongTermMemory, MemoryType, MemoryPriority
from knowledge_graphs.knowledge_graph import KnowledgeGraph, EntityType, RelationshipType
from agents.memory_agent import MemoryAgent


def test_pillar_17():
    """Test Pillar 17: Long-Term Memory & Knowledge Graphs."""
    print("🧠 Testing Pillar 17: Long-Term Memory & Knowledge Graphs")
    print("=" * 60)
    
    # Create temporary directories for testing
    temp_dir = tempfile.mkdtemp()
    memory_dir = os.path.join(temp_dir, "memory")
    graph_dir = os.path.join(temp_dir, "knowledge_graphs")
    
    try:
        # Test 1: Long-Term Memory System
        print("\n📚 Testing Long-Term Memory System")
        print("-" * 40)
        
        # Initialize memory system
        memory_system = LongTermMemory(memory_dir)
        
        # Test memory storage
        print("✅ Testing Memory Storage")
        
        # Store different types of memories
        episodic_id = memory_system.store_memory(
            content="I had a conversation with John about AI safety yesterday",
            memory_type=MemoryType.EPISODIC,
            priority=MemoryPriority.HIGH,
            tags=["conversation", "ai_safety", "john"]
        )
        print(f"   Episodic memory stored: {episodic_id}")
        
        semantic_id = memory_system.store_memory(
            content="AI safety is the practice of ensuring AI systems behave safely and beneficially",
            memory_type=MemoryType.SEMANTIC,
            priority=MemoryPriority.CRITICAL,
            tags=["ai_safety", "definition", "concept"]
        )
        print(f"   Semantic memory stored: {semantic_id}")
        
        procedural_id = memory_system.store_memory(
            content="To implement safety checks, first identify potential risks, then design mitigation strategies",
            memory_type=MemoryType.PROCEDURAL,
            priority=MemoryPriority.HIGH,
            tags=["procedure", "safety_checks", "implementation"]
        )
        print(f"   Procedural memory stored: {procedural_id}")
        
        working_id = memory_system.store_memory(
            content="Current task: analyze user query for safety concerns",
            memory_type=MemoryType.WORKING,
            priority=MemoryPriority.MEDIUM,
            tags=["current_task", "analysis"]
        )
        print(f"   Working memory stored: {working_id}")
        
        # Test memory retrieval
        print("\n✅ Testing Memory Retrieval")
        
        # Retrieve memories by query
        ai_safety_memories = memory_system.retrieve_memories(
            query="AI safety",
            max_results=5
        )
        print(f"   AI safety memories found: {len(ai_safety_memories)}")
        
        for memory in ai_safety_memories:
            print(f"     - {memory.content[:50]}... ({memory.memory_type.value})")
        
        # Test memory consolidation
        print("\n✅ Testing Memory Consolidation")
        memory_system.consolidate_memories()
        print("   Memory consolidation completed")
        
        # Test memory statistics
        print("\n✅ Testing Memory Statistics")
        stats = memory_system.get_memory_stats()
        print(f"   Total memories: {stats['total_memories']}")
        print(f"   Episodic memories: {stats['episodic_memories']}")
        print(f"   Semantic memories: {stats['semantic_memories']}")
        print(f"   Procedural memories: {stats['procedural_memories']}")
        print(f"   Working memories: {stats['working_memories']}")
        
        # Test 2: Knowledge Graph System
        print("\n🔗 Testing Knowledge Graph System")
        print("-" * 40)
        
        # Initialize knowledge graph
        knowledge_graph = KnowledgeGraph(graph_dir)
        
        # Test entity creation
        print("✅ Testing Entity Creation")
        
        # Create entities
        person_id = knowledge_graph.add_entity(
            name="John Smith",
            entity_type=EntityType.PERSON,
            attributes={"age": 35, "profession": "AI Researcher"},
            description="AI safety researcher working on alignment"
        )
        print(f"   Person entity created: {person_id}")
        
        org_id = knowledge_graph.add_entity(
            name="AI Safety Institute",
            entity_type=EntityType.ORGANIZATION,
            attributes={"founded": 2020, "focus": "AI Safety"},
            description="Research organization focused on AI safety"
        )
        print(f"   Organization entity created: {org_id}")
        
        concept_id = knowledge_graph.add_entity(
            name="AI Alignment",
            entity_type=EntityType.CONCEPT,
            attributes={"category": "AI Safety", "complexity": "high"},
            description="Ensuring AI systems pursue human-intended goals"
        )
        print(f"   Concept entity created: {concept_id}")
        
        location_id = knowledge_graph.add_entity(
            name="San Francisco",
            entity_type=EntityType.LOCATION,
            attributes={"country": "USA", "state": "California"},
            description="Major tech hub in California"
        )
        print(f"   Location entity created: {location_id}")
        
        # Test relationship creation
        print("\n✅ Testing Relationship Creation")
        
        # Create relationships
        works_rel = knowledge_graph.add_relationship(
            source_entity=person_id,
            target_entity=org_id,
            relationship_type=RelationshipType.WORKS_FOR,
            attributes={"role": "Senior Researcher", "start_date": "2021"},
            bidirectional=True
        )
        print(f"   Works-for relationship created: {works_rel}")
        
        knows_rel = knowledge_graph.add_relationship(
            source_entity=person_id,
            target_entity=concept_id,
            relationship_type=RelationshipType.KNOWS,
            attributes={"expertise_level": "expert"},
            bidirectional=False
        )
        print(f"   Knows relationship created: {knows_rel}")
        
        located_rel = knowledge_graph.add_relationship(
            source_entity=org_id,
            target_entity=location_id,
            relationship_type=RelationshipType.LOCATED_IN,
            attributes={"address": "123 Tech Street"},
            bidirectional=False
        )
        print(f"   Located-in relationship created: {located_rel}")
        
        # Test entity search
        print("\n✅ Testing Entity Search")
        
        john_entities = knowledge_graph.find_entity("John Smith")
        print(f"   John Smith entities found: {len(john_entities)}")
        
        person_entities = knowledge_graph.find_entity("", EntityType.PERSON)
        print(f"   Person entities found: {len(person_entities)}")
        
        # Test path finding
        print("\n✅ Testing Path Finding")
        
        path = knowledge_graph.find_path(person_id, location_id)
        print(f"   Path from person to location: {len(path)} steps")
        for step in path:
            source_name = step['source_entity'].name if hasattr(step['source_entity'], 'name') else str(step['source_entity'])
            target_name = step['target_entity'].name if hasattr(step['target_entity'], 'name') else str(step['target_entity'])
            rel_type = step['relationship']['relationship_type'] if isinstance(step['relationship'], dict) else str(step['relationship'])
            print(f"     {source_name} -> {target_name} ({rel_type})")
        
        # Test reasoning
        print("\n✅ Testing Graph Reasoning")
        
        reasoning = knowledge_graph.reason_about_entity(person_id, "general")
        print(f"   Reasoning insights: {len(reasoning['insights'])}")
        for insight in reasoning['insights']:
            print(f"     - {insight['pattern']}")
        
        # Test graph statistics
        print("\n✅ Testing Graph Statistics")
        graph_stats = knowledge_graph.get_graph_statistics()
        print(f"   Total entities: {graph_stats['total_entities']}")
        print(f"   Total relationships: {graph_stats['total_relationships']}")
        print(f"   Graph density: {graph_stats['density']:.3f}")
        
        # Test 3: Memory Agent Integration
        print("\n🤖 Testing Memory Agent Integration")
        print("-" * 40)
        
        # Initialize memory agent
        memory_agent = MemoryAgent(memory_dir=memory_dir)
        
        # Test agent operations
        print("✅ Testing Memory Agent Operations")
        
        # Store memory via agent
        agent_memory_id = memory_agent.generate(
            "The user asked about implementing safety checks in AI systems",
            operation="store_memory",
            memory_type="episodic",
            priority="high",
            tags=["user_query", "safety_checks", "ai_systems"]
        )
        print(f"   Agent stored memory: {agent_memory_id['memory_id']}")
        
        # Retrieve memories via agent
        retrieval_result = memory_agent.generate(
            "safety checks",
            operation="retrieve_memories",
            memory_types=["episodic", "semantic"],
            max_results=5
        )
        print(f"   Agent retrieved {retrieval_result['results_count']} memories")
        
        # Get memory statistics via agent
        stats_result = memory_agent.generate("", operation="get_memory_stats")
        print(f"   Agent memory stats: {stats_result['basic_stats']['total_memories']} total memories")
        
        # Test 4: Advanced Features
        print("\n🚀 Testing Advanced Features")
        print("-" * 40)
        
        # Test memory export
        print("✅ Testing Memory Export")
        export_file = memory_system.export_memory_data()
        print(f"   Memory data exported to: {export_file}")
        
        # Test graph export
        print("✅ Testing Graph Export")
        graph_export_file = knowledge_graph.export_graph_data()
        print(f"   Graph data exported to: {graph_export_file}")
        
        # Test subgraph extraction
        print("✅ Testing Subgraph Extraction")
        subgraph = knowledge_graph.get_subgraph([person_id, org_id], include_neighbors=True)
        print(f"   Subgraph extracted: {subgraph['statistics']['node_count']} nodes, {subgraph['statistics']['edge_count']} edges")
        
        # Test memory recommendations
        print("✅ Testing Memory Recommendations")
        recommendations = memory_agent.get_memory_recommendations()
        print(f"   Memory recommendations: {recommendations['recommendations_count']}")
        
        print("\n🎉 Pillar 17 Test Completed Successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Persistent memory systems with multiple memory types")
        print("✅ Knowledge graph construction and entity management")
        print("✅ Relationship modeling and graph traversal")
        print("✅ Memory consolidation and optimization")
        print("✅ Intelligent memory retrieval and search")
        print("✅ Graph-based reasoning and path finding")
        print("✅ Memory agent integration and management")
        print("✅ Data export and persistence")
        print("✅ Advanced graph operations and subgraph extraction")
        print("✅ Memory recommendations and optimization")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_memory_knowledge_integration():
    """Test integration between memory and knowledge graph systems."""
    print("\n🔗 Testing Memory-Knowledge Integration")
    print("=" * 50)
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    memory_dir = os.path.join(temp_dir, "memory")
    graph_dir = os.path.join(temp_dir, "knowledge_graphs")
    
    try:
        # Initialize both systems
        memory_system = LongTermMemory(memory_dir)
        knowledge_graph = KnowledgeGraph(graph_dir)
        memory_agent = MemoryAgent(memory_dir=memory_dir)
        
        # Test integrated workflow
        print("✅ Testing Integrated Workflow")
        
        # 1. Store memory about a person
        person_memory_id = memory_system.store_memory(
            content="Alice is an AI researcher who works on safety alignment",
            memory_type=MemoryType.EPISODIC,
            priority=MemoryPriority.HIGH,
            tags=["person", "alice", "ai_researcher", "safety_alignment"]
        )
        
        # 2. Create corresponding knowledge graph entities
        alice_entity_id = knowledge_graph.add_entity(
            name="Alice",
            entity_type=EntityType.PERSON,
            attributes={"profession": "AI Researcher", "focus": "Safety Alignment"},
            description="AI researcher working on safety alignment"
        )
        
        safety_concept_id = knowledge_graph.add_entity(
            name="Safety Alignment",
            entity_type=EntityType.CONCEPT,
            attributes={"category": "AI Safety", "importance": "critical"},
            description="Ensuring AI systems align with human values"
        )
        
        # 3. Create relationships
        works_on_rel = knowledge_graph.add_relationship(
            source_entity=alice_entity_id,
            target_entity=safety_concept_id,
            relationship_type=RelationshipType.WORKS_FOR,
            attributes={"role": "Researcher", "expertise": "high"}
        )
        
        # 4. Test cross-system queries
        print("✅ Testing Cross-System Queries")
        
        # Find memories about Alice
        alice_memories = memory_system.retrieve_memories("Alice")
        print(f"   Memories about Alice: {len(alice_memories)}")
        
        # Find entities related to Alice
        alice_entities = knowledge_graph.find_entity("Alice")
        print(f"   Entities for Alice: {len(alice_entities)}")
        
        # Find relationships for Alice
        alice_relationships = knowledge_graph.find_relationships(source_entity=alice_entity_id)
        print(f"   Relationships for Alice: {len(alice_relationships)}")
        
        # 5. Test integrated reasoning
        print("✅ Testing Integrated Reasoning")
        
        # Combine memory and knowledge graph data
        integrated_insights = {
            'memory_insights': len(alice_memories),
            'entity_insights': len(alice_entities),
            'relationship_insights': len(alice_relationships),
            'combined_knowledge': f"Alice is an AI researcher with {len(alice_relationships)} known relationships"
        }
        
        print(f"   Integrated insights: {integrated_insights['combined_knowledge']}")
        
        print("✅ Memory-Knowledge Integration Test Completed Successfully!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_pillar_17()
    test_memory_knowledge_integration() 