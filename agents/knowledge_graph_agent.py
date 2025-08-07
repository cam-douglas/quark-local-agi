#!/usr/bin/env python3
"""
Knowledge Graph Agent for Quark AI Assistant

This agent implements Pillar 17: Long-Term Memory & Knowledge Graphs
by integrating knowledge graph construction, entity extraction, relationship
discovery, and advanced reasoning capabilities.
"""

from typing import Dict, List, Any, Optional
from .base import Agent
from knowledge_graphs.knowledge_graph import KnowledgeGraph
from knowledge_graphs.entity_extractor import EntityExtractor
from knowledge_graphs.relationship_extractor import RelationshipExtractor
from knowledge_graphs.graph_reasoner import GraphReasoner
from knowledge_graphs.world_model import WorldModel
# from memory.long_term_memory import LongTermMemory


class KnowledgeGraphAgent(Agent):
    """
    Knowledge Graph Agent implementing Pillar 17 capabilities.
    
    Features:
    - Knowledge graph construction
    - Entity and relationship extraction
    - Cross-document reasoning
    - Long-term memory integration
    - Advanced graph reasoning
    """
    
    def __init__(self, model_name: str = "knowledge_graph_agent"):
        super().__init__(model_name)
        # No specific model needed for knowledge graph operations
        
        # Initialize components
        self.knowledge_graph = KnowledgeGraph()
        self.entity_extractor = EntityExtractor()
        self.relationship_extractor = RelationshipExtractor()
        self.graph_reasoner = GraphReasoner(self.knowledge_graph)
        self.world_model = WorldModel(self.knowledge_graph)
        # self.long_term_memory = LongTermMemory()
        
        # Agent state
        self.processed_documents = set()
        self.extraction_stats = {
            'entities_extracted': 0,
            'relationships_extracted': 0,
            'documents_processed': 0
        }
    
    def load_model(self):
        """Initialize the knowledge graph agent."""
        # All components are initialized in __init__
        return True
    
    def _ensure_model(self):
        """Ensure the agent is properly initialized."""
        return True
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Main interface for knowledge graph operations.
        
        Args:
            prompt: Input prompt or document
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with operation results
        """
        operation = kwargs.get('operation', 'process_document')
        
        if operation == 'process_document':
            return self._process_document(prompt, kwargs)
        elif operation == 'query_knowledge':
            return self._query_knowledge(prompt, kwargs)
        elif operation == 'extract_entities':
            return self._extract_entities(prompt, kwargs)
        elif operation == 'find_relationships':
            return self._find_relationships(prompt, kwargs)
        elif operation == 'reason':
            return self._perform_reasoning(prompt, kwargs)
        elif operation == 'store_memory':
            return self._store_memory(prompt, kwargs)
        elif operation == 'retrieve_memory':
            return self._retrieve_memory(prompt, kwargs)
        elif operation == 'get_statistics':
            return self._get_statistics()
        else:
            return self._process_document(prompt, kwargs)
    
    def _process_document(self, document: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document and extract knowledge."""
        document_id = kwargs.get('document_id', f"doc_{len(self.processed_documents)}")
        
        if document_id in self.processed_documents:
            return {
                'operation': 'process_document',
                'status': 'already_processed',
                'document_id': document_id
            }
        
        # Add to world model
        self.world_model.add_document(document_id, document)
        
        # Extract entities
        entities = self.entity_extractor.extract_entities(document)
        
        # Add entities to knowledge graph
        entity_ids = []
        for entity in entities:
            entity_id = self.knowledge_graph.add_entity(
                name=entity.text,
                entity_type=entity.entity_type,
                attributes=entity.attributes,
                confidence=entity.confidence,
                source_document=document_id
            )
            entity_ids.append(entity_id)
        
        # Extract relationships
        entity_names = [entity.text for entity in entities]
        relationships = self.relationship_extractor.extract_relationships(document, entity_names)
        
        # Add relationships to knowledge graph
        relationship_ids = []
        for rel in relationships:
            # Find entity IDs for the relationship
            source_id = self._find_entity_id(rel.source_entity)
            target_id = self._find_entity_id(rel.target_entity)
            
            if source_id and target_id:
                rel_id = self.knowledge_graph.add_relationship(
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relationship_type=rel.relationship_type,
                    attributes=rel.attributes,
                    confidence=rel.confidence,
                    source_document=document_id
                )
                relationship_ids.append(rel_id)
        
        # Store in long-term memory
        memory_id = self.long_term_memory.store_memory(
            content=document,
            memory_type='document',
            importance=kwargs.get('importance', 0.5),
            associations=entity_names,
            metadata={'document_id': document_id}
        )
        
        # Update statistics
        self.extraction_stats['entities_extracted'] += len(entities)
        self.extraction_stats['relationships_extracted'] += len(relationships)
        self.extraction_stats['documents_processed'] += 1
        self.processed_documents.add(document_id)
        
        return {
            'operation': 'process_document',
            'status': 'success',
            'document_id': document_id,
            'entities_extracted': len(entities),
            'relationships_extracted': len(relationships),
            'memory_id': memory_id,
            'statistics': self.extraction_stats
        }
    
    def _query_knowledge(self, query: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Query the knowledge graph and world model."""
        # Query world model
        world_results = self.world_model.query_cross_document(query)
        
        # Query long-term memory
        memories = self.long_term_memory.retrieve_memories(query)
        
        # Query knowledge graph
        entities = self.knowledge_graph.find_entities(query)
        
        return {
            'operation': 'query_knowledge',
            'query': query,
            'world_model_results': world_results,
            'memories': [{'content': m.content, 'importance': m.importance} for m in memories],
            'entities': [{'name': e.name, 'type': e.entity_type} for e in entities],
            'total_results': len(world_results) + len(memories) + len(entities)
        }
    
    def _extract_entities(self, text: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities from text."""
        entities = self.entity_extractor.extract_entities(text)
        
        return {
            'operation': 'extract_entities',
            'entities': [
                {
                    'text': entity.text,
                    'type': entity.entity_type,
                    'confidence': entity.confidence,
                    'attributes': entity.attributes
                }
                for entity in entities
            ],
            'total_entities': len(entities)
        }
    
    def _find_relationships(self, text: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Find relationships in text."""
        entities = kwargs.get('entities', [])
        if not entities:
            # Extract entities first
            extracted_entities = self.entity_extractor.extract_entities(text)
            entities = [entity.text for entity in extracted_entities]
        
        relationships = self.relationship_extractor.extract_relationships(text, entities)
        
        return {
            'operation': 'find_relationships',
            'relationships': [
                {
                    'source': rel.source_entity,
                    'target': rel.target_entity,
                    'type': rel.relationship_type,
                    'confidence': rel.confidence,
                    'context': rel.context
                }
                for rel in relationships
            ],
            'total_relationships': len(relationships)
        }
    
    def _perform_reasoning(self, query: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Perform advanced reasoning over the knowledge graph."""
        reasoning_type = kwargs.get('reasoning_type', 'connections')
        
        if reasoning_type == 'connections':
            entities = kwargs.get('entities', [])
            if len(entities) >= 2:
                results = self.graph_reasoner.find_connections(entities[0], entities[1])
            else:
                results = []
        elif reasoning_type == 'similarity':
            entity = kwargs.get('entity', '')
            if entity:
                results = self.graph_reasoner.find_similar_entities(entity)
            else:
                results = []
        elif reasoning_type == 'communities':
            results = self.graph_reasoner.detect_communities()
        elif reasoning_type == 'centrality':
            results = self.graph_reasoner.find_central_entities()
        elif reasoning_type == 'inference':
            entities = kwargs.get('entities', [])
            if len(entities) >= 2:
                results = self.graph_reasoner.infer_relationships(entities[0], entities[1])
            else:
                results = []
        else:
            results = []
        
        return {
            'operation': 'reason',
            'reasoning_type': reasoning_type,
            'results': [
                {
                    'query': result.query,
                    'result_type': result.result_type,
                    'confidence': result.confidence,
                    'evidence': result.evidence,
                    'entities_involved': result.entities_involved,
                    'relationships_used': result.relationships_used
                }
                for result in results
            ],
            'total_results': len(results)
        }
    
    def _store_memory(self, content: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Store content in long-term memory."""
        memory_id = self.long_term_memory.store_memory(
            content=content,
            memory_type=kwargs.get('memory_type', 'general'),
            importance=kwargs.get('importance', 0.5),
            associations=kwargs.get('associations', []),
            metadata=kwargs.get('metadata', {})
        )
        
        return {
            'operation': 'store_memory',
            'memory_id': memory_id,
            'status': 'success'
        }
    
    def _retrieve_memory(self, query: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve memories from long-term memory."""
        memories = self.long_term_memory.retrieve_memories(
            query=query,
            memory_type=kwargs.get('memory_type'),
            limit=kwargs.get('limit', 10)
        )
        
        return {
            'operation': 'retrieve_memory',
            'memories': [
                {
                    'id': memory.id,
                    'content': memory.content,
                    'type': memory.memory_type,
                    'importance': memory.importance,
                    'associations': memory.associations
                }
                for memory in memories
            ],
            'total_memories': len(memories)
        }
    
    def _get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        kg_stats = self.knowledge_graph.get_graph_statistics()
        memory_stats = self.long_term_memory.get_memory_statistics()
        reasoning_stats = self.graph_reasoner.get_reasoning_statistics()
        
        return {
            'operation': 'get_statistics',
            'knowledge_graph': kg_stats,
            'long_term_memory': memory_stats,
            'reasoning': reasoning_stats,
            'extraction': self.extraction_stats,
            'documents_processed': len(self.processed_documents)
        }
    
    def _find_entity_id(self, entity_name: str) -> Optional[str]:
        """Find entity ID by name."""
        entity_name_lower = entity_name.lower()
        for entity_id, entity in self.knowledge_graph.entities.items():
            if entity_name_lower in entity.name.lower() or entity.name.lower() in entity_name_lower:
                return entity_id
        return None 