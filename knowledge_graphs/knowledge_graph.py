#!/usr/bin/env python3
"""
Knowledge Graph System for Quark AI Assistant
=================================================

Implements knowledge graph construction, entity extraction,
relationship modeling, and graph-based reasoning.

Part of Pillar 17: Long-Term Memory & Knowledge Graphs
"""

import os
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import networkx as nx
from collections import defaultdict

from agents.base import Agent


class EntityType(Enum):
    """Types of entities in the knowledge graph."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    OBJECT = "object"
    ACTION = "action"
    ATTRIBUTE = "attribute"


class RelationshipType(Enum):
    """Types of relationships between entities."""
    IS_A = "is_a"                    # Taxonomic relationship
    PART_OF = "part_of"              # Meronymic relationship
    LOCATED_IN = "located_in"         # Spatial relationship
    WORKS_FOR = "works_for"           # Employment relationship
    KNOWS = "knows"                   # Social relationship
    CREATED = "created"               # Creation relationship
    USES = "uses"                     # Usage relationship
    SIMILAR_TO = "similar_to"         # Similarity relationship
    OPPOSITE_OF = "opposite_of"       # Opposition relationship
    CAUSES = "causes"                 # Causal relationship
    BELONGS_TO = "belongs_to"         # Membership relationship
    INTERACTS_WITH = "interacts_with" # Interaction relationship


@dataclass
class Entity:
    """Entity in the knowledge graph."""
    id: str
    name: str
    entity_type: EntityType
    attributes: Dict[str, Any]
    confidence: float
    source: str
    created_at: float
    last_updated: float
    aliases: List[str]  # Alternative names
    description: str


@dataclass
class Relationship:
    """Relationship between entities in the knowledge graph."""
    id: str
    source_entity: str
    target_entity: str
    relationship_type: RelationshipType
    attributes: Dict[str, Any]
    confidence: float
    source: str
    created_at: float
    last_updated: float
    bidirectional: bool  # Whether relationship works both ways


@dataclass
class GraphQuery:
    """Query for knowledge graph operations."""
    query_type: str  # "entity", "relationship", "path", "subgraph"
    parameters: Dict[str, Any]
    max_results: int
    min_confidence: float


class KnowledgeGraph:
    """Advanced knowledge graph system with reasoning capabilities."""
    
    def __init__(self, graph_dir: str = None):
        self.graph_dir = graph_dir or os.path.join(os.path.dirname(__file__), '..', 'knowledge_graphs', 'data')
        os.makedirs(self.graph_dir, exist_ok=True)
        
        # Graph structure
        self.graph = nx.DiGraph()
        self.entities = {}
        self.relationships = {}
        
        # Initialize the graph
        self.initialize()
    
    def initialize(self):
        """Initialize the knowledge graph"""
        # Indexing
        self.entity_name_index = defaultdict(list)  # Name to entity IDs
        self.entity_type_index = defaultdict(list)  # Type to entity IDs
        self.relationship_type_index = defaultdict(list)  # Type to relationship IDs
        
        # Graph settings
        self.max_entities = 10000
        self.max_relationships = 50000
        self.confidence_threshold = 0.5
        
        # Load existing graph data
        self._load_graph()
    
    def _load_graph(self):
        """Load existing knowledge graph from storage."""
        try:
            # Load entities
            entities_file = os.path.join(self.graph_dir, 'entities.json')
            if os.path.exists(entities_file):
                with open(entities_file, 'r') as f:
                    entities_data = json.load(f)
                    
                for entity_id, entity_data in entities_data.items():
                    entity = Entity(
                        id=entity_id,
                        name=entity_data['name'],
                        entity_type=EntityType(entity_data['entity_type']),
                        attributes=entity_data['attributes'],
                        confidence=entity_data['confidence'],
                        source=entity_data['source'],
                        created_at=entity_data['created_at'],
                        last_updated=entity_data['last_updated'],
                        aliases=entity_data['aliases'],
                        description=entity_data['description']
                    )
                    self.entities[entity_id] = entity
                    self._add_entity_to_graph(entity)
            
            # Load relationships
            relationships_file = os.path.join(self.graph_dir, 'relationships.json')
            if os.path.exists(relationships_file):
                with open(relationships_file, 'r') as f:
                    relationships_data = json.load(f)
                    
                for rel_id, rel_data in relationships_data.items():
                    relationship = Relationship(
                        id=rel_id,
                        source_entity=rel_data['source_entity'],
                        target_entity=rel_data['target_entity'],
                        relationship_type=RelationshipType(rel_data['relationship_type']),
                        attributes=rel_data['attributes'],
                        confidence=rel_data['confidence'],
                        source=rel_data['source'],
                        created_at=rel_data['created_at'],
                        last_updated=rel_data['last_updated'],
                        bidirectional=rel_data['bidirectional']
                    )
                    self.relationships[rel_id] = relationship
                    self._add_relationship_to_graph(relationship)
                    
        except Exception as e:
            print(f"Error loading knowledge graph: {e}")
    
    def _add_entity_to_graph(self, entity: Entity):
        """Add entity to the graph and indexes."""
        # Add to graph
        self.graph.add_node(entity.id, **asdict(entity))
        
        # Update indexes
        self.entity_name_index[entity.name.lower()].append(entity.id)
        self.entity_type_index[entity.entity_type.value].append(entity.id)
        
        # Add aliases to name index
        for alias in entity.aliases:
            self.entity_name_index[alias.lower()].append(entity.id)
    
    def _add_relationship_to_graph(self, relationship: Relationship):
        """Add relationship to the graph and indexes."""
        # Add to graph
        self.graph.add_edge(
            relationship.source_entity,
            relationship.target_entity,
            key=relationship.id,
            **asdict(relationship)
        )
        
        # Update relationship type index
        self.relationship_type_index[relationship.relationship_type.value].append(relationship.id)
        
        # Add reverse edge if bidirectional
        if relationship.bidirectional:
            self.graph.add_edge(
                relationship.target_entity,
                relationship.source_entity,
                key=f"{relationship.id}_reverse",
                **{**asdict(relationship), 'id': f"{relationship.id}_reverse"}
            )
    
    def add_entity(self, name: str, entity_type: EntityType, 
                  attributes: Dict[str, Any] = None, confidence: float = 1.0,
                  source: str = "user_input", aliases: List[str] = None,
                  description: str = "") -> str:
        """Add a new entity to the knowledge graph."""
        entity_id = self._generate_entity_id(name, entity_type)
        
        entity = Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            attributes=attributes or {},
            confidence=confidence,
            source=source,
            created_at=time.time(),
            last_updated=time.time(),
            aliases=aliases or [],
            description=description
        )
        
        self.entities[entity_id] = entity
        self._add_entity_to_graph(entity)
        
        return entity_id
    
    def _generate_entity_id(self, name: str, entity_type: EntityType) -> str:
        """Generate unique entity ID."""
        timestamp = str(time.time())
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"{entity_type.value}_{name_hash}_{timestamp}"
    
    def add_relationship(self, source_entity: str, target_entity: str,
                       relationship_type: RelationshipType,
                       attributes: Dict[str, Any] = None, confidence: float = 1.0,
                       source: str = "user_input", bidirectional: bool = False) -> str:
        """Add a new relationship to the knowledge graph."""
        # Check if entities exist
        if source_entity not in self.entities:
            raise ValueError(f"Source entity {source_entity} not found")
        if target_entity not in self.entities:
            raise ValueError(f"Target entity {target_entity} not found")
        
        relationship_id = self._generate_relationship_id(source_entity, target_entity, relationship_type)
        
        relationship = Relationship(
            id=relationship_id,
            source_entity=source_entity,
            target_entity=target_entity,
            relationship_type=relationship_type,
            attributes=attributes or {},
            confidence=confidence,
            source=source,
            created_at=time.time(),
            last_updated=time.time(),
            bidirectional=bidirectional
        )
        
        self.relationships[relationship_id] = relationship
        self._add_relationship_to_graph(relationship)
        
        return relationship_id
    
    def _generate_relationship_id(self, source: str, target: str, rel_type: RelationshipType) -> str:
        """Generate unique relationship ID."""
        timestamp = str(time.time())
        rel_hash = hashlib.md5(f"{source}_{target}_{rel_type.value}".encode()).hexdigest()[:8]
        return f"rel_{rel_hash}_{timestamp}"
    
    def find_entity(self, name: str, entity_type: EntityType = None) -> List[Entity]:
        """Find entities by name and optionally type."""
        matching_entities = []
        
        # Search by name
        name_lower = name.lower()
        candidate_ids = self.entity_name_index.get(name_lower, [])
        
        for entity_id in candidate_ids:
            entity = self.entities.get(entity_id)
            if entity and (entity_type is None or entity.entity_type == entity_type):
                matching_entities.append(entity)
        
        return matching_entities
    
    def find_relationships(self, source_entity: str = None, target_entity: str = None,
                         relationship_type: RelationshipType = None) -> List[Relationship]:
        """Find relationships based on criteria."""
        matching_relationships = []
        
        for relationship in self.relationships.values():
            if source_entity and relationship.source_entity != source_entity:
                continue
            if target_entity and relationship.target_entity != target_entity:
                continue
            if relationship_type and relationship.relationship_type != relationship_type:
                continue
            
            matching_relationships.append(relationship)
        
        return matching_relationships
    
    def get_all_entities(self) -> List[Entity]:
        """Get all entities in the knowledge graph."""
        return list(self.entities.values())
    
    def get_entity_neighbors(self, entity_id: str, relationship_types: List[RelationshipType] = None,
                           max_depth: int = 1) -> Dict[str, Any]:
        """Get neighboring entities and relationships."""
        if entity_id not in self.entities:
            return {"error": f"Entity {entity_id} not found"}
        
        neighbors = {
            'entity': self.entities[entity_id],
            'neighbors': [],
            'relationships': []
        }
        
        # Get direct neighbors
        if entity_id in self.graph:
            for neighbor_id in self.graph.neighbors(entity_id):
                neighbor_entity = self.entities.get(neighbor_id)
                if neighbor_entity:
                    neighbors['neighbors'].append(neighbor_entity)
                    
                    # Get relationships between entities
                    edges = self.graph.get_edge_data(entity_id, neighbor_id)
                    for edge_key, edge_data in edges.items():
                        if relationship_types is None or edge_data['relationship_type'] in relationship_types:
                            neighbors['relationships'].append(edge_data)
        
        return neighbors
    
    def find_path(self, source_entity: str, target_entity: str, 
                 max_path_length: int = 5) -> List[Dict[str, Any]]:
        """Find path between two entities."""
        if source_entity not in self.entities or target_entity not in self.entities:
            return []
        
        try:
            # Find shortest path
            path = nx.shortest_path(self.graph, source_entity, target_entity)
            
            if len(path) > max_path_length:
                return []  # Path too long
            
            # Convert path to detailed format
            path_details = []
            for i in range(len(path) - 1):
                current_entity = path[i]
                next_entity = path[i + 1]
                
                # Get relationship between these entities
                edges = self.graph.get_edge_data(current_entity, next_entity)
                if edges:
                    for edge_key, edge_data in edges.items():
                        path_details.append({
                            'source_entity': self.entities[current_entity],
                            'target_entity': self.entities[next_entity],
                            'relationship': edge_data
                        })
                        break  # Take first relationship
            
            return path_details
            
        except nx.NetworkXNoPath:
            return []
    
    def get_subgraph(self, entity_ids: List[str], include_neighbors: bool = True) -> Dict[str, Any]:
        """Get subgraph containing specified entities."""
        if not entity_ids:
            return {"error": "No entity IDs provided"}
        
        # Create subgraph
        subgraph_nodes = set(entity_ids)
        
        if include_neighbors:
            # Add neighbors
            for entity_id in entity_ids:
                if entity_id in self.graph:
                    neighbors = list(self.graph.neighbors(entity_id))
                    subgraph_nodes.update(neighbors)
        
        # Create subgraph
        subgraph = self.graph.subgraph(subgraph_nodes)
        
        # Convert to serializable format
        subgraph_data = {
            'nodes': [],
            'edges': [],
            'statistics': {
                'node_count': subgraph.number_of_nodes(),
                'edge_count': subgraph.number_of_edges()
            }
        }
        
        # Add nodes
        for node_id in subgraph.nodes():
            entity = self.entities.get(node_id)
            if entity:
                subgraph_data['nodes'].append(asdict(entity))
        
        # Add edges
        for source, target, key, data in subgraph.edges(data=True, keys=True):
            subgraph_data['edges'].append({
                'source': source,
                'target': target,
                'key': key,
                'data': data
            })
        
        return subgraph_data
    
    def reason_about_entity(self, entity_id: str, reasoning_type: str = "general") -> Dict[str, Any]:
        """Perform reasoning about an entity."""
        if entity_id not in self.entities:
            return {"error": f"Entity {entity_id} not found"}
        
        entity = self.entities[entity_id]
        reasoning_results = {
            'entity': asdict(entity),
            'reasoning_type': reasoning_type,
            'insights': [],
            'related_entities': [],
            'patterns': []
        }
        
        if reasoning_type == "general":
            # General reasoning about entity
            neighbors = self.get_entity_neighbors(entity_id)
            
            # Analyze relationships
            relationship_types = defaultdict(int)
            for rel in neighbors['relationships']:
                relationship_types[rel['relationship_type']] += 1
            
            # Generate insights
            for rel_type, count in relationship_types.items():
                reasoning_results['insights'].append({
                    'type': 'relationship_pattern',
                    'pattern': f"Entity has {count} {rel_type} relationships",
                    'confidence': min(count / 10.0, 1.0)
                })
            
            # Find related entities
            for neighbor in neighbors['neighbors']:
                reasoning_results['related_entities'].append({
                    'entity': asdict(neighbor),
                    'relationship_count': len(self.find_relationships(
                        source_entity=entity_id, target_entity=neighbor.id
                    ))
                })
        
        elif reasoning_type == "taxonomic":
            # Taxonomic reasoning
            is_a_relationships = self.find_relationships(
                source_entity=entity_id, relationship_type=RelationshipType.IS_A
            )
            
            for rel in is_a_relationships:
                parent_entity = self.entities.get(rel.target_entity)
                if parent_entity:
                    reasoning_results['insights'].append({
                        'type': 'taxonomic',
                        'pattern': f"{entity.name} is a {parent_entity.name}",
                        'confidence': rel.confidence
                    })
        
        elif reasoning_type == "causal":
            # Causal reasoning
            causal_relationships = self.find_relationships(
                source_entity=entity_id, relationship_type=RelationshipType.CAUSES
            )
            
            for rel in causal_relationships:
                effect_entity = self.entities.get(rel.target_entity)
                if effect_entity:
                    reasoning_results['insights'].append({
                        'type': 'causal',
                        'pattern': f"{entity.name} causes {effect_entity.name}",
                        'confidence': rel.confidence
                    })
        
        return reasoning_results
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'graph_nodes': self.graph.number_of_nodes(),
            'graph_edges': self.graph.number_of_edges(),
            'entity_types': {
                entity_type.value: len([e for e in self.entities.values() if e.entity_type == entity_type])
                for entity_type in EntityType
            },
            'relationship_types': {
                rel_type.value: len([r for r in self.relationships.values() if r.relationship_type == rel_type])
                for rel_type in RelationshipType
            },
            'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            'connected_components': nx.number_strongly_connected_components(self.graph),
            'density': nx.density(self.graph)
        }
    
    def export_graph_data(self, filename: str = None) -> str:
        """Export knowledge graph data to JSON."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"knowledge_graph_export_{timestamp}.json"
        
        filepath = os.path.join(self.graph_dir, filename)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'graph_statistics': self.get_graph_statistics(),
            'entities': {k: asdict(v) for k, v in self.entities.items()},
            'relationships': {k: asdict(v) for k, v in self.relationships.items()}
        }
        
        # Convert enum values to strings
        for entity_id, entity_data in export_data['entities'].items():
            entity_data['entity_type'] = entity_data['entity_type'].value
        
        for rel_id, rel_data in export_data['relationships'].items():
            rel_data['relationship_type'] = rel_data['relationship_type'].value
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filepath
    
    def _save_graph(self):
        """Save graph data to disk."""
        try:
            # Save entities
            entities_data = {}
            for entity_id, entity in self.entities.items():
                entities_data[entity_id] = asdict(entity)
                entities_data[entity_id]['entity_type'] = entity.entity_type.value
            
            entities_file = os.path.join(self.graph_dir, 'entities.json')
            with open(entities_file, 'w') as f:
                json.dump(entities_data, f, indent=2)
            
            # Save relationships
            relationships_data = {}
            for rel_id, relationship in self.relationships.items():
                relationships_data[rel_id] = asdict(relationship)
                relationships_data[rel_id]['relationship_type'] = relationship.relationship_type.value
            
            relationships_file = os.path.join(self.graph_dir, 'relationships.json')
            with open(relationships_file, 'w') as f:
                json.dump(relationships_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving knowledge graph: {e}") 