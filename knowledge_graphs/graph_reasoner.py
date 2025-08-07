#!/usr/bin/env python3
"""
Graph Reasoner for Knowledge Graph Analysis

This module implements advanced reasoning capabilities over knowledge graphs
for Pillar 17, including path finding, inference, and pattern recognition.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import networkx as nx
from networkx.algorithms import shortest_path, all_simple_paths, pagerank
from networkx.algorithms.community import greedy_modularity_communities
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class ReasoningResult:
    """Represents a reasoning result with metadata."""
    query: str
    result_type: str
    confidence: float
    evidence: List[str]
    entities_involved: List[str]
    relationships_used: List[str]
    reasoning_path: List[str]


class GraphReasoner:
    """
    Advanced reasoning engine for knowledge graphs.
    
    Capabilities:
    - Path-based reasoning
    - Pattern recognition
    - Entity similarity analysis
    - Community detection
    - Inference generation
    """
    
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def find_connections(self, entity1: str, entity2: str, max_paths: int = 5) -> List[ReasoningResult]:
        """
        Find connections between two entities.
        
        Args:
            entity1: First entity name
            entity2: Second entity name
            max_paths: Maximum number of paths to return
            
        Returns:
            List of reasoning results showing connections
        """
        # Find entities in the graph
        entity1_id = self._find_entity_id(entity1)
        entity2_id = self._find_entity_id(entity2)
        
        if not entity1_id or not entity2_id:
            return []
        
        # Find all paths between entities
        try:
            paths = list(all_simple_paths(self.knowledge_graph.graph, entity1_id, entity2_id))
        except nx.NetworkXNoPath:
            return []
        
        results = []
        for i, path in enumerate(paths[:max_paths]):
            # Convert path to relationship sequence
            relationships = []
            for j in range(len(path) - 1):
                edge_data = self.knowledge_graph.graph.get_edge_data(path[j], path[j + 1])
                for edge_key in edge_data:
                    rel = self.knowledge_graph.relationships.get(edge_key)
                    if rel:
                        relationships.append(rel)
                        break
            
            # Create reasoning result
            evidence = [f"Path {i+1}: {' -> '.join([self.knowledge_graph.entities[node_id].name for node_id in path])}"]
            entities_involved = [self.knowledge_graph.entities[node_id].name for node_id in path]
            relationships_used = [rel.relationship_type for rel in relationships]
            
            result = ReasoningResult(
                query=f"Connection between {entity1} and {entity2}",
                result_type="path_connection",
                confidence=0.8 - (i * 0.1),  # First path has higher confidence
                evidence=evidence,
                entities_involved=entities_involved,
                relationships_used=relationships_used,
                reasoning_path=[f"Found {len(path)}-hop path via {', '.join(relationships_used)}"]
            )
            results.append(result)
        
        return results
    
    def find_similar_entities(self, entity: str, similarity_threshold: float = 0.7) -> List[ReasoningResult]:
        """
        Find entities similar to the given entity.
        
        Args:
            entity: Entity name to find similarities for
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of reasoning results with similar entities
        """
        entity_id = self._find_entity_id(entity)
        if not entity_id:
            return []
        
        entity_obj = self.knowledge_graph.entities[entity_id]
        similar_entities = []
        
        # Find entities with similar attributes
        for other_id, other_entity in self.knowledge_graph.entities.items():
            if other_id == entity_id:
                continue
            
            # Calculate similarity based on entity type and attributes
            type_similarity = 1.0 if entity_obj.entity_type == other_entity.entity_type else 0.0
            
            # Attribute similarity
            common_attrs = set(entity_obj.attributes.keys()) & set(other_entity.attributes.keys())
            attr_similarity = len(common_attrs) / max(len(entity_obj.attributes), len(other_entity.attributes), 1)
            
            # Embedding similarity if available
            embedding_similarity = 0.0
            if entity_obj.embedding and other_entity.embedding:
                embedding_similarity = np.dot(entity_obj.embedding, other_entity.embedding) / (
                    np.linalg.norm(entity_obj.embedding) * np.linalg.norm(other_entity.embedding)
                )
            
            # Combined similarity
            total_similarity = (type_similarity + attr_similarity + embedding_similarity) / 3
            
            if total_similarity >= similarity_threshold:
                similar_entities.append((other_entity, total_similarity))
        
        # Sort by similarity
        similar_entities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for similar_entity, similarity in similar_entities[:10]:
            result = ReasoningResult(
                query=f"Entities similar to {entity}",
                result_type="entity_similarity",
                confidence=similarity,
                evidence=[f"Similarity score: {similarity:.3f}"],
                entities_involved=[entity, similar_entity.name],
                relationships_used=[],
                reasoning_path=[f"Found similar entity based on type, attributes, and embeddings"]
            )
            results.append(result)
        
        return results
    
    def detect_communities(self) -> List[ReasoningResult]:
        """
        Detect communities in the knowledge graph.
        
        Returns:
            List of reasoning results showing communities
        """
        if len(self.knowledge_graph.graph.nodes()) < 2:
            return []
        
        try:
            # Convert to undirected graph for community detection
            undirected_graph = self.knowledge_graph.graph.to_undirected()
            communities = list(greedy_modularity_communities(undirected_graph))
            
            results = []
            for i, community in enumerate(communities):
                community_entities = [self.knowledge_graph.entities[node_id].name for node_id in community]
                
                result = ReasoningResult(
                    query="Community detection",
                    result_type="community",
                    confidence=0.8,
                    evidence=[f"Community {i+1} with {len(community)} entities"],
                    entities_involved=community_entities,
                    relationships_used=[],
                    reasoning_path=[f"Detected community using modularity optimization"]
                )
                results.append(result)
            
            return results
        except Exception as e:
            print(f"Error in community detection: {e}")
            return []
    
    def find_central_entities(self, top_k: int = 10) -> List[ReasoningResult]:
        """
        Find the most central entities in the knowledge graph.
        
        Args:
            top_k: Number of top central entities to return
            
        Returns:
            List of reasoning results with central entities
        """
        if len(self.knowledge_graph.graph.nodes()) < 2:
            return []
        
        try:
            # Calculate PageRank centrality
            pagerank_scores = pagerank(self.knowledge_graph.graph)
            
            # Sort by centrality
            sorted_entities = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
            
            results = []
            for entity_id, centrality in sorted_entities[:top_k]:
                entity_name = self.knowledge_graph.entities[entity_id].name
                
                result = ReasoningResult(
                    query="Central entities",
                    result_type="centrality",
                    confidence=centrality,
                    evidence=[f"PageRank centrality: {centrality:.4f}"],
                    entities_involved=[entity_name],
                    relationships_used=[],
                    reasoning_path=[f"Calculated centrality using PageRank algorithm"]
                )
                results.append(result)
            
            return results
        except Exception as e:
            print(f"Error in centrality calculation: {e}")
            return []
    
    def infer_relationships(self, entity1: str, entity2: str) -> List[ReasoningResult]:
        """
        Infer potential relationships between entities.
        
        Args:
            entity1: First entity name
            entity2: Second entity name
            
        Returns:
            List of reasoning results with inferred relationships
        """
        entity1_id = self._find_entity_id(entity1)
        entity2_id = self._find_entity_id(entity2)
        
        if not entity1_id or not entity2_id:
            return []
        
        entity1_obj = self.knowledge_graph.entities[entity1_id]
        entity2_obj = self.knowledge_graph.entities[entity2_id]
        
        # Check if relationship already exists
        existing_relationships = []
        for rel in self.knowledge_graph.relationships.values():
            if (rel.source_entity_id == entity1_id and rel.target_entity_id == entity2_id) or \
               (rel.source_entity_id == entity2_id and rel.target_entity_id == entity1_id):
                existing_relationships.append(rel)
        
        if existing_relationships:
            results = []
            for rel in existing_relationships:
                result = ReasoningResult(
                    query=f"Existing relationships between {entity1} and {entity2}",
                    result_type="existing_relationship",
                    confidence=rel.confidence,
                    evidence=[f"Found {rel.relationship_type} relationship"],
                    entities_involved=[entity1, entity2],
                    relationships_used=[rel.relationship_type],
                    reasoning_path=[f"Direct relationship exists in knowledge graph"]
                )
                results.append(result)
            return results
        
        # Infer potential relationships based on entity types and patterns
        inferred_relationships = self._infer_relationship_patterns(entity1_obj, entity2_obj)
        
        results = []
        for rel_type, confidence, evidence in inferred_relationships:
            result = ReasoningResult(
                query=f"Inferred relationships between {entity1} and {entity2}",
                result_type="inferred_relationship",
                confidence=confidence,
                evidence=[evidence],
                entities_involved=[entity1, entity2],
                relationships_used=[rel_type],
                reasoning_path=[f"Inferred based on entity types and patterns"]
            )
            results.append(result)
        
        return results
    
    def _find_entity_id(self, entity_name: str) -> Optional[str]:
        """Find entity ID by name."""
        entity_name_lower = entity_name.lower()
        for entity_id, entity in self.knowledge_graph.entities.items():
            if entity_name_lower in entity.name.lower() or entity.name.lower() in entity_name_lower:
                return entity_id
        return None
    
    def _infer_relationship_patterns(self, entity1, entity2) -> List[Tuple[str, float, str]]:
        """Infer potential relationships based on entity types and patterns."""
        relationships = []
        
        # Type-based inference
        if entity1.entity_type == 'person' and entity2.entity_type == 'organization':
            relationships.append(('employment', 0.6, 'Person-organization pattern suggests employment'))
        elif entity1.entity_type == 'organization' and entity2.entity_type == 'person':
            relationships.append(('employment', 0.6, 'Organization-person pattern suggests employment'))
        
        if entity1.entity_type == 'person' and entity2.entity_type == 'location':
            relationships.append(('location', 0.5, 'Person-location pattern suggests residence/work'))
        elif entity1.entity_type == 'location' and entity2.entity_type == 'person':
            relationships.append(('location', 0.5, 'Location-person pattern suggests residence/work'))
        
        if entity1.entity_type == 'organization' and entity2.entity_type == 'location':
            relationships.append(('location', 0.7, 'Organization-location pattern suggests headquarters'))
        elif entity1.entity_type == 'location' and entity2.entity_type == 'organization':
            relationships.append(('location', 0.7, 'Location-organization pattern suggests headquarters'))
        
        # Attribute-based inference
        common_attrs = set(entity1.attributes.keys()) & set(entity2.attributes.keys())
        if common_attrs:
            relationships.append(('related_to', 0.4, f'Shared attributes: {", ".join(common_attrs)}'))
        
        return relationships
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get statistics about reasoning capabilities."""
        stats = {
            'total_entities': len(self.knowledge_graph.entities),
            'total_relationships': len(self.knowledge_graph.relationships),
            'graph_density': nx.density(self.knowledge_graph.graph),
            'connected_components': nx.number_connected_components(self.knowledge_graph.graph.to_undirected()),
            'average_clustering': nx.average_clustering(self.knowledge_graph.graph)
        }
        
        return stats 