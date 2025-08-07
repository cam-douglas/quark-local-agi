"""
Knowledge Graphs Module for Quark AI Assistant

This module implements Pillar 17: Long-Term Memory & Knowledge Graphs
Focus: World-modeling, knowledge-graph ingestion, cross-document reasoning
"""

from .knowledge_graph import KnowledgeGraph
from .entity_extractor import EntityExtractor
from .relationship_extractor import RelationshipExtractor
from .graph_reasoner import GraphReasoner
from .world_model import WorldModel

__all__ = [
    'KnowledgeGraph',
    'EntityExtractor', 
    'RelationshipExtractor',
    'GraphReasoner',
    'WorldModel'
] 