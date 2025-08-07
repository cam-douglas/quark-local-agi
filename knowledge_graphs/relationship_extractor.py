#!/usr/bin/env python3
"""
Relationship Extractor for Knowledge Graph Construction

This module implements automatic relationship discovery between entities
for building knowledge graphs in Pillar 17.
"""

import re
import spacy
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ExtractedRelationship:
    """Represents an extracted relationship with metadata."""
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float
    attributes: Dict[str, Any]
    context: str


class RelationshipExtractor:
    """
    Multi-method relationship extractor for knowledge graph construction.
    """
    
    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy
        
        # Initialize spaCy
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy model not available.")
                self.nlp = None
                self.use_spacy = False
        
        # Common relationship patterns
        self.relationship_patterns = {
            'ownership': [
                r'(\w+)\s+(?:owns?|has|possesses?)\s+(\w+)',
                r'(\w+)\'s\s+(\w+)'
            ],
            'location': [
                r'(\w+)\s+(?:is\s+)?(?:in|at|located\s+in)\s+(\w+)',
                r'(\w+)\s+(?:lives\s+in|works\s+in)\s+(\w+)'
            ],
            'employment': [
                r'(\w+)\s+(?:works\s+for|employed\s+by)\s+(\w+)',
                r'(\w+)\s+(?:is\s+)?(?:CEO|manager|employee)\s+(?:of|at)\s+(\w+)'
            ],
            'family': [
                r'(\w+)\s+(?:is\s+)?(?:father|mother|son|daughter)\s+(?:of)\s+(\w+)',
                r'(\w+)\s+(?:married\s+to|spouse\s+of)\s+(\w+)'
            ]
        }
    
    def extract_relationships(self, text: str, entities: List[str]) -> List[ExtractedRelationship]:
        """Extract relationships between entities in text."""
        relationships = []
        
        # Extract using patterns
        pattern_relationships = self._extract_with_patterns(text, entities)
        relationships.extend(pattern_relationships)
        
        # Extract using spaCy if available
        if self.use_spacy and self.nlp:
            spacy_relationships = self._extract_with_spacy(text, entities)
            relationships.extend(spacy_relationships)
        
        return self._merge_relationships(relationships)
    
    def _extract_with_patterns(self, text: str, entities: List[str]) -> List[ExtractedRelationship]:
        """Extract relationships using regex patterns."""
        relationships = []
        
        for rel_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        source_entity = groups[0]
                        target_entity = groups[1]
                        
                        # Check if entities are in our list
                        source_found = any(source_entity.lower() in entity.lower() for entity in entities)
                        target_found = any(target_entity.lower() in entity.lower() for entity in entities)
                        
                        if source_found and target_found:
                            relationship = ExtractedRelationship(
                                source_entity=source_entity,
                                target_entity=target_entity,
                                relationship_type=rel_type,
                                confidence=0.7,
                                attributes={'pattern': pattern},
                                context=match.group()
                            )
                            relationships.append(relationship)
        
        return relationships
    
    def _extract_with_spacy(self, text: str, entities: List[str]) -> List[ExtractedRelationship]:
        """Extract relationships using spaCy dependency parsing."""
        relationships = []
        doc = self.nlp(text)
        
        # Simple co-occurrence analysis
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            entities_in_sentence = [entity for entity in entities if entity.lower() in sentence_lower]
            
            if len(entities_in_sentence) >= 2:
                for i in range(len(entities_in_sentence)):
                    for j in range(i + 1, len(entities_in_sentence)):
                        source = entities_in_sentence[i]
                        target = entities_in_sentence[j]
                        
                        relationship = ExtractedRelationship(
                            source_entity=source,
                            target_entity=target,
                            relationship_type='related_to',
                            confidence=0.5,
                            attributes={'method': 'co_occurrence'},
                            context=sentence.strip()
                        )
                        relationships.append(relationship)
        
        return relationships
    
    def _merge_relationships(self, relationships: List[ExtractedRelationship]) -> List[ExtractedRelationship]:
        """Merge similar relationships and remove duplicates."""
        if not relationships:
            return []
        
        # Group by entity pairs and relationship type
        relationship_groups = {}
        
        for rel in relationships:
            key = (rel.source_entity.lower(), rel.target_entity.lower(), rel.relationship_type)
            
            if key not in relationship_groups:
                relationship_groups[key] = []
            relationship_groups[key].append(rel)
        
        # Merge groups
        merged_relationships = []
        
        for key, group in relationship_groups.items():
            if len(group) == 1:
                merged_relationships.append(group[0])
            else:
                # Take the highest confidence relationship
                best_relationship = max(group, key=lambda x: x.confidence)
                merged_relationships.append(best_relationship)
        
        return merged_relationships 