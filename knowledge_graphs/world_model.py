#!/usr/bin/env python3
"""
World Model for Knowledge Graph Integration

This module implements world modeling capabilities for Pillar 17.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from sentence_transformers import SentenceTransformer


@dataclass
class WorldModelFact:
    """Represents a fact in the world model."""
    id: str
    content: str
    fact_type: str
    confidence: float
    source_documents: List[str]
    entities: List[str]
    created_at: float


class WorldModel:
    """
    World model for knowledge integration and reasoning.
    """
    
    def __init__(self, knowledge_graph=None):
        self.knowledge_graph = knowledge_graph
        self.facts: Dict[str, WorldModelFact] = {}
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_document(self, document_id: str, content: str, metadata: Dict[str, Any] = None):
        """Add a document to the world model."""
        self.documents[document_id] = {
            'content': content,
            'metadata': metadata or {},
            'added_at': time.time()
        }
        
        # Extract facts from document
        facts = self._extract_facts_from_document(document_id, content)
        for fact in facts:
            self.add_fact(fact)
    
    def add_fact(self, fact: WorldModelFact):
        """Add a fact to the world model."""
        self.facts[fact.id] = fact
    
    def query_cross_document(self, query: str) -> List[Dict[str, Any]]:
        """Perform cross-document reasoning."""
        # Find relevant facts
        relevant_facts = self._find_relevant_facts(query)
        
        # Generate answers
        answers = []
        if relevant_facts:
            answer = {
                'query': query,
                'answer': ' '.join([fact.content for fact in relevant_facts]),
                'confidence': 0.7,
                'sources': list(set([fact.source_documents[0] for fact in relevant_facts]))
            }
            answers.append(answer)
        
        return answers
    
    def _extract_facts_from_document(self, document_id: str, content: str) -> List[WorldModelFact]:
        """Extract facts from document content."""
        facts = []
        sentences = content.split('.')
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            fact = WorldModelFact(
                id=f"{document_id}_fact_{i}",
                content=sentence,
                fact_type='general',
                confidence=0.8,
                source_documents=[document_id],
                entities=[],
                created_at=time.time()
            )
            facts.append(fact)
        
        return facts
    
    def _find_relevant_facts(self, query: str) -> List[WorldModelFact]:
        """Find facts relevant to a query."""
        query_embedding = self.embedding_model.encode(query)
        relevant_facts = []
        
        for fact in self.facts.values():
            # Simple keyword matching
            query_words = query.lower().split()
            fact_words = fact.content.lower().split()
            
            if any(word in fact_words for word in query_words):
                relevant_facts.append(fact)
        
        return relevant_facts[:5]  # Top 5 relevant facts 