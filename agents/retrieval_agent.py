#!/usr/bin/env python3
"""
Retrieval Agent - Pillar 5
Handles semantic search, document retrieval, and knowledge base queries
"""

import os
import sys
import time
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.base import Agent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """A retrieved document with metadata."""
    id: str
    content: str
    title: str
    source: str
    url: Optional[str]
    relevance_score: float
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class RetrievalResult:
    """Result from document retrieval."""
    query: str
    documents: List[Document]
    total_results: int
    search_time: float
    confidence: float
    search_type: str
    timestamp: datetime

class RetrievalAgent(Agent):
    """Retrieval Agent for semantic search and document retrieval."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__("retrieval")
        self.model_name = model_name
        
        # Initialize document store (in-memory for now)
        self.document_store = {}
        self.embeddings = {}
        
        # Search configuration
        self.max_results = 10
        self.min_relevance_score = 0.3
        self.search_types = ["semantic", "keyword", "hybrid", "exact"]
        
        # Load models
        self.load_model()
        
        # Initialize with sample documents
        self._initialize_sample_documents()
        
    def load_model(self):
        """Load retrieval models for semantic search."""
        logger.info("Loading retrieval models...")
        
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Load sentence transformer for embeddings
            self.embedding_model = SentenceTransformer(self.model_name)
            
            # Import similarity functions
            self.cosine_similarity = cosine_similarity
            self.np = np
            
            logger.info("✅ Retrieval models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading retrieval models: {e}")
            self.embedding_model = None
            self.cosine_similarity = None
            self.np = None
    
    def generate(self, prompt: str, **kwargs) -> RetrievalResult:
        """Generate retrieval results for the given query."""
        try:
            # Ensure models are loaded
            self._ensure_model()
            
            # Determine search type
            search_type = kwargs.get("search_type", "semantic")
            
            # Perform retrieval
            documents = self._retrieve_documents(prompt, search_type)
            
            # Calculate confidence based on relevance scores
            confidence = self._calculate_confidence(documents)
            
            # Create retrieval result
            result = RetrievalResult(
                query=prompt,
                documents=documents,
                total_results=len(documents),
                search_time=0.0,  # Would be measured in real implementation
                confidence=confidence,
                search_type=search_type,
                timestamp=datetime.now()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            return self._create_fallback_result(prompt)
    
    def _retrieve_documents(self, query: str, search_type: str) -> List[Document]:
        """Retrieve relevant documents based on search type."""
        if search_type == "semantic":
            return self._semantic_search(query)
        elif search_type == "keyword":
            return self._keyword_search(query)
        elif search_type == "hybrid":
            return self._hybrid_search(query)
        elif search_type == "exact":
            return self._exact_search(query)
        else:
            return self._semantic_search(query)  # Default to semantic search
    
    def _semantic_search(self, query: str) -> List[Document]:
        """Perform semantic search using embeddings."""
        if not self.embedding_model:
            return self._fallback_search(query)
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Calculate similarities with all documents
            similarities = []
            for doc_id, doc_data in self.document_store.items():
                if doc_id in self.embeddings:
                    doc_embedding = self.embeddings[doc_id]
                    similarity = self.cosine_similarity(query_embedding, [doc_embedding])[0][0]
                    similarities.append((doc_id, similarity))
            
            # Sort by similarity and filter by minimum score
            similarities.sort(key=lambda x: x[1], reverse=True)
            relevant_docs = [
                (doc_id, score) for doc_id, score in similarities 
                if score >= self.min_relevance_score
            ][:self.max_results]
            
            # Convert to Document objects
            documents = []
            for doc_id, score in relevant_docs:
                doc_data = self.document_store[doc_id]
                documents.append(Document(
                    id=doc_id,
                    content=doc_data["content"],
                    title=doc_data["title"],
                    source=doc_data["source"],
                    url=doc_data.get("url"),
                    relevance_score=score,
                    metadata=doc_data.get("metadata", {}),
                    timestamp=datetime.now()
                ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return self._fallback_search(query)
    
    def _keyword_search(self, query: str) -> List[Document]:
        """Perform keyword-based search."""
        query_terms = query.lower().split()
        relevant_docs = []
        
        for doc_id, doc_data in self.document_store.items():
            content_lower = doc_data["content"].lower()
            title_lower = doc_data["title"].lower()
            
            # Calculate keyword matches
            content_matches = sum(1 for term in query_terms if term in content_lower)
            title_matches = sum(1 for term in query_terms if term in title_lower)
            
            # Weight title matches more heavily
            total_matches = content_matches + (title_matches * 2)
            
            if total_matches > 0:
                relevance_score = min(total_matches / len(query_terms), 1.0)
                relevant_docs.append((doc_id, relevance_score))
        
        # Sort by relevance and limit results
        relevant_docs.sort(key=lambda x: x[1], reverse=True)
        relevant_docs = relevant_docs[:self.max_results]
        
        # Convert to Document objects
        documents = []
        for doc_id, score in relevant_docs:
            doc_data = self.document_store[doc_id]
            documents.append(Document(
                id=doc_id,
                content=doc_data["content"],
                title=doc_data["title"],
                source=doc_data["source"],
                url=doc_data.get("url"),
                relevance_score=score,
                metadata=doc_data.get("metadata", {}),
                timestamp=datetime.now()
            ))
        
        return documents
    
    def _hybrid_search(self, query: str) -> List[Document]:
        """Perform hybrid search combining semantic and keyword approaches."""
        semantic_docs = self._semantic_search(query)
        keyword_docs = self._keyword_search(query)
        
        # Combine and deduplicate results
        all_docs = {}
        
        # Add semantic results
        for doc in semantic_docs:
            all_docs[doc.id] = doc
        
        # Add keyword results, updating scores if document already exists
        for doc in keyword_docs:
            if doc.id in all_docs:
                # Average the scores
                existing_doc = all_docs[doc.id]
                existing_doc.relevance_score = (existing_doc.relevance_score + doc.relevance_score) / 2
            else:
                all_docs[doc.id] = doc
        
        # Sort by relevance score and return top results
        sorted_docs = sorted(all_docs.values(), key=lambda x: x.relevance_score, reverse=True)
        return sorted_docs[:self.max_results]
    
    def _exact_search(self, query: str) -> List[Document]:
        """Perform exact phrase search."""
        query_lower = query.lower()
        relevant_docs = []
        
        for doc_id, doc_data in self.document_store.items():
            content_lower = doc_data["content"].lower()
            title_lower = doc_data["title"].lower()
            
            # Check for exact phrase matches
            content_match = query_lower in content_lower
            title_match = query_lower in title_lower
            
            if content_match or title_match:
                # Calculate relevance based on match position and frequency
                relevance_score = 0.5  # Base score for exact match
                
                if title_match:
                    relevance_score += 0.3  # Bonus for title match
                
                if content_match:
                    # Count occurrences
                    occurrences = content_lower.count(query_lower)
                    relevance_score += min(occurrences * 0.1, 0.2)  # Cap at 0.2
                
                relevant_docs.append((doc_id, relevance_score))
        
        # Sort by relevance and limit results
        relevant_docs.sort(key=lambda x: x[1], reverse=True)
        relevant_docs = relevant_docs[:self.max_results]
        
        # Convert to Document objects
        documents = []
        for doc_id, score in relevant_docs:
            doc_data = self.document_store[doc_id]
            documents.append(Document(
                id=doc_id,
                content=doc_data["content"],
                title=doc_data["title"],
                source=doc_data["source"],
                url=doc_data.get("url"),
                relevance_score=score,
                metadata=doc_data.get("metadata", {}),
                timestamp=datetime.now()
            ))
        
        return documents
    
    def _calculate_confidence(self, documents: List[Document]) -> float:
        """Calculate confidence score based on retrieval results."""
        if not documents:
            return 0.0
        
        # Average relevance score of top results
        avg_relevance = sum(doc.relevance_score for doc in documents) / len(documents)
        
        # Boost confidence if we have multiple high-quality results
        if len(documents) >= 3:
            avg_relevance += 0.1
        
        # Boost confidence if top result is very relevant
        if documents and documents[0].relevance_score > 0.8:
            avg_relevance += 0.1
        
        return min(avg_relevance, 1.0)
    
    def _fallback_search(self, query: str) -> List[Document]:
        """Fallback search when models are not available."""
        # Simple keyword-based fallback
        return self._keyword_search(query)
    
    def _create_fallback_result(self, query: str) -> RetrievalResult:
        """Create a fallback retrieval result when search fails."""
        documents = self._fallback_search(query)
        
        return RetrievalResult(
            query=query,
            documents=documents,
            total_results=len(documents),
            search_time=0.0,
            confidence=0.5,
            search_type="fallback",
            timestamp=datetime.now()
        )
    
    def _initialize_sample_documents(self):
        """Initialize the document store with sample documents."""
        sample_documents = [
            {
                "id": "doc_001",
                "title": "Introduction to Artificial Intelligence",
                "content": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.",
                "source": "AI Textbook",
                "url": "https://example.com/ai-intro",
                "metadata": {"category": "education", "difficulty": "beginner"}
            },
            {
                "id": "doc_002", 
                "title": "Machine Learning Fundamentals",
                "content": "Machine Learning is a subset of AI that focuses on algorithms and statistical models that enable computers to improve their performance on a specific task through experience. It includes supervised learning, unsupervised learning, and reinforcement learning.",
                "source": "ML Guide",
                "url": "https://example.com/ml-fundamentals",
                "metadata": {"category": "education", "difficulty": "intermediate"}
            },
            {
                "id": "doc_003",
                "title": "Natural Language Processing",
                "content": "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It involves tasks such as text classification, sentiment analysis, machine translation, and question answering.",
                "source": "NLP Handbook",
                "url": "https://example.com/nlp-guide",
                "metadata": {"category": "technology", "difficulty": "advanced"}
            },
            {
                "id": "doc_004",
                "title": "Deep Learning and Neural Networks",
                "content": "Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns. It has been particularly successful in image recognition, speech recognition, and natural language processing.",
                "source": "Deep Learning Book",
                "url": "https://example.com/deep-learning",
                "metadata": {"category": "technology", "difficulty": "advanced"}
            },
            {
                "id": "doc_005",
                "title": "Computer Vision Applications",
                "content": "Computer Vision is a field of AI that enables computers to interpret and understand visual information from the world. Applications include image recognition, object detection, facial recognition, and autonomous vehicles.",
                "source": "Computer Vision Guide",
                "url": "https://example.com/computer-vision",
                "metadata": {"category": "technology", "difficulty": "intermediate"}
            }
        ]
        
        # Add documents to store
        for doc in sample_documents:
            self.document_store[doc["id"]] = doc
        
        # Generate embeddings for semantic search
        if self.embedding_model:
            try:
                for doc_id, doc_data in self.document_store.items():
                    # Combine title and content for embedding
                    text = f"{doc_data['title']} {doc_data['content']}"
                    embedding = self.embedding_model.encode([text])[0]
                    self.embeddings[doc_id] = embedding
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
    
    def add_document(self, doc_id: str, title: str, content: str, source: str, 
                    url: Optional[str] = None, metadata: Optional[Dict] = None):
        """Add a new document to the retrieval system."""
        doc_data = {
            "id": doc_id,
            "title": title,
            "content": content,
            "source": source,
            "url": url,
            "metadata": metadata or {}
        }
        
        self.document_store[doc_id] = doc_data
        
        # Generate embedding for semantic search
        if self.embedding_model:
            try:
                text = f"{title} {content}"
                embedding = self.embedding_model.encode([text])[0]
                self.embeddings[doc_id] = embedding
            except Exception as e:
                logger.error(f"Error generating embedding for document {doc_id}: {e}")
    
    def remove_document(self, doc_id: str):
        """Remove a document from the retrieval system."""
        if doc_id in self.document_store:
            del self.document_store[doc_id]
        
        if doc_id in self.embeddings:
            del self.embeddings[doc_id]
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        return {
            "total_documents": len(self.document_store),
            "total_embeddings": len(self.embeddings),
            "search_types": self.search_types,
            "max_results": self.max_results,
            "min_relevance_score": self.min_relevance_score,
            "model_loaded": self.embedding_model is not None
        }

