"""
Retrieval-Augmented Generation (RAG) Agent
Pillar 22: Fast retriever + small generator architecture, parameter-efficient knowledge management
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import hashlib
import pickle

from .base import Agent as BaseAgent
from knowledge_graphs.knowledge_graph import KnowledgeGraph
from memory.long_term_memory import LongTermMemory


class RetrievalMethod(Enum):
    """Retrieval methods for RAG"""
    SEMANTIC_SEARCH = "semantic_search"
    KEYWORD_SEARCH = "keyword_search"
    HYBRID_SEARCH = "hybrid_search"
    GRAPH_TRAVERSAL = "graph_traversal"
    MEMORY_RETRIEVAL = "memory_retrieval"


class GenerationStrategy(Enum):
    """Generation strategies for RAG"""
    CONDITIONAL_GENERATION = "conditional_generation"
    TEMPLATE_BASED = "template_based"
    ADAPTIVE_GENERATION = "adaptive_generation"
    CONTEXT_AWARE = "context_aware"


@dataclass
class RetrievedContext:
    """Retrieved context for generation"""
    content: str
    source: str
    relevance_score: float
    retrieval_method: RetrievalMethod
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class GenerationResult:
    """Result of RAG generation"""
    generated_text: str
    retrieved_contexts: List[RetrievedContext]
    confidence_score: float
    generation_strategy: GenerationStrategy
    processing_time: float
    metadata: Dict[str, Any]


class RAGAgent(BaseAgent):
    """
    Retrieval-Augmented Generation Agent
    
    Implements fast retriever + small generator architecture for
    parameter-efficient knowledge management and dynamic updates.
    """
    
    # Class attributes for dataclasses
    RetrievedContext = RetrievedContext
    GenerationResult = GenerationResult
    RetrievalMethod = RetrievalMethod
    GenerationStrategy = GenerationStrategy
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("rag")
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Knowledge stores
        self.knowledge_graph = KnowledgeGraph()
        self.long_term_memory = LongTermMemory()
        
        # Retrieval systems
        self.retrieval_methods = {
            RetrievalMethod.SEMANTIC_SEARCH: self._semantic_search,
            RetrievalMethod.KEYWORD_SEARCH: self._keyword_search,
            RetrievalMethod.HYBRID_SEARCH: self._hybrid_search,
            RetrievalMethod.GRAPH_TRAVERSAL: self._graph_traversal,
            RetrievalMethod.MEMORY_RETRIEVAL: self._memory_retrieval
        }
        
        # Generation systems
        self.generation_strategies = {
            GenerationStrategy.CONDITIONAL_GENERATION: self._conditional_generation,
            GenerationStrategy.TEMPLATE_BASED: self._template_based_generation,
            GenerationStrategy.ADAPTIVE_GENERATION: self._adaptive_generation,
            GenerationStrategy.CONTEXT_AWARE: self._context_aware_generation
        }
        
        # Performance tracking
        self.retrieval_stats = defaultdict(int)
        self.generation_stats = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Caching system
        self.response_cache = {}
        self.context_cache = {}
        self.cache_ttl = timedelta(hours=1)
        
        # Configuration
        self.max_context_length = 2000
        self.max_retrieved_contexts = 5
        self.min_relevance_threshold = 0.3
        self.generation_timeout = 30.0  # seconds
        
        # Knowledge update tracking
        self.last_knowledge_update = datetime.now()
        self.knowledge_update_interval = timedelta(hours=6)
        
        # Initialize RAG system
        self._initialize_rag_system()
    
    def load_model(self):
        """Load RAG models and components"""
        try:
            # Initialize RAG systems
            self._initialize_rag_system()
            return True
        except Exception as e:
            print(f"Error loading RAG models: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate RAG response or perform RAG operation.
        
        Args:
            prompt: RAG query or operation
            **kwargs: Additional parameters
            
        Returns:
            RAG operation result
        """
        try:
            # Parse the prompt to determine operation
            if "query" in prompt.lower():
                return asyncio.run(self._perform_rag_operation(
                    prompt, "hybrid_search", "adaptive_generation", 3
                ))
            elif "add" in prompt.lower() and "knowledge" in prompt.lower():
                return asyncio.run(self.add_knowledge({"entities": [], "memories": []}))
            elif "stats" in prompt.lower():
                return asyncio.run(self.get_system_stats())
            elif "clear" in prompt.lower() and "cache" in prompt.lower():
                return asyncio.run(self.clear_cache())
            else:
                return {"error": f"Unknown RAG operation: {prompt}"}
                
        except Exception as e:
            return {"error": f"RAG operation failed: {str(e)}"}
    
    @property
    def name(self) -> str:
        """Get the agent name"""
        return self.model_name
    
    def _initialize_rag_system(self):
        """Initialize the RAG system"""
        self.logger.info("Initializing RAG system...")
        
        # Initialize knowledge stores
        self._initialize_knowledge_stores()
        
        # Set up retrieval methods
        self._setup_retrieval_methods()
        
        # Set up generation strategies
        self._setup_generation_strategies()
        
        # Start knowledge update task if event loop is running
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self._periodic_knowledge_update())
        except RuntimeError:
            # No running event loop, skip async task creation
            pass
        
        self.logger.info("RAG system initialized successfully")
    
    def _initialize_knowledge_stores(self):
        """Initialize knowledge stores"""
        self.logger.info("Initializing knowledge stores...")
        
        # Initialize knowledge graph
        self.knowledge_graph.initialize()
        
        # Initialize long-term memory
        self.long_term_memory.initialize()
        
        # Load initial knowledge
        self._load_initial_knowledge()
    
    def _load_initial_knowledge(self):
        """Load initial knowledge into the system"""
        self.logger.info("Loading initial knowledge...")
        
        # Load from knowledge graph
        initial_entities = self.knowledge_graph.get_all_entities()
        self.logger.info(f"Loaded {len(initial_entities)} entities from knowledge graph")
        
        # Load from long-term memory
        initial_memories = self.long_term_memory.get_recent_memories(limit=100)
        self.logger.info(f"Loaded {len(initial_memories)} memories from long-term memory")
        
        # Index for fast retrieval
        self._build_retrieval_index()
    
    def _build_retrieval_index(self):
        """Build fast retrieval index"""
        self.logger.info("Building retrieval index...")
        
        # Build semantic index
        self._build_semantic_index()
        
        # Build keyword index
        self._build_keyword_index()
        
        # Build graph index
        self._build_graph_index()
        
        self.logger.info("Retrieval index built successfully")
    
    def _build_semantic_index(self):
        """Build semantic search index"""
        # This would integrate with a semantic search system
        self.semantic_index = {
            "embeddings": {},
            "similarity_matrix": {},
            "index_built": True
        }
    
    def _build_keyword_index(self):
        """Build keyword search index"""
        # This would integrate with a keyword search system
        self.keyword_index = {
            "inverted_index": {},
            "term_frequencies": {},
            "index_built": True
        }
    
    def _build_graph_index(self):
        """Build graph traversal index"""
        # This would integrate with the knowledge graph
        self.graph_index = {
            "node_index": {},
            "edge_index": {},
            "path_cache": {},
            "index_built": True
        }
    
    def _setup_retrieval_methods(self):
        """Set up retrieval methods"""
        self.logger.info("Setting up retrieval methods...")
        
        # Configure retrieval methods
        for method in RetrievalMethod:
            if method in self.retrieval_methods:
                self.logger.info(f"Configured retrieval method: {method.value}")
    
    def _setup_generation_strategies(self):
        """Set up generation strategies"""
        self.logger.info("Setting up generation strategies...")
        
        # Configure generation strategies
        for strategy in GenerationStrategy:
            if strategy in self.generation_strategies:
                self.logger.info(f"Configured generation strategy: {strategy.value}")
    
    async def _periodic_knowledge_update(self):
        """Periodic knowledge update task"""
        while True:
            try:
                await asyncio.sleep(self.knowledge_update_interval.total_seconds())
                
                # Update knowledge stores
                await self._update_knowledge_stores()
                
                # Rebuild indices if needed
                await self._rebuild_indices_if_needed()
                
                # Update last update timestamp
                self.last_knowledge_update = datetime.now()
                
                self.logger.info("Knowledge stores updated successfully")
                
            except Exception as e:
                self.logger.error(f"Error in periodic knowledge update: {e}")
    
    async def _update_knowledge_stores(self):
        """Update knowledge stores with new information"""
        self.logger.info("Updating knowledge stores...")
        
        # Update knowledge graph
        new_entities = await self._get_new_entities()
        for entity in new_entities:
            self.knowledge_graph.add_entity(entity)
        
        # Update long-term memory
        new_memories = await self._get_new_memories()
        for memory in new_memories:
            self.long_term_memory.store_memory(memory)
        
        # Update retrieval indices
        await self._update_retrieval_indices()
    
    async def _get_new_entities(self) -> List[Dict[str, Any]]:
        """Get new entities to add to knowledge graph"""
        # This would integrate with external data sources
        return []
    
    async def _get_new_memories(self) -> List[Dict[str, Any]]:
        """Get new memories to add to long-term memory"""
        # This would integrate with the memory system
        return []
    
    async def _update_retrieval_indices(self):
        """Update retrieval indices with new information"""
        # Update semantic index
        await self._update_semantic_index()
        
        # Update keyword index
        await self._update_keyword_index()
        
        # Update graph index
        await self._update_graph_index()
    
    async def _update_semantic_index(self):
        """Update semantic search index"""
        # This would integrate with semantic search system
        pass
    
    async def _update_keyword_index(self):
        """Update keyword search index"""
        # This would integrate with keyword search system
        pass
    
    async def _update_graph_index(self):
        """Update graph traversal index"""
        # This would integrate with knowledge graph
        pass
    
    async def _rebuild_indices_if_needed(self):
        """Rebuild indices if they need updating"""
        # Check if indices need rebuilding
        if self._should_rebuild_indices():
            self.logger.info("Rebuilding retrieval indices...")
            self._build_retrieval_index()
    
    def _should_rebuild_indices(self) -> bool:
        """Check if indices should be rebuilt"""
        # Check if knowledge has changed significantly
        return False  # Placeholder
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming messages for RAG operations"""
        start_time = time.time()
        
        try:
            # Extract query and parameters
            query = message.get("query", "")
            retrieval_method = message.get("retrieval_method", RetrievalMethod.HYBRID_SEARCH.value)
            generation_strategy = message.get("generation_strategy", GenerationStrategy.ADAPTIVE_GENERATION.value)
            max_contexts = message.get("max_contexts", self.max_retrieved_contexts)
            
            # Check cache first
            cache_key = self._generate_cache_key(query, retrieval_method, generation_strategy)
            if cache_key in self.response_cache:
                cached_result = self.response_cache[cache_key]
                if datetime.now() - cached_result["timestamp"] < self.cache_ttl:
                    self.cache_hits += 1
                    return {
                        "status": "success",
                        "result": cached_result["result"],
                        "cached": True,
                        "processing_time": time.time() - start_time
                    }
            
            self.cache_misses += 1
            
            # Perform RAG operation
            result = await self._perform_rag_operation(
                query, retrieval_method, generation_strategy, max_contexts
            )
            
            # Cache the result
            self.response_cache[cache_key] = {
                "result": result,
                "timestamp": datetime.now()
            }
            
            return {
                "status": "success",
                "result": result,
                "cached": False,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Error in RAG processing: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _generate_cache_key(self, query: str, retrieval_method: str, generation_strategy: str) -> str:
        """Generate cache key for RAG operation"""
        key_data = f"{query}:{retrieval_method}:{generation_strategy}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _perform_rag_operation(
        self, 
        query: str, 
        retrieval_method: str, 
        generation_strategy: str, 
        max_contexts: int
    ) -> GenerationResult:
        """Perform RAG operation: retrieve and generate"""
        
        # Step 1: Retrieve relevant contexts
        retrieved_contexts = await self._retrieve_contexts(
            query, RetrievalMethod(retrieval_method), max_contexts
        )
        
        # Step 2: Generate response using retrieved contexts
        generation_result = await self._generate_response(
            query, retrieved_contexts, GenerationStrategy(generation_strategy)
        )
        
        return generation_result
    
    async def _retrieve_contexts(
        self, 
        query: str, 
        method: RetrievalMethod, 
        max_contexts: int
    ) -> List[RetrievedContext]:
        """Retrieve relevant contexts using specified method"""
        
        # Get retrieval function
        retrieval_func = self.retrieval_methods.get(method)
        if not retrieval_func:
            raise ValueError(f"Unknown retrieval method: {method}")
        
        # Perform retrieval
        raw_results = await retrieval_func(query, max_contexts)
        
        # Convert to RetrievedContext objects
        contexts = []
        for result in raw_results:
            context = RetrievedContext(
                content=result["content"],
                source=result["source"],
                relevance_score=result["relevance_score"],
                retrieval_method=method,
                metadata=result.get("metadata", {}),
                timestamp=datetime.now()
            )
            contexts.append(context)
        
        # Filter by relevance threshold
        contexts = [ctx for ctx in contexts if ctx.relevance_score >= self.min_relevance_threshold]
        
        # Sort by relevance
        contexts.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Update statistics
        self.retrieval_stats[method.value] += 1
        
        return contexts[:max_contexts]
    
    async def _semantic_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform semantic search"""
        # This would integrate with a semantic search system
        results = []
        
        # Simulate semantic search
        for i in range(min(max_results, 3)):
            results.append({
                "content": f"Semantic search result {i+1} for query: {query}",
                "source": "semantic_index",
                "relevance_score": np.random.uniform(0.7, 0.95),
                "metadata": {"embedding_similarity": np.random.uniform(0.8, 0.9)}
            })
        
        return results
    
    async def _keyword_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform keyword search"""
        # This would integrate with a keyword search system
        results = []
        
        # Extract keywords from query
        keywords = query.lower().split()
        
        # Simulate keyword search
        for i in range(min(max_results, 3)):
            results.append({
                "content": f"Keyword search result {i+1} for keywords: {keywords}",
                "source": "keyword_index",
                "relevance_score": np.random.uniform(0.6, 0.85),
                "metadata": {"keyword_matches": len(keywords)}
            })
        
        return results
    
    async def _hybrid_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword search"""
        # Combine semantic and keyword search results
        semantic_results = await self._semantic_search(query, max_results // 2)
        keyword_results = await self._keyword_search(query, max_results // 2)
        
        # Merge and deduplicate results
        all_results = semantic_results + keyword_results
        
        # Sort by relevance
        all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return all_results[:max_results]
    
    async def _graph_traversal(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform graph traversal search"""
        # This would integrate with the knowledge graph
        results = []
        
        # Simulate graph traversal
        for i in range(min(max_results, 3)):
            results.append({
                "content": f"Graph traversal result {i+1} for query: {query}",
                "source": "knowledge_graph",
                "relevance_score": np.random.uniform(0.65, 0.9),
                "metadata": {"path_length": np.random.randint(1, 5)}
            })
        
        return results
    
    async def _memory_retrieval(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform memory-based retrieval"""
        # This would integrate with the long-term memory system
        results = []
        
        # Simulate memory retrieval
        for i in range(min(max_results, 3)):
            results.append({
                "content": f"Memory retrieval result {i+1} for query: {query}",
                "source": "long_term_memory",
                "relevance_score": np.random.uniform(0.6, 0.8),
                "metadata": {"memory_type": "episodic" if i % 2 == 0 else "semantic"}
            })
        
        return results
    
    async def _generate_response(
        self, 
        query: str, 
        contexts: List[RetrievedContext], 
        strategy: GenerationStrategy
    ) -> GenerationResult:
        """Generate response using retrieved contexts"""
        
        # Get generation function
        generation_func = self.generation_strategies.get(strategy)
        if not generation_func:
            raise ValueError(f"Unknown generation strategy: {strategy}")
        
        # Perform generation
        start_time = time.time()
        generated_text = await generation_func(query, contexts)
        processing_time = time.time() - start_time
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(query, contexts, generated_text)
        
        # Update statistics
        self.generation_stats[strategy.value] += 1
        
        return GenerationResult(
            generated_text=generated_text,
            retrieved_contexts=contexts,
            confidence_score=confidence_score,
            generation_strategy=strategy,
            processing_time=processing_time,
            metadata={"context_count": len(contexts)}
        )
    
    async def _conditional_generation(self, query: str, contexts: List[RetrievedContext]) -> str:
        """Generate response using conditional generation"""
        # This would integrate with a language model
        context_text = "\n".join([ctx.content for ctx in contexts])
        
        # Simulate conditional generation
        response = f"Based on the retrieved information: {context_text[:100]}...\n\n"
        response += f"Answer to your query '{query}': This is a generated response using conditional generation."
        
        return response
    
    async def _template_based_generation(self, query: str, contexts: List[RetrievedContext]) -> str:
        """Generate response using template-based generation"""
        # This would use predefined templates
        template = "Based on {context_count} relevant sources:\n{context_summary}\n\nAnswer: {answer}"
        
        context_summary = "\n".join([f"- {ctx.content[:50]}..." for ctx in contexts])
        answer = f"This is a template-based response to: {query}"
        
        response = template.format(
            context_count=len(contexts),
            context_summary=context_summary,
            answer=answer
        )
        
        return response
    
    async def _adaptive_generation(self, query: str, contexts: List[RetrievedContext]) -> str:
        """Generate response using adaptive generation"""
        # This would adapt the generation strategy based on context
        if len(contexts) == 0:
            return f"I don't have enough information to answer: {query}"
        elif len(contexts) == 1:
            return f"Based on one source: {contexts[0].content[:100]}...\n\nAnswer: This is an adaptive response."
        else:
            return f"Based on {len(contexts)} sources, here's what I found:\n\n" + \
                   "\n".join([f"- {ctx.content[:50]}..." for ctx in contexts]) + \
                   f"\n\nAnswer: This is an adaptive response to: {query}"
    
    async def _context_aware_generation(self, query: str, contexts: List[RetrievedContext]) -> str:
        """Generate response using context-aware generation"""
        # This would be aware of the context and adapt accordingly
        if not contexts:
            return f"I couldn't find relevant information for: {query}"
        
        # Analyze context relevance
        high_relevance = [ctx for ctx in contexts if ctx.relevance_score > 0.8]
        medium_relevance = [ctx for ctx in contexts if 0.5 <= ctx.relevance_score <= 0.8]
        
        response = f"Query: {query}\n\n"
        
        if high_relevance:
            response += "Highly relevant sources:\n"
            for ctx in high_relevance:
                response += f"- {ctx.content[:80]}...\n"
        
        if medium_relevance:
            response += "\nAdditional relevant sources:\n"
            for ctx in medium_relevance:
                response += f"- {ctx.content[:60]}...\n"
        
        response += f"\nContext-aware answer: This response was generated considering {len(contexts)} relevant sources."
        
        return response
    
    def _calculate_confidence_score(
        self, 
        query: str, 
        contexts: List[RetrievedContext], 
        generated_text: str
    ) -> float:
        """Calculate confidence score for generated response"""
        
        if not contexts:
            return 0.1  # Low confidence if no contexts
        
        # Calculate average relevance of contexts
        avg_relevance = np.mean([ctx.relevance_score for ctx in contexts])
        
        # Calculate context coverage
        context_coverage = min(len(contexts) / self.max_retrieved_contexts, 1.0)
        
        # Calculate text quality (simplified)
        text_quality = min(len(generated_text) / 100, 1.0)
        
        # Combine factors
        confidence = (avg_relevance * 0.5 + context_coverage * 0.3 + text_quality * 0.2)
        
        return min(confidence, 1.0)
    
    async def add_knowledge(self, knowledge_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add new knowledge to the RAG system"""
        try:
            # Add to knowledge graph
            if "entities" in knowledge_data:
                for entity in knowledge_data["entities"]:
                    # Extract entity parameters
                    name = entity.get("name", "Unknown")
                    entity_type_str = entity.get("type", "concept")
                    
                    # Convert string to EntityType enum
                    from knowledge_graphs.knowledge_graph import EntityType
                    entity_type_map = {
                        "person": EntityType.PERSON,
                        "organization": EntityType.ORGANIZATION,
                        "location": EntityType.LOCATION,
                        "concept": EntityType.CONCEPT,
                        "event": EntityType.EVENT,
                        "object": EntityType.OBJECT,
                        "action": EntityType.ACTION,
                        "attribute": EntityType.ATTRIBUTE
                    }
                    entity_type = entity_type_map.get(entity_type_str, EntityType.CONCEPT)
                    
                    # Extract other parameters
                    attributes = entity.get("properties", {})
                    description = attributes.get("description", "")
                    
                    # Add entity to knowledge graph
                    self.knowledge_graph.add_entity(
                        name=name,
                        entity_type=entity_type,
                        attributes=attributes,
                        description=description
                    )
            
            # Add to long-term memory
            if "memories" in knowledge_data:
                for memory in knowledge_data["memories"]:
                    # Extract memory parameters
                    content = memory.get("content", "")
                    memory_type_str = memory.get("type", "semantic")
                    
                    # Convert string to MemoryType enum
                    from memory.long_term_memory import MemoryType
                    memory_type_map = {
                        "episodic": MemoryType.EPISODIC,
                        "semantic": MemoryType.SEMANTIC,
                        "procedural": MemoryType.PROCEDURAL,
                        "working": MemoryType.WORKING
                    }
                    memory_type = memory_type_map.get(memory_type_str, MemoryType.SEMANTIC)
                    
                    # Store memory
                    self.long_term_memory.store_memory(
                        content=content,
                        memory_type=memory_type
                    )
            
            # Update indices
            await self._update_retrieval_indices()
            
            return {
                "status": "success",
                "message": "Knowledge added successfully",
                "entities_added": len(knowledge_data.get("entities", [])),
                "memories_added": len(knowledge_data.get("memories", []))
            }
            
        except Exception as e:
            self.logger.error(f"Error adding knowledge: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        return {
            "retrieval_stats": dict(self.retrieval_stats),
            "generation_stats": dict(self.generation_stats),
            "cache_stats": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            },
            "knowledge_stats": {
                "entities": len(self.knowledge_graph.get_all_entities()),
                "memories": len(self.long_term_memory.get_recent_memories(limit=1000)),
                "last_update": self.last_knowledge_update.isoformat()
            },
            "performance": {
                "avg_retrieval_time": 0.1,  # Placeholder
                "avg_generation_time": 0.5,  # Placeholder
                "total_operations": sum(self.retrieval_stats.values())
            }
        }
    
    async def clear_cache(self) -> Dict[str, Any]:
        """Clear RAG system cache"""
        cache_size = len(self.response_cache)
        self.response_cache.clear()
        self.context_cache.clear()
        
        return {
            "status": "success",
            "message": "Cache cleared successfully",
            "cleared_entries": cache_size
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent"""
        return {
            "name": "RAGAgent",
            "description": "Retrieval-Augmented Generation agent with fast retriever + small generator architecture",
            "capabilities": [
                "Semantic search",
                "Keyword search",
                "Hybrid search",
                "Graph traversal",
                "Memory retrieval",
                "Conditional generation",
                "Template-based generation",
                "Adaptive generation",
                "Context-aware generation"
            ],
            "status": "active",
            "retrieval_methods": [method.value for method in RetrievalMethod],
            "generation_strategies": [strategy.value for strategy in GenerationStrategy],
            "stats": {
                "total_retrievals": sum(self.retrieval_stats.values()),
                "total_generations": sum(self.generation_stats.values()),
                "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            }
        } 