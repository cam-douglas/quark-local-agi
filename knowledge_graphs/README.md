# Pillar 17: Long-Term Memory & Knowledge Graphs

This directory contains the implementation of **Pillar 17** of the Quark AI Assistant, focusing on **Long-Term Memory & Knowledge Graphs**.

## 🎯 Overview

Pillar 17 implements advanced knowledge graph construction, entity extraction, relationship discovery, and cross-document reasoning capabilities. It provides the foundation for world modeling and semantic knowledge representation.

## 🏗️ Architecture

### Core Components

1. **Knowledge Graph** (`knowledge_graph.py`)
   - Entity and relationship management
   - Graph traversal and reasoning
   - Semantic similarity search
   - Export/import functionality

2. **Entity Extractor** (`entity_extractor.py`)
   - Multi-method entity extraction (spaCy, transformers, patterns, rules)
   - Named Entity Recognition (NER)
   - Custom entity patterns
   - Entity type classification

3. **Relationship Extractor** (`relationship_extractor.py`)
   - Pattern-based relationship discovery
   - Dependency parsing
   - Co-occurrence analysis
   - Relationship type inference

4. **Graph Reasoner** (`graph_reasoner.py`)
   - Path-based reasoning
   - Entity similarity analysis
   - Community detection
   - Centrality analysis
   - Relationship inference

5. **World Model** (`world_model.py`)
   - Cross-document knowledge integration
   - Fact consolidation
   - Temporal reasoning
   - Causal inference

### Memory System

6. **Long-Term Memory** (`../memory/long_term_memory.py`)
   - Persistent memory storage
   - Semantic retrieval
   - Memory consolidation
   - Importance-based retention

### Integration

7. **Knowledge Graph Agent** (`../agents/knowledge_graph_agent.py`)
   - Unified interface for all components
   - Document processing pipeline
   - Query and reasoning operations
   - Statistics and monitoring

## 🚀 Features

### Knowledge Graph Construction
- **Entity Extraction**: Multi-method entity discovery from text
- **Relationship Discovery**: Automatic relationship extraction between entities
- **Graph Management**: Add, update, and query entities and relationships
- **Semantic Search**: Find entities using natural language queries

### Advanced Reasoning
- **Path Finding**: Discover connections between entities
- **Similarity Analysis**: Find similar entities based on attributes and embeddings
- **Community Detection**: Identify clusters of related entities
- **Centrality Analysis**: Find the most important entities in the graph
- **Relationship Inference**: Suggest potential relationships between entities

### Long-Term Memory
- **Persistent Storage**: JSON-based memory persistence
- **Semantic Retrieval**: Find memories using semantic similarity
- **Memory Consolidation**: Merge similar memories to reduce redundancy
- **Importance Tracking**: Prioritize memories based on importance and access patterns

### Cross-Document Reasoning
- **World Modeling**: Integrate knowledge across multiple documents
- **Fact Extraction**: Extract and store facts from documents
- **Cross-Reference Queries**: Query knowledge across document boundaries
- **Temporal Reasoning**: Analyze temporal relationships between events

## 📊 Usage Examples

### Basic Document Processing
```python
from agents.knowledge_graph_agent import KnowledgeGraphAgent

agent = KnowledgeGraphAgent()

# Process a document
result = agent.generate(
    "Apple Inc. is a technology company. Tim Cook is the CEO.",
    operation='process_document',
    document_id='apple_info'
)
```

### Entity Extraction
```python
# Extract entities from text
entities = agent.generate(
    "Google was founded by Larry Page and Sergey Brin in 1998.",
    operation='extract_entities'
)
```

### Knowledge Querying
```python
# Query the knowledge graph
results = agent.generate(
    "Apple CEO",
    operation='query_knowledge'
)
```

### Advanced Reasoning
```python
# Find connections between entities
connections = agent.generate(
    "",
    operation='reason',
    reasoning_type='connections',
    entities=['Apple', 'Tim Cook']
)
```

### Memory Operations
```python
# Store memory
memory_id = agent.generate(
    "Tesla produces electric vehicles",
    operation='store_memory',
    memory_type='company_info',
    importance=0.8
)

# Retrieve memories
memories = agent.generate(
    "electric vehicles",
    operation='retrieve_memory'
)
```

## 🛠️ CLI Interface

The knowledge graph functionality is accessible via CLI:

```bash
# Process a document
python cli/knowledge_graph_cli.py process_document --input "Apple is a technology company"

# Extract entities
python cli/knowledge_graph_cli.py extract_entities --input "Google was founded by Larry Page"

# Query knowledge
python cli/knowledge_graph_cli.py query_knowledge --input "Apple CEO"

# Get statistics
python cli/knowledge_graph_cli.py get_statistics
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python tests/test_pillar_17.py
```

This will test all major functionality including:
- Document processing
- Entity extraction
- Relationship discovery
- Knowledge querying
- Graph reasoning
- Memory operations

## 📈 Performance

### Scalability
- **Knowledge Graph**: Supports up to 10,000 entities and relationships
- **Memory System**: Configurable memory limits with automatic consolidation
- **Processing**: Efficient entity and relationship extraction

### Accuracy
- **Entity Extraction**: Multi-method approach with confidence scoring
- **Relationship Discovery**: Pattern-based with semantic validation
- **Memory Retrieval**: Semantic similarity with importance weighting

## 🔧 Configuration

### Knowledge Graph Settings
```python
# Configure graph parameters
knowledge_graph = KnowledgeGraph(graph_name="custom_graph")
```

### Memory Settings
```python
# Configure memory parameters
long_term_memory = LongTermMemory(storage_path="custom_memory.json")
```

### Entity Extraction Settings
```python
# Configure extraction methods
entity_extractor = EntityExtractor(
    use_spacy=True,
    use_transformers=True
)
```

## 📚 Dependencies

### Required Packages
- `networkx`: Graph operations
- `sentence-transformers`: Semantic embeddings
- `spacy`: NLP processing
- `numpy`: Numerical operations
- `nltk`: Natural language processing

### Optional Dependencies
- `transformers`: Advanced NER models
- `chromadb`: Vector storage (for enhanced memory)

## 🎯 Future Enhancements

### Planned Features
- **Graph Visualization**: Interactive graph visualization tools
- **Advanced NLP**: Integration with more sophisticated NLP models
- **Real-time Updates**: Live knowledge graph updates
- **Distributed Processing**: Multi-node knowledge graph processing
- **Advanced Reasoning**: More sophisticated reasoning algorithms

### Research Areas
- **Causal Inference**: Advanced causal relationship discovery
- **Temporal Reasoning**: Enhanced temporal knowledge representation
- **Multi-modal Knowledge**: Integration of text, image, and structured data
- **Federated Learning**: Distributed knowledge graph learning

## 🤝 Contributing

To contribute to Pillar 17:

1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** your changes
4. **Add** tests for new functionality
5. **Submit** a pull request

### Development Guidelines
- Follow the existing code structure
- Add comprehensive docstrings
- Include type hints
- Write unit tests for new features
- Update documentation

## 📄 License

This implementation is part of the Quark AI Assistant project and follows the same licensing terms.

---

**Pillar 17 Status**: ✅ **COMPLETED**

This implementation provides a solid foundation for advanced knowledge representation and reasoning capabilities in the Quark AI Assistant. 