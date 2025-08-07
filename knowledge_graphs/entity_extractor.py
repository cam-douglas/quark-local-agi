#!/usr/bin/env python3
"""
Entity Extractor for Knowledge Graph Construction

This module implements automatic entity extraction from text documents
for building knowledge graphs in Pillar 17.
"""

import re
import spacy
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from transformers import pipeline
import nltk
from nltk.chunk import RegexpParser
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


@dataclass
class ExtractedEntity:
    """Represents an extracted entity with metadata."""
    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float
    attributes: Dict[str, Any]


class EntityExtractor:
    """
    Multi-method entity extractor for knowledge graph construction.
    
    Supports:
    - Named Entity Recognition (NER)
    - Rule-based extraction
    - Pattern matching
    - Custom entity types
    """
    
    def __init__(self, use_spacy: bool = True, use_transformers: bool = True):
        self.use_spacy = use_spacy
        self.use_transformers = use_transformers
        
        # Initialize spaCy
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy model not available. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
                self.use_spacy = False
        
        # Initialize transformers NER
        if use_transformers:
            try:
                self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
            except Exception as e:
                print(f"Warning: Transformers NER not available: {e}")
                self.ner_pipeline = None
                self.use_transformers = False
        
        # Download NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('maxent_ne_chunker')
            nltk.download('words')
        
        # Custom entity patterns
        self.custom_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
            'phone': r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b',
            'time': r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
            'currency': r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|EUR|GBP)',
            'percentage': r'\d+(?:\.\d+)?%',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        }
        
        # Entity type mappings
        self.entity_type_mappings = {
            'PERSON': 'person',
            'ORG': 'organization',
            'GPE': 'location',
            'LOC': 'location',
            'DATE': 'date',
            'TIME': 'time',
            'MONEY': 'currency',
            'PERCENT': 'percentage',
            'QUANTITY': 'quantity',
            'CARDINAL': 'number',
            'ORDINAL': 'number',
            'FAC': 'facility',
            'PRODUCT': 'product',
            'EVENT': 'event',
            'WORK_OF_ART': 'work_of_art',
            'LAW': 'law',
            'LANGUAGE': 'language',
            'NORP': 'nationality',
            'email': 'contact',
            'url': 'url',
            'phone': 'contact',
            'ip_address': 'technical',
            'credit_card': 'sensitive'
        }
    
    def extract_entities(self, text: str, methods: List[str] = None) -> List[ExtractedEntity]:
        """
        Extract entities from text using multiple methods.
        
        Args:
            text: Input text to extract entities from
            methods: List of extraction methods to use
            
        Returns:
            List of extracted entities
        """
        if methods is None:
            methods = ['spacy', 'transformers', 'patterns', 'rules']
        
        all_entities = []
        
        for method in methods:
            if method == 'spacy' and self.use_spacy:
                entities = self._extract_with_spacy(text)
                all_entities.extend(entities)
            elif method == 'transformers' and self.use_transformers:
                entities = self._extract_with_transformers(text)
                all_entities.extend(entities)
            elif method == 'patterns':
                entities = self._extract_with_patterns(text)
                all_entities.extend(entities)
            elif method == 'rules':
                entities = self._extract_with_rules(text)
                all_entities.extend(entities)
        
        # Remove duplicates and merge overlapping entities
        return self._merge_entities(all_entities)
    
    def _extract_with_spacy(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using spaCy NER."""
        if not self.nlp:
            return []
        
        entities = []
        doc = self.nlp(text)
        
        for ent in doc.ents:
            entity_type = self.entity_type_mappings.get(ent.label_, ent.label_.lower())
            entity = ExtractedEntity(
                text=ent.text,
                entity_type=entity_type,
                start_pos=ent.start_char,
                end_pos=ent.end_char,
                confidence=0.9,  # spaCy doesn't provide confidence scores
                attributes={
                    'spacy_label': ent.label_,
                    'lemma': ent.lemma_,
                    'is_sentiment': ent.label_ in ['PERSON', 'ORG', 'GPE']
                }
            )
            entities.append(entity)
        
        return entities
    
    def _extract_with_transformers(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using transformers NER pipeline."""
        if not self.ner_pipeline:
            return []
        
        entities = []
        results = self.ner_pipeline(text)
        
        for result in results:
            entity_type = self.entity_type_mappings.get(result['entity'], result['entity'].lower())
            entity = ExtractedEntity(
                text=result['word'],
                entity_type=entity_type,
                start_pos=result.get('start', 0),
                end_pos=result.get('end', len(result['word'])),
                confidence=result['score'],
                attributes={
                    'transformers_label': result['entity'],
                    'index': result.get('index', 0)
                }
            )
            entities.append(entity)
        
        return entities
    
    def _extract_with_patterns(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using regex patterns."""
        entities = []
        
        for pattern_name, pattern in self.custom_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                entity_type = self.entity_type_mappings.get(pattern_name, pattern_name)
                entity = ExtractedEntity(
                    text=match.group(),
                    entity_type=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.8,  # Pattern matches are generally reliable
                    attributes={
                        'pattern_name': pattern_name,
                        'groups': match.groups()
                    }
                )
                entities.append(entity)
        
        return entities
    
    def _extract_with_rules(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using rule-based approaches."""
        entities = []
        
        # NLTK-based extraction
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Noun phrase chunking
            grammar = r"""
                NP: {<DT|PP\$>?<JJ.*>*<NN.*>+}
                NP: {<NNP>+}
            """
            chunk_parser = RegexpParser(grammar)
            chunks = chunk_parser.parse(pos_tags)
            
            for chunk in chunks:
                if hasattr(chunk, 'label') and chunk.label() == 'NP':
                    chunk_text = ' '.join(word for word, tag in chunk.leaves())
                    
                    # Determine entity type based on POS tags
                    entity_type = self._determine_entity_type(chunk.leaves())
                    
                    entity = ExtractedEntity(
                        text=chunk_text,
                        entity_type=entity_type,
                        start_pos=text.find(chunk_text),
                        end_pos=text.find(chunk_text) + len(chunk_text),
                        confidence=0.6,  # Rule-based extraction is less reliable
                        attributes={
                            'pos_tags': chunk.leaves(),
                            'chunk_type': 'NP'
                        }
                    )
                    entities.append(entity)
        except Exception as e:
            print(f"Error in rule-based extraction: {e}")
        
        return entities
    
    def _determine_entity_type(self, pos_tags: List[Tuple[str, str]]) -> str:
        """Determine entity type based on POS tags."""
        tags = [tag for _, tag in pos_tags]
        
        if 'NNP' in tags:
            return 'proper_noun'
        elif 'NN' in tags:
            return 'noun_phrase'
        elif 'JJ' in tags and 'NN' in tags:
            return 'descriptive_phrase'
        else:
            return 'phrase'
    
    def _merge_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Merge overlapping entities and remove duplicates."""
        if not entities:
            return []
        
        # Sort by start position
        entities.sort(key=lambda x: x.start_pos)
        
        merged = []
        current = entities[0]
        
        for next_entity in entities[1:]:
            # Check for overlap
            if (current.end_pos >= next_entity.start_pos and 
                current.entity_type == next_entity.entity_type):
                # Merge overlapping entities
                current = ExtractedEntity(
                    text=current.text,
                    entity_type=current.entity_type,
                    start_pos=current.start_pos,
                    end_pos=max(current.end_pos, next_entity.end_pos),
                    confidence=max(current.confidence, next_entity.confidence),
                    attributes={**current.attributes, **next_entity.attributes}
                )
            else:
                merged.append(current)
                current = next_entity
        
        merged.append(current)
        
        # Remove exact duplicates
        unique_entities = []
        seen = set()
        
        for entity in merged:
            key = (entity.text.lower(), entity.entity_type, entity.start_pos)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def add_custom_pattern(self, name: str, pattern: str, entity_type: str = None):
        """Add a custom regex pattern for entity extraction."""
        self.custom_patterns[name] = pattern
        if entity_type:
            self.entity_type_mappings[name] = entity_type
    
    def get_entity_statistics(self, entities: List[ExtractedEntity]) -> Dict[str, Any]:
        """Get statistics about extracted entities."""
        stats = {
            'total_entities': len(entities),
            'entity_types': {},
            'confidence_distribution': {
                'high': 0,    # > 0.8
                'medium': 0,  # 0.5-0.8
                'low': 0      # < 0.5
            }
        }
        
        for entity in entities:
            # Count by type
            stats['entity_types'][entity.entity_type] = stats['entity_types'].get(entity.entity_type, 0) + 1
            
            # Count by confidence
            if entity.confidence > 0.8:
                stats['confidence_distribution']['high'] += 1
            elif entity.confidence > 0.5:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1
        
        return stats 