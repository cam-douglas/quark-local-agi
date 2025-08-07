#!/usr/bin/env python3
"""
Natural Language Understanding Agent - Pillar 5
Handles intent classification, entity extraction, and language understanding
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
class Intent:
    """Identified intent from user input."""
    intent: str
    confidence: float
    entities: List[Dict[str, Any]]
    context: Dict[str, Any]
    timestamp: datetime

@dataclass
class NLUResult:
    """Complete NLU analysis result."""
    primary_intent: Intent
    secondary_intents: List[Intent]
    entities: List[Dict[str, Any]]
    sentiment: str
    confidence: float
    language: str
    timestamp: datetime

class NLUAgent(Agent):
    """Natural Language Understanding Agent for intent classification and entity extraction."""
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        super().__init__("nlu")
        self.model_name = model_name
        self.intent_labels = [
            "question", "statement", "command", "request", "complaint",
            "greeting", "farewell", "thanks", "apology", "clarification",
            "confirmation", "denial", "agreement", "disagreement", "suggestion",
            "warning", "information", "help", "feedback", "other"
        ]
        
        # Entity types for extraction
        self.entity_types = [
            "person", "organization", "location", "date", "time", "money",
            "percentage", "number", "email", "url", "phone", "product"
        ]
        
        # Sentiment labels
        self.sentiment_labels = ["positive", "negative", "neutral"]
        
        # Language detection labels
        self.language_labels = ["english", "spanish", "french", "german", "other"]
        
        # Load models
        self.load_model()
        
    def load_model(self):
        """Load NLU models for intent classification and entity extraction."""
        logger.info("Loading NLU models...")
        
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            
            # Intent classification model
            self.intent_classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                hypothesis_template="This example is {}."
            )
            
            # Sentiment analysis model
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            # Named entity recognition model
            self.ner_model = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english"
            )
            
            # Language detection model
            self.language_detector = pipeline(
                "text-classification",
                model="papluca/xlm-roberta-base-language-detection"
            )
            
            logger.info("✅ NLU models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading NLU models: {e}")
            # Fallback to basic functionality
            self.intent_classifier = None
            self.sentiment_analyzer = None
            self.ner_model = None
            self.language_detector = None
    
    def generate(self, prompt: str, **kwargs) -> NLUResult:
        """Generate NLU analysis for the given prompt."""
        try:
            # Ensure models are loaded
            self._ensure_model()
            
            # Perform intent classification
            primary_intent = self._classify_intent(prompt)
            
            # Extract entities
            entities = self._extract_entities(prompt)
            
            # Analyze sentiment
            sentiment = self._analyze_sentiment(prompt)
            
            # Detect language
            language = self._detect_language(prompt)
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(primary_intent, entities, sentiment)
            
            # Create NLU result
            result = NLUResult(
                primary_intent=primary_intent,
                secondary_intents=[],  # Could be extended for multiple intents
                entities=entities,
                sentiment=sentiment,
                confidence=confidence,
                language=language,
                timestamp=datetime.now()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in NLU analysis: {e}")
            # Return fallback result
            return self._create_fallback_result(prompt)
    
    def _classify_intent(self, text: str) -> Intent:
        """Classify the intent of the given text."""
        if not self.intent_classifier:
            return self._fallback_intent_classification(text)
        
        try:
            # Use the intent classifier
            result = self.intent_classifier(
                sequences=text,
                candidate_labels=self.intent_labels,
                hypothesis_template="This example is {}."
            )
            
            # Extract the most likely intent
            intent_label = result['labels'][0]
            confidence = result['scores'][0]
            
            # Create context based on the intent
            context = self._create_intent_context(intent_label, text)
            
            return Intent(
                intent=intent_label,
                confidence=confidence,
                entities=[],
                context=context,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            return self._fallback_intent_classification(text)
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from the text."""
        if not self.ner_model:
            return self._fallback_entity_extraction(text)
        
        try:
            # Use the NER model
            entities = self.ner_model(text)
            
            # Process and format entities
            processed_entities = []
            for entity in entities:
                processed_entities.append({
                    "text": entity["word"],
                    "type": entity["entity_group"],
                    "start": entity["start"],
                    "end": entity["end"],
                    "confidence": entity["score"]
                })
            
            return processed_entities
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return self._fallback_entity_extraction(text)
    
    def _analyze_sentiment(self, text: str) -> str:
        """Analyze the sentiment of the text."""
        if not self.sentiment_analyzer:
            return self._fallback_sentiment_analysis(text)
        
        try:
            # Use the sentiment analyzer
            result = self.sentiment_analyzer(text)
            
            # Map sentiment labels
            sentiment_mapping = {
                "LABEL_0": "negative",
                "LABEL_1": "neutral", 
                "LABEL_2": "positive"
            }
            
            sentiment = sentiment_mapping.get(result[0]["label"], "neutral")
            return sentiment
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._fallback_sentiment_analysis(text)
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        if not self.language_detector:
            return "english"  # Default fallback
        
        try:
            # Use the language detector
            result = self.language_detector(text)
            
            # Map language codes to our labels
            language_mapping = {
                "en": "english",
                "es": "spanish", 
                "fr": "french",
                "de": "german"
            }
            
            detected_lang = result[0]["label"]
            language = language_mapping.get(detected_lang, "other")
            return language
            
        except Exception as e:
            logger.error(f"Error in language detection: {e}")
            return "english"  # Default fallback
    
    def _calculate_confidence(self, intent: Intent, entities: List[Dict], sentiment: str) -> float:
        """Calculate overall confidence score for the NLU analysis."""
        # Base confidence from intent classification
        base_confidence = intent.confidence
        
        # Boost confidence if entities were found
        if entities:
            base_confidence += 0.1
        
        # Boost confidence if sentiment is clear
        if sentiment in ["positive", "negative"]:
            base_confidence += 0.05
        
        return min(base_confidence, 1.0)
    
    def _create_intent_context(self, intent: str, text: str) -> Dict[str, Any]:
        """Create context information based on the intent."""
        context = {
            "intent_type": intent,
            "text_length": len(text),
            "has_question_mark": "?" in text,
            "has_exclamation_mark": "!" in text,
            "word_count": len(text.split())
        }
        
        # Add intent-specific context
        if intent == "question":
            context["question_type"] = self._classify_question_type(text)
        elif intent == "command":
            context["command_type"] = self._classify_command_type(text)
        elif intent == "request":
            context["request_type"] = self._classify_request_type(text)
        
        return context
    
    def _classify_question_type(self, text: str) -> str:
        """Classify the type of question."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["what", "which", "where", "when", "who", "why", "how"]):
            return "wh_question"
        elif text_lower.startswith("can you") or text_lower.startswith("could you"):
            return "ability_question"
        elif text_lower.startswith("is") or text_lower.startswith("are") or text_lower.startswith("do"):
            return "yes_no_question"
        else:
            return "general_question"
    
    def _classify_command_type(self, text: str) -> str:
        """Classify the type of command."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["help", "assist", "support"]):
            return "help_command"
        elif any(word in text_lower for word in ["show", "display", "list", "find"]):
            return "information_command"
        elif any(word in text_lower for word in ["create", "make", "build", "generate"]):
            return "creation_command"
        elif any(word in text_lower for word in ["stop", "end", "quit", "exit"]):
            return "termination_command"
        else:
            return "general_command"
    
    def _classify_request_type(self, text: str) -> str:
        """Classify the type of request."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["please", "could you", "would you"]):
            return "polite_request"
        elif any(word in text_lower for word in ["need", "want", "require"]):
            return "need_request"
        elif any(word in text_lower for word in ["can you", "will you"]):
            return "ability_request"
        else:
            return "general_request"
    
    def _fallback_intent_classification(self, text: str) -> Intent:
        """Fallback intent classification when models are not available."""
        text_lower = text.lower()
        
        # Simple rule-based classification
        if any(word in text_lower for word in ["what", "how", "why", "when", "where", "who", "which"]):
            intent = "question"
            confidence = 0.8
        elif any(word in text_lower for word in ["please", "could you", "would you", "can you"]):
            intent = "request"
            confidence = 0.7
        elif text_lower.endswith("!") or any(word in text_lower for word in ["stop", "go", "do"]):
            intent = "command"
            confidence = 0.6
        elif any(word in text_lower for word in ["hello", "hi", "hey", "good morning"]):
            intent = "greeting"
            confidence = 0.9
        elif any(word in text_lower for word in ["bye", "goodbye", "see you"]):
            intent = "farewell"
            confidence = 0.9
        elif any(word in text_lower for word in ["thank", "thanks"]):
            intent = "thanks"
            confidence = 0.9
        else:
            intent = "statement"
            confidence = 0.5
        
        return Intent(
            intent=intent,
            confidence=confidence,
            entities=[],
            context={"fallback": True},
            timestamp=datetime.now()
        )
    
    def _fallback_entity_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Fallback entity extraction when models are not available."""
        entities = []
        
        # Simple rule-based entity extraction
        import re
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append({
                "text": match.group(),
                "type": "email",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.9
            })
        
        # Extract URLs
        url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
        for match in re.finditer(url_pattern, text):
            entities.append({
                "text": match.group(),
                "type": "url",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.9
            })
        
        # Extract numbers
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        for match in re.finditer(number_pattern, text):
            entities.append({
                "text": match.group(),
                "type": "number",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.8
            })
        
        return entities
    
    def _fallback_sentiment_analysis(self, text: str) -> str:
        """Fallback sentiment analysis when models are not available."""
        text_lower = text.lower()
        
        # Simple rule-based sentiment analysis
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "happy", "love", "like"]
        negative_words = ["bad", "terrible", "awful", "hate", "dislike", "sad", "angry", "frustrated"]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _create_fallback_result(self, prompt: str) -> NLUResult:
        """Create a fallback NLU result when analysis fails."""
        intent = self._fallback_intent_classification(prompt)
        entities = self._fallback_entity_extraction(prompt)
        sentiment = self._fallback_sentiment_analysis(prompt)
        
        return NLUResult(
            primary_intent=intent,
            secondary_intents=[],
            entities=entities,
            sentiment=sentiment,
            confidence=0.5,
            language="english",
            timestamp=datetime.now()
        )
    
    def get_intent_statistics(self) -> Dict[str, Any]:
        """Get statistics about intent classification performance."""
        return {
            "total_analyses": 0,  # Would be tracked in a real implementation
            "average_confidence": 0.0,
            "most_common_intents": [],
            "model_loaded": self.intent_classifier is not None
        }

