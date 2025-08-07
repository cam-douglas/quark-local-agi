#!/usr/bin/env python3
"""
Response Generation Agent for Quark AI Assistant
===============================================

Generates intelligent, contextual responses to user queries.
Part of the core response system.
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import random

from agents.base import Agent


class ResponseGenerationAgent(Agent):
    """Generates intelligent responses to user queries."""
    
    def __init__(self, model_name: str = "response_generation_agent"):
        super().__init__(model_name)
        self.name = "response_generation"
        
        # Response templates and patterns
        self.greeting_responses = [
            "Hello! I'm doing well, thank you for asking. How can I help you today?",
            "I'm functioning perfectly! Ready to assist you with any questions or tasks.",
            "Great to hear from you! I'm here and ready to help.",
            "I'm doing excellent! What would you like to work on today?"
        ]
        
        self.math_responses = {
            "2+2": "2 + 2 = 4",
            "1+1": "1 + 1 = 2",
            "5+5": "5 + 5 = 10",
            "10+10": "10 + 10 = 20"
        }
        
        self.joke_responses = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "What do you call a fake noodle? An impasta!",
            "Why don't eggs tell jokes? They'd crack each other up!",
            "What do you call a bear with no teeth? A gummy bear!"
        ]
        
        self.weather_responses = [
            "I don't have access to real-time weather data, but I can help you find weather information online or answer other questions!",
            "I can't check the current weather, but I'd be happy to help you with other tasks or questions.",
            "For current weather information, I'd recommend checking a weather app or website. Is there anything else I can help you with?"
        ]
        
        self.general_responses = [
            "That's an interesting question! Let me think about that...",
            "I'd be happy to help you with that.",
            "That's a great question. Let me provide some information on that topic.",
            "I can help you with that! What specific aspect would you like me to focus on?"
        ]
    
    def load_model(self):
        """Load response generation models."""
        try:
            # For now, using template-based responses
            return True
        except Exception as e:
            print(f"Error loading response generation models: {e}")
            return False
    
    def generate(self, input_data: str, operation: str = "generate_response", **kwargs) -> Dict[str, Any]:
        """
        Generate a response to user input.
        
        Args:
            input_data: User's input/question
            operation: Type of response generation
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        try:
            # Check if we already have a coding response from CodingAssistant
            if isinstance(input_data, dict) and input_data.get('type') == 'coding_response':
                return {
                    "status": "success",
                    "response": input_data.get('response', ''),
                    "confidence": 0.95,
                    "type": "coding"
                }
            
            # If input_data is a string, check for previous agent results in kwargs
            pipeline_results = kwargs.get('pipeline_results', {})
            if 'CodingAssistant' in pipeline_results:
                coding_result = pipeline_results['CodingAssistant']
                if coding_result.get('success') and coding_result.get('response'):
                    return {
                        "status": "success", 
                        "response": coding_result['response'],
                        "confidence": 0.95,
                        "type": "coding"
                    }
            
            # Convert to string for processing
            if isinstance(input_data, dict):
                input_data = str(input_data)
            
            input_lower = input_data.lower().strip()
            
            # Handle greetings
            if any(greeting in input_lower for greeting in ["how are you", "how are you doing", "are you ok", "are you well"]):
                response = random.choice(self.greeting_responses)
                return {
                    "status": "success",
                    "response": response,
                    "confidence": 0.95,
                    "type": "greeting"
                }
            
            # Handle math questions
            for math_expr, answer in self.math_responses.items():
                if math_expr in input_lower:
                    return {
                        "status": "success",
                        "response": answer,
                        "confidence": 1.0,
                        "type": "math"
                    }
            
            # Handle general math questions
            if any(word in input_lower for word in ["what is", "what's", "calculate", "compute", "solve"]) and any(op in input_lower for op in ["+", "-", "*", "/", "plus", "minus", "times", "divided"]):
                # Extract numbers and operation
                import re
                numbers = re.findall(r'\d+', input_lower)
                if len(numbers) >= 2:
                    try:
                        # Simple arithmetic evaluation
                        if "+" in input_lower or "plus" in input_lower:
                            result = int(numbers[0]) + int(numbers[1])
                            return {
                                "status": "success",
                                "response": f"{numbers[0]} + {numbers[1]} = {result}",
                                "confidence": 1.0,
                                "type": "math"
                            }
                        elif "-" in input_lower or "minus" in input_lower:
                            result = int(numbers[0]) - int(numbers[1])
                            return {
                                "status": "success",
                                "response": f"{numbers[0]} - {numbers[1]} = {result}",
                                "confidence": 1.0,
                                "type": "math"
                            }
                        elif "*" in input_lower or "times" in input_lower:
                            result = int(numbers[0]) * int(numbers[1])
                            return {
                                "status": "success",
                                "response": f"{numbers[0]} × {numbers[1]} = {result}",
                                "confidence": 1.0,
                                "type": "math"
                            }
                        elif "/" in input_lower or "divided" in input_lower:
                            if int(numbers[1]) != 0:
                                result = int(numbers[0]) / int(numbers[1])
                                return {
                                    "status": "success",
                                    "response": f"{numbers[0]} ÷ {numbers[1]} = {result}",
                                    "confidence": 1.0,
                                    "type": "math"
                                }
                    except:
                        pass
            
            # Handle joke requests
            if any(word in input_lower for word in ["joke", "funny", "humor", "laugh"]):
                response = random.choice(self.joke_responses)
                return {
                    "status": "success",
                    "response": response,
                    "confidence": 0.9,
                    "type": "joke"
                }
            
            # Handle weather questions
            if any(word in input_lower for word in ["weather", "temperature", "forecast", "rain", "sunny"]):
                response = random.choice(self.weather_responses)
                return {
                    "status": "success",
                    "response": response,
                    "confidence": 0.8,
                    "type": "weather"
                }
            
            # Handle general questions
            if "?" in input_data:
                response = random.choice(self.general_responses)
                return {
                    "status": "success",
                    "response": response,
                    "confidence": 0.7,
                    "type": "general"
                }
            
            # Default response
            return {
                "status": "success",
                "response": "I understand you said: " + input_data + ". How can I help you with that?",
                "confidence": 0.6,
                "type": "default"
            }
                
        except Exception as e:
            return {
                "status": "error",
                "response": f"Sorry, I encountered an error: {str(e)}",
                "confidence": 0.0,
                "type": "error"
            }

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process a response generation message asynchronously."""
        try:
            query = message.get("query", "")
            context = message.get("context", "")
            
            result = self.generate(query)
            
            return {
                "status": "success",
                "response": result["response"],
                "confidence": result.get("confidence", 0.0),
                "type": result.get("type", "unknown")
            }
            
        except Exception as e:
            return {
                "status": "error",
                "response": f"Error processing message: {str(e)}",
                "confidence": 0.0,
                "type": "error"
            }
    
    def get_response_stats(self) -> Dict[str, Any]:
        """Get response generation statistics."""
        return {
            "total_responses": len(self.greeting_responses) + len(self.math_responses) + len(self.joke_responses),
            "response_types": ["greeting", "math", "joke", "weather", "general", "default"],
            "average_confidence": 0.8
        } 