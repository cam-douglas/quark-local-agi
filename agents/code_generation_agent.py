#!/usr/bin/env python3
"""
Code Generation Agent
Open-source Claude-like code generation capabilities for Quark

This agent provides advanced code generation, completion, and assistance
using state-of-the-art open-source language models.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import re
from datetime import datetime

# Import transformers for model loading
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        CodeGenTokenizer, CodeGenForCausalLM,
        pipeline, TextGenerationPipeline
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class CodeGenerationRequest:
    """Request for code generation"""
    prompt: str
    language: str = "python"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    context: Optional[str] = None
    task_type: str = "completion"  # completion, generation, explanation, refactoring

@dataclass
class CodeGenerationResponse:
    """Response from code generation"""
    generated_code: str
    confidence: float
    language: str
    explanation: Optional[str] = None
    suggestions: List[str] = None
    metadata: Dict[str, Any] = None

class CodeGenerationAgent:
    """Advanced code generation agent with Claude-like capabilities"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        self.model_configs = self._get_model_configs()
        self.device = self._get_device()
        self.model = None  # Compatibility with Agent base class
        self.initialize_models()
    
    def _ensure_model(self):
        """Ensure models are loaded (compatibility with Agent base class)"""
        if not self.models:
            self.initialize_models()
        
    def _get_device(self) -> str:
        """Determine the best device for model inference"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configurations for different code generation models"""
        return {
            "codegen": {
                "model_name": "Salesforce/codegen-350M-mono",
                "description": "Salesforce CodeGen - Python-focused code generation",
                "specialties": ["python", "general_coding"],
                "max_length": 2048
            },
            "starcoder": {
                "model_name": "bigcode/starcoder",
                "description": "StarCoder - Multi-language code generation",
                "specialties": ["python", "javascript", "typescript", "java", "cpp"],
                "max_length": 8192
            },
            "codellama": {
                "model_name": "codellama/CodeLlama-7b-Python-hf",
                "description": "Code Llama - Advanced Python code generation",
                "specialties": ["python", "code_explanation", "debugging"],
                "max_length": 4096
            },
            "wizardcoder": {
                "model_name": "WizardLM/WizardCoder-Python-7B-V1.0",
                "description": "WizardCoder - Instruction-tuned Python coding",
                "specialties": ["python", "problem_solving", "algorithms"],
                "max_length": 2048
            }
        }
    
    def initialize_models(self):
        """Initialize code generation models"""
        logger.info("🤖 Initializing code generation models...")
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("❌ Transformers library not available. Please install with: pip install transformers torch")
            return
        
        # Start with CodeGen as it's more lightweight
        try:
            self._load_codegen_model()
            logger.info("✅ CodeGen model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load CodeGen model: {e}")
        
        # Add a general-purpose pipeline for quick responses
        try:
            self._setup_general_pipeline()
            logger.info("✅ General code pipeline setup successful")
        except Exception as e:
            logger.error(f"❌ Failed to setup general pipeline: {e}")
    
    def _load_codegen_model(self):
        """Load the CodeGen model for Python code generation"""
        model_name = "Salesforce/codegen-350M-mono"
        
        logger.info(f"📦 Loading CodeGen model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizers["codegen"] = CodeGenTokenizer.from_pretrained(model_name)
        self.models["codegen"] = CodeGenForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device != "cpu" else None
        )
        
        if self.device != "cpu":
            self.models["codegen"] = self.models["codegen"].to(self.device)
    
    def _setup_general_pipeline(self):
        """Setup a general text generation pipeline for code"""
        try:
            # Use a smaller model for general code assistance
            self.pipelines["general"] = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
            )
        except Exception as e:
            logger.warning(f"Could not load general pipeline, using fallback: {e}")
            # Fallback to a simpler approach
            self.pipelines["general"] = None
    
    async def generate_code(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Generate code based on the request"""
        logger.info(f"🔄 Generating code for: {request.task_type} in {request.language}")
        
        try:
            if request.task_type == "completion":
                return await self._complete_code(request)
            elif request.task_type == "generation":
                return await self._generate_new_code(request)
            elif request.task_type == "explanation":
                return await self._explain_code(request)
            elif request.task_type == "refactoring":
                return await self._refactor_code(request)
            else:
                return await self._complete_code(request)  # Default to completion
                
        except Exception as e:
            logger.error(f"❌ Code generation failed: {e}")
            return CodeGenerationResponse(
                generated_code="# Error generating code",
                confidence=0.0,
                language=request.language,
                explanation=f"Generation failed: {str(e)}"
            )
    
    async def _complete_code(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Complete partial code"""
        if "codegen" in self.models:
            return await self._use_codegen_model(request)
        else:
            return await self._use_fallback_completion(request)
    
    async def _generate_new_code(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Generate new code from description"""
        # Enhance prompt for code generation
        enhanced_prompt = self._enhance_generation_prompt(request.prompt, request.language)
        enhanced_request = CodeGenerationRequest(
            prompt=enhanced_prompt,
            language=request.language,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            context=request.context,
            task_type=request.task_type
        )
        
        return await self._complete_code(enhanced_request)
    
    async def _explain_code(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Explain existing code"""
        explanation_prompt = f"""
# Code Explanation Request
# Please explain the following {request.language} code:

{request.prompt}

# Explanation:
"""
        
        explanation_request = CodeGenerationRequest(
            prompt=explanation_prompt,
            language=request.language,
            max_length=min(request.max_length, 1024),
            temperature=0.3,  # Lower temperature for explanations
            top_p=request.top_p,
            context=request.context,
            task_type="completion"
        )
        
        response = await self._complete_code(explanation_request)
        response.explanation = response.generated_code
        response.generated_code = request.prompt  # Return original code
        
        return response
    
    async def _refactor_code(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Refactor existing code"""
        refactor_prompt = f"""
# Code Refactoring Request
# Please refactor the following {request.language} code to improve readability, performance, and maintainability:

{request.prompt}

# Refactored code:
"""
        
        refactor_request = CodeGenerationRequest(
            prompt=refactor_prompt,
            language=request.language,
            max_length=request.max_length,
            temperature=0.5,  # Moderate temperature for refactoring
            top_p=request.top_p,
            context=request.context,
            task_type="completion"
        )
        
        return await self._complete_code(refactor_request)
    
    async def _use_codegen_model(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Use the CodeGen model for code generation"""
        try:
            tokenizer = self.tokenizers["codegen"]
            model = self.models["codegen"]
            
            # Prepare the input
            inputs = tokenizer.encode(request.prompt, return_tensors="pt")
            if self.device != "cpu":
                inputs = inputs.to(self.device)
            
            # Generate code
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=min(inputs.shape[1] + request.max_length, 2048),
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode the generated code
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_code = generated_text[len(request.prompt):].strip()
            
            # Clean up the generated code
            generated_code = self._clean_generated_code(generated_code, request.language)
            
            # Calculate confidence (simplified)
            confidence = min(0.8, len(generated_code) / 100)
            
            return CodeGenerationResponse(
                generated_code=generated_code,
                confidence=confidence,
                language=request.language,
                explanation=f"Generated using CodeGen model",
                metadata={
                    "model": "codegen",
                    "temperature": request.temperature,
                    "max_length": request.max_length
                }
            )
            
        except Exception as e:
            logger.error(f"❌ CodeGen model generation failed: {e}")
            return await self._use_fallback_completion(request)
    
    async def _use_fallback_completion(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """Fallback code completion using templates and patterns"""
        logger.info("🔄 Using fallback code completion")
        
        # Simple pattern-based completion
        completion = self._pattern_based_completion(request.prompt, request.language)
        
        return CodeGenerationResponse(
            generated_code=completion,
            confidence=0.5,
            language=request.language,
            explanation="Generated using pattern-based fallback",
            suggestions=self._get_code_suggestions(request.prompt, request.language),
            metadata={"method": "fallback", "pattern_based": True}
        )
    
    def _enhance_generation_prompt(self, prompt: str, language: str) -> str:
        """Enhance prompt for better code generation"""
        if language.lower() == "python":
            enhanced = f"""# Python code generation
# Task: {prompt}
# Please generate clean, well-commented Python code:

"""
        elif language.lower() == "javascript":
            enhanced = f"""// JavaScript code generation
// Task: {prompt}
// Please generate clean, modern JavaScript code:

"""
        elif language.lower() == "typescript":
            enhanced = f"""// TypeScript code generation
// Task: {prompt}
// Please generate clean, type-safe TypeScript code:

"""
        else:
            enhanced = f"""// {language} code generation
// Task: {prompt}
// Please generate clean, well-structured code:

"""
        
        return enhanced
    
    def _clean_generated_code(self, code: str, language: str) -> str:
        """Clean up generated code"""
        # Remove common generation artifacts
        code = re.sub(r'<\|endoftext\|>', '', code)
        code = re.sub(r'</s>', '', code)
        code = re.sub(r'<unk>', '', code)
        
        # Split by lines and clean
        lines = code.split('\n')
        cleaned_lines = []
        seen_lines = set()
        repetition_count = {}
        
        for line in lines:
            # Skip empty lines at the beginning
            if not cleaned_lines and not line.strip():
                continue
            
            # Stop at certain markers
            if any(marker in line.lower() for marker in ['# end', '# stop', 'endoftext']):
                break
            
            # Stop at excessive repetition
            line_stripped = line.strip()
            if line_stripped:
                repetition_count[line_stripped] = repetition_count.get(line_stripped, 0) + 1
                if repetition_count[line_stripped] > 3:  # Stop after 3 repetitions
                    break
                    
            cleaned_lines.append(line)
        
        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        
        # Limit output to reasonable length (50 lines max)
        if len(cleaned_lines) > 50:
            cleaned_lines = cleaned_lines[:50]
            cleaned_lines.append("# ... (output truncated)")
        
        return '\n'.join(cleaned_lines)
    
    def _pattern_based_completion(self, prompt: str, language: str) -> str:
        """Simple pattern-based code completion"""
        prompt_lower = prompt.lower().strip()
        
        if language.lower() == "python":
            return self._python_patterns(prompt, prompt_lower)
        elif language.lower() in ["javascript", "js"]:
            return self._javascript_patterns(prompt, prompt_lower)
        elif language.lower() in ["typescript", "ts"]:
            return self._typescript_patterns(prompt, prompt_lower)
        else:
            return self._generic_patterns(prompt, prompt_lower)
    
    def _python_patterns(self, prompt: str, prompt_lower: str) -> str:
        """Python-specific code patterns"""
        if "def " in prompt_lower and prompt.rstrip().endswith(":"):
            return '''    """
    Function implementation
    """
    pass'''
        
        elif "class " in prompt_lower and prompt.rstrip().endswith(":"):
            return '''    """
    Class implementation
    """
    def __init__(self):
        pass'''
        
        elif prompt_lower.startswith("import ") or prompt_lower.startswith("from "):
            return "\n# Additional imports can be added here"
        
        elif "for " in prompt_lower and prompt.rstrip().endswith(":"):
            return '''    # Loop implementation
    pass'''
        
        elif "if " in prompt_lower and prompt.rstrip().endswith(":"):
            return '''    # Condition implementation
    pass'''
        
        elif "try:" in prompt_lower:
            return '''    # Try block implementation
    pass
except Exception as e:
    # Error handling
    print(f"Error: {e}")'''
        
        else:
            return "# Code implementation needed"
    
    def _javascript_patterns(self, prompt: str, prompt_lower: str) -> str:
        """JavaScript-specific code patterns"""
        if "function " in prompt_lower and prompt.rstrip().endswith("{"):
            return '''    // Function implementation
    return null;
}'''
        
        elif "const " in prompt_lower and "=>" in prompt:
            return " {\n    // Arrow function implementation\n    return null;\n};"
        
        elif prompt_lower.startswith("class ") and prompt.rstrip().endswith("{"):
            return '''    constructor() {
        // Constructor implementation
    }
}'''
        
        elif "if " in prompt_lower and prompt.rstrip().endswith("{"):
            return '''    // Condition implementation
}'''
        
        elif "for " in prompt_lower and prompt.rstrip().endswith("{"):
            return '''    // Loop implementation
}'''
        
        else:
            return "// Code implementation needed"
    
    def _typescript_patterns(self, prompt: str, prompt_lower: str) -> str:
        """TypeScript-specific code patterns"""
        js_completion = self._javascript_patterns(prompt, prompt_lower)
        
        # Add TypeScript-specific enhancements
        if "interface " in prompt_lower and prompt.rstrip().endswith("{"):
            return '''    // Interface properties
    [key: string]: any;
}'''
        
        elif "type " in prompt_lower and "=" in prompt:
            return " string | number | boolean;"
        
        else:
            return js_completion
    
    def _generic_patterns(self, prompt: str, prompt_lower: str) -> str:
        """Generic code patterns for other languages"""
        if prompt.rstrip().endswith("{"):
            return '''    // Implementation needed
}'''
        elif prompt.rstrip().endswith(":"):
            return "    # Implementation needed"
        else:
            return "// Code completion needed"
    
    def _get_code_suggestions(self, prompt: str, language: str) -> List[str]:
        """Get code suggestions based on the prompt"""
        suggestions = []
        prompt_lower = prompt.lower()
        
        if language.lower() == "python":
            if "def " in prompt_lower:
                suggestions.extend([
                    "Add docstring for function documentation",
                    "Consider type hints for parameters and return value",
                    "Add error handling with try/except blocks"
                ])
            elif "class " in prompt_lower:
                suggestions.extend([
                    "Add __init__ method for initialization",
                    "Consider adding __str__ and __repr__ methods",
                    "Add class docstring"
                ])
        
        elif language.lower() in ["javascript", "typescript"]:
            if "function " in prompt_lower:
                suggestions.extend([
                    "Add JSDoc comments for documentation",
                    "Consider using async/await for asynchronous operations",
                    "Add input validation"
                ])
            elif "class " in prompt_lower:
                suggestions.extend([
                    "Add constructor method",
                    "Consider private properties with #",
                    "Add getter/setter methods if needed"
                ])
        
        # General suggestions
        suggestions.extend([
            "Add comments for complex logic",
            "Consider error handling",
            "Follow language-specific naming conventions"
        ])
        
        return suggestions[:3]  # Return top 3 suggestions
    
    async def get_code_explanation(self, code: str, language: str = "python") -> str:
        """Get explanation for provided code"""
        request = CodeGenerationRequest(
            prompt=code,
            language=language,
            task_type="explanation",
            max_length=512,
            temperature=0.3
        )
        
        response = await self.generate_code(request)
        return response.explanation or "Code explanation not available"
    
    async def refactor_code(self, code: str, language: str = "python") -> str:
        """Refactor provided code"""
        request = CodeGenerationRequest(
            prompt=code,
            language=language,
            task_type="refactoring",
            max_length=1024,
            temperature=0.5
        )
        
        response = await self.generate_code(request)
        return response.generated_code
    
    async def complete_code(self, partial_code: str, language: str = "python") -> str:
        """Complete partial code"""
        request = CodeGenerationRequest(
            prompt=partial_code,
            language=language,
            task_type="completion",
            max_length=512,
            temperature=0.7
        )
        
        response = await self.generate_code(request)
        return response.generated_code
    
    async def generate_from_description(self, description: str, language: str = "python") -> str:
        """Generate code from natural language description"""
        request = CodeGenerationRequest(
            prompt=description,
            language=language,
            task_type="generation",
            max_length=1024,
            temperature=0.6
        )
        
        response = await self.generate_code(request)
        return response.generated_code
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "loaded_models": list(self.models.keys()),
            "available_models": self.model_configs,
            "device": self.device,
            "transformers_available": TRANSFORMERS_AVAILABLE
        }

# Global instance
code_generation_agent = CodeGenerationAgent()

async def main():
    """Test the code generation agent"""
    print("🤖 Testing Quark Code Generation Agent")
    print("=" * 50)
    
    agent = CodeGenerationAgent()
    
    # Test code completion
    test_prompt = "def fibonacci(n):"
    print(f"Input: {test_prompt}")
    
    completion = await agent.complete_code(test_prompt, "python")
    print(f"Completion:\n{completion}")
    
    print("\n" + "=" * 50)
    
    # Test code generation
    description = "Create a function that sorts a list of dictionaries by a given key"
    print(f"Description: {description}")
    
    generated = await agent.generate_from_description(description, "python")
    print(f"Generated:\n{generated}")
    
    print("\n" + "=" * 50)
    print("✅ Code generation agent test completed!")

if __name__ == "__main__":
    asyncio.run(main())