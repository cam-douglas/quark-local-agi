#!/usr/bin/env python3
"""
Creative Intelligence Agent - Pillar 31
Advanced creative intelligence system
"""

import os
import sys
import time
import logging
import json
import random
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.base import Agent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreativeType(Enum):
    """Types of creative outputs"""
    ARTISTIC = "artistic"
    LITERARY = "literary"
    MUSICAL = "musical"
    TECHNICAL = "technical"
    CONCEPTUAL = "conceptual"
    INNOVATIVE = "innovative"
    COLLABORATIVE = "collaborative"

class CreativeStyle(Enum):
    """Creative styles and approaches"""
    ABSTRACT = "abstract"
    REALISTIC = "realistic"
    SURREAL = "surreal"
    MINIMALIST = "minimalist"
    EXPRESSIVE = "expressive"
    TECHNICAL = "technical"
    EXPERIMENTAL = "experimental"

class CreativeMood(Enum):
    """Creative moods and emotions"""
    JOYFUL = "joyful"
    MELANCHOLIC = "melancholic"
    ENERGETIC = "energetic"
    CALM = "calm"
    MYSTERIOUS = "mysterious"
    DRAMATIC = "dramatic"
    PLAYFUL = "playful"

@dataclass
class CreativeContext:
    """Context for creative generation"""
    theme: str
    style: CreativeStyle
    mood: CreativeMood
    medium: str
    constraints: List[str]
    inspiration: List[str]
    target_audience: str
    complexity_level: str
    collaboration_mode: bool

@dataclass
class CreativeElement:
    """A creative element or component"""
    id: str
    type: str
    content: str
    style: str
    position: Optional[Dict[str, Any]]
    relationships: List[str]
    metadata: Dict[str, Any]

@dataclass
class CreativeWork:
    """A complete creative work"""
    id: str
    title: str
    type: CreativeType
    style: CreativeStyle
    mood: CreativeMood
    context: CreativeContext
    elements: List[CreativeElement]
    composition: Dict[str, Any]
    interpretation: str
    timestamp: datetime
    collaboration_contributors: List[str]
    evolution_history: List[Dict[str, Any]]

@dataclass
class CreativeMetrics:
    """Metrics for creative performance"""
    total_works: int
    successful_works: int
    average_complexity: float
    innovation_score: float
    collaboration_rate: float
    audience_engagement: float
    creative_evolution: float

class CreativeIntelligenceAgent(Agent):
    """
    Advanced creative intelligence agent
    Capable of generating artistic, literary, musical, and innovative content
    """
    
    def __init__(self):
        super().__init__("creative_intelligence")
        self.creative_works: List[CreativeWork] = []
        self.creative_patterns: Dict[str, Any] = {}
        self.collaboration_networks: Dict[str, List[str]] = {}
        self.metrics = CreativeMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.creative_algorithms: Dict[str, Any] = {}
        self.inspiration_database: Dict[str, List[str]] = {}
        self.style_preferences: Dict[str, float] = {}
        
    def load_model(self):
        """Load creative intelligence models"""
        logger.info("Loading creative intelligence models...")
        
        # Initialize creative algorithms
        self.creative_algorithms = {
            "composition": self._compose_creative_work,
            "style_transfer": self._transfer_style,
            "collaboration": self._collaborate_creative,
            "innovation": self._generate_innovations,
            "interpretation": self._interpret_creative_work,
            "evolution": self._evolve_creative_work
        }
        
        # Initialize inspiration database
        self.inspiration_database = {
            "nature": ["mountains", "oceans", "forests", "animals", "weather"],
            "emotions": ["love", "joy", "sadness", "anger", "hope", "fear"],
            "technology": ["artificial intelligence", "robotics", "virtual reality", "space exploration"],
            "culture": ["art", "music", "literature", "philosophy", "history"],
            "abstract": ["patterns", "shapes", "colors", "movement", "energy"]
        }
        
        # Initialize style preferences
        self.style_preferences = {
            "abstract": 0.3,
            "realistic": 0.2,
            "surreal": 0.2,
            "minimalist": 0.1,
            "expressive": 0.2
        }
        
        logger.info("✅ Creative intelligence models loaded successfully")
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate creative content based on the prompt"""
        try:
            # Parse the creative request
            context = self._parse_creative_request(prompt, kwargs)
            
            # Generate creative work
            creative_work = self._generate_creative_work(context)
            
            # Return the creative result
            return self._format_creative_result(creative_work)
            
        except Exception as e:
            logger.error(f"Error in creative generation: {e}")
            return f"Creative generation error: {str(e)}"
    
    def _parse_creative_request(self, prompt: str, kwargs: Dict[str, Any]) -> CreativeContext:
        """Parse a creative request into a structured context"""
        # Extract creative parameters
        theme = self._extract_theme(prompt)
        style = self._determine_style(prompt, kwargs.get('style'))
        mood = self._determine_mood(prompt, kwargs.get('mood'))
        medium = self._determine_medium(prompt, kwargs.get('medium'))
        constraints = self._extract_constraints(prompt)
        inspiration = self._extract_inspiration(prompt)
        target_audience = kwargs.get('audience', 'general')
        complexity_level = kwargs.get('complexity', 'medium')
        collaboration_mode = kwargs.get('collaboration', False)
        
        return CreativeContext(
            theme=theme,
            style=style,
            mood=mood,
            medium=medium,
            constraints=constraints,
            inspiration=inspiration,
            target_audience=target_audience,
            complexity_level=complexity_level,
            collaboration_mode=collaboration_mode
        )
    
    def _extract_theme(self, prompt: str) -> str:
        """Extract the main theme from the prompt"""
        themes = ["nature", "technology", "emotions", "culture", "abstract", "humanity", "space", "time"]
        for theme in themes:
            if theme in prompt.lower():
                return theme
        return "abstract"
    
    def _determine_style(self, prompt: str, style_param: Optional[str] = None) -> CreativeStyle:
        """Determine the creative style"""
        if style_param:
            try:
                return CreativeStyle(style_param)
            except ValueError:
                pass
        
        if "abstract" in prompt.lower():
            return CreativeStyle.ABSTRACT
        elif "realistic" in prompt.lower():
            return CreativeStyle.REALISTIC
        elif "surreal" in prompt.lower():
            return CreativeStyle.SURREAL
        elif "minimalist" in prompt.lower():
            return CreativeStyle.MINIMALIST
        elif "expressive" in prompt.lower():
            return CreativeStyle.EXPRESSIVE
        elif "technical" in prompt.lower():
            return CreativeStyle.TECHNICAL
        else:
            return CreativeStyle.EXPERIMENTAL
    
    def _determine_mood(self, prompt: str, mood_param: Optional[str] = None) -> CreativeMood:
        """Determine the creative mood"""
        if mood_param:
            try:
                return CreativeMood(mood_param)
            except ValueError:
                pass
        
        if any(word in prompt.lower() for word in ["joy", "happy", "bright", "cheerful"]):
            return CreativeMood.JOYFUL
        elif any(word in prompt.lower() for word in ["sad", "melancholy", "melancholic", "dark", "sorrow"]):
            return CreativeMood.MELANCHOLIC
        elif any(word in prompt.lower() for word in ["energy", "energetic", "dynamic", "powerful", "intense"]):
            return CreativeMood.ENERGETIC
        elif any(word in prompt.lower() for word in ["calm", "peaceful", "serene", "tranquil"]):
            return CreativeMood.CALM
        elif any(word in prompt.lower() for word in ["mystery", "mysterious", "enigmatic", "unknown", "hidden"]):
            return CreativeMood.MYSTERIOUS
        elif any(word in prompt.lower() for word in ["drama", "dramatic", "tension", "conflict", "intense"]):
            return CreativeMood.DRAMATIC
        else:
            return CreativeMood.PLAYFUL
    
    def _determine_medium(self, prompt: str, medium_param: Optional[str] = None) -> str:
        """Determine the creative medium"""
        if medium_param:
            return medium_param
        
        if any(word in prompt.lower() for word in ["paint", "art", "visual", "image"]):
            return "visual_art"
        elif any(word in prompt.lower() for word in ["write", "story", "poem", "text"]):
            return "literature"
        elif any(word in prompt.lower() for word in ["music", "sound", "melody", "rhythm"]):
            return "music"
        elif any(word in prompt.lower() for word in ["design", "architecture", "structure"]):
            return "design"
        else:
            return "mixed_media"
    
    def _extract_constraints(self, prompt: str) -> List[str]:
        """Extract creative constraints"""
        constraints = []
        if "simple" in prompt.lower():
            constraints.append("simplicity")
        if "complex" in prompt.lower():
            constraints.append("complexity")
        if "color" in prompt.lower():
            constraints.append("color_specific")
        if "size" in prompt.lower():
            constraints.append("size_constraint")
        return constraints
    
    def _extract_inspiration(self, prompt: str) -> List[str]:
        """Extract inspiration sources"""
        inspiration = []
        for category, sources in self.inspiration_database.items():
            for source in sources:
                if source in prompt.lower():
                    inspiration.append(source)
        return inspiration
    
    def _generate_creative_work(self, context: CreativeContext) -> CreativeWork:
        """Generate a complete creative work"""
        work_id = f"creative_work_{int(time.time())}"
        
        # Determine creative type based on medium
        creative_type = self._determine_creative_type(context.medium)
        
        # Generate title
        title = self._generate_title(context)
        
        # Generate creative elements
        elements = self._generate_creative_elements(context)
        
        # Create composition
        composition = self._create_composition(elements, context)
        
        # Generate interpretation
        interpretation = self._generate_interpretation(context, elements)
        
        # Create creative work
        creative_work = CreativeWork(
            id=work_id,
            title=title,
            type=creative_type,
            style=context.style,
            mood=context.mood,
            context=context,
            elements=elements,
            composition=composition,
            interpretation=interpretation,
            timestamp=datetime.now(),
            collaboration_contributors=[],
            evolution_history=[]
        )
        
        # Store the work
        self.creative_works.append(creative_work)
        self._update_metrics(creative_work)
        
        return creative_work
    
    def _determine_creative_type(self, medium: str) -> CreativeType:
        """Determine the creative type based on medium"""
        if medium == "visual_art":
            return CreativeType.ARTISTIC
        elif medium == "literature":
            return CreativeType.LITERARY
        elif medium == "music":
            return CreativeType.MUSICAL
        elif medium == "design":
            return CreativeType.TECHNICAL
        else:
            return CreativeType.CONCEPTUAL
    
    def _generate_title(self, context: CreativeContext) -> str:
        """Generate a creative title"""
        titles = {
            "nature": [
                "Whispers of the Forest",
                "Ocean's Embrace",
                "Mountain's Majesty",
                "River's Journey"
            ],
            "technology": [
                "Digital Dreams",
                "Circuit Symphony",
                "Virtual Horizons",
                "AI Reflections"
            ],
            "emotions": [
                "Heart's Echo",
                "Soul's Dance",
                "Emotion's Canvas",
                "Feeling's Flow"
            ],
            "culture": [
                "Cultural Tapestry",
                "Heritage's Voice",
                "Tradition's Evolution",
                "Society's Mirror"
            ],
            "abstract": [
                "Abstract Realities",
                "Form and Void",
                "Pattern's Dance",
                "Concept's Flow"
            ]
        }
        
        theme_titles = titles.get(context.theme, titles["abstract"])
        return random.choice(theme_titles)
    
    def _generate_creative_elements(self, context: CreativeContext) -> List[CreativeElement]:
        """Generate creative elements based on context"""
        elements = []
        num_elements = random.randint(3, 8)
        
        for i in range(num_elements):
            element = CreativeElement(
                id=f"element_{i+1}",
                type=self._generate_element_type(context),
                content=self._generate_element_content(context, i),
                style=context.style.value,
                position=self._generate_element_position(i, num_elements),
                relationships=self._generate_element_relationships(i, num_elements),
                metadata=self._generate_element_metadata(context, i)
            )
            elements.append(element)
        
        return elements
    
    def _generate_element_type(self, context: CreativeContext) -> str:
        """Generate element type based on medium"""
        if context.medium == "visual_art":
            return random.choice(["shape", "color", "texture", "line", "form"])
        elif context.medium == "literature":
            return random.choice(["metaphor", "imagery", "rhythm", "theme", "character"])
        elif context.medium == "music":
            return random.choice(["melody", "rhythm", "harmony", "timbre", "dynamics"])
        else:
            return random.choice(["concept", "pattern", "structure", "movement", "energy"])
    
    def _generate_element_content(self, context: CreativeContext, index: int) -> str:
        """Generate element content"""
        content_templates = {
            "shape": ["circle", "triangle", "square", "organic", "geometric"],
            "color": ["vibrant red", "deep blue", "warm gold", "cool green", "mysterious purple"],
            "texture": ["smooth", "rough", "layered", "transparent", "opaque"],
            "line": ["flowing", "angular", "curved", "broken", "continuous"],
            "form": ["solid", "hollow", "twisted", "folded", "extended"]
        }
        
        element_type = self._generate_element_type(context)
        if element_type in content_templates:
            return random.choice(content_templates[element_type])
        else:
            return f"creative_element_{index+1}"
    
    def _generate_element_position(self, index: int, total: int) -> Dict[str, Any]:
        """Generate element position"""
        return {
            "x": random.uniform(0, 100),
            "y": random.uniform(0, 100),
            "z": random.uniform(0, 10),
            "scale": random.uniform(0.5, 2.0),
            "rotation": random.uniform(0, 360)
        }
    
    def _generate_element_relationships(self, index: int, total: int) -> List[str]:
        """Generate element relationships"""
        relationships = []
        for i in range(total):
            if i != index and random.random() < 0.3:
                relationship_types = ["complements", "contrasts", "supports", "enhances"]
                relationships.append(f"element_{i+1}:{random.choice(relationship_types)}")
        return relationships
    
    def _generate_element_metadata(self, context: CreativeContext, index: int) -> Dict[str, Any]:
        """Generate element metadata"""
        return {
            "complexity": random.uniform(0.1, 1.0),
            "importance": random.uniform(0.1, 1.0),
            "mood_contribution": context.mood.value,
            "style_alignment": context.style.value,
            "generation_time": time.time()
        }
    
    def _create_composition(self, elements: List[CreativeElement], context: CreativeContext) -> Dict[str, Any]:
        """Create composition structure"""
        composition = {
            "layout": self._generate_layout(elements),
            "balance": self._calculate_balance(elements),
            "harmony": self._calculate_harmony(elements, context),
            "contrast": self._calculate_contrast(elements),
            "rhythm": self._generate_rhythm(elements),
            "focus_points": self._identify_focus_points(elements)
        }
        return composition
    
    def _generate_layout(self, elements: List[CreativeElement]) -> Dict[str, Any]:
        """Generate layout structure"""
        return {
            "type": random.choice(["grid", "flow", "radial", "asymmetric", "modular"]),
            "density": len(elements) / 10.0,
            "spacing": random.uniform(0.1, 2.0),
            "hierarchy": self._create_hierarchy(elements)
        }
    
    def _calculate_balance(self, elements: List[CreativeElement]) -> float:
        """Calculate compositional balance"""
        if not elements:
            return 0.5
        weights = [elem.metadata.get("importance", 0.5) for elem in elements]
        return sum(weights) / len(weights)
    
    def _calculate_harmony(self, elements: List[CreativeElement], context: CreativeContext) -> float:
        """Calculate compositional harmony"""
        style_consistency = sum(1 for elem in elements if elem.style == context.style.value) / len(elements)
        mood_consistency = sum(1 for elem in elements if elem.metadata.get("mood_contribution") == context.mood.value) / len(elements)
        return (style_consistency + mood_consistency) / 2
    
    def _calculate_contrast(self, elements: List[CreativeElement]) -> float:
        """Calculate compositional contrast"""
        if len(elements) < 2:
            return 0.0
        complexities = [elem.metadata.get("complexity", 0.5) for elem in elements]
        return max(complexities) - min(complexities)
    
    def _generate_rhythm(self, elements: List[CreativeElement]) -> Dict[str, Any]:
        """Generate rhythmic structure"""
        return {
            "pattern": random.choice(["regular", "irregular", "flowing", "staccato", "syncopated"]),
            "tempo": random.uniform(0.5, 2.0),
            "repetition": random.uniform(0.1, 0.8)
        }
    
    def _identify_focus_points(self, elements: List[CreativeElement]) -> List[str]:
        """Identify focus points in composition"""
        important_elements = [elem.id for elem in elements if elem.metadata.get("importance", 0) > 0.7]
        return important_elements[:3]  # Top 3 focus points
    
    def _create_hierarchy(self, elements: List[CreativeElement]) -> Dict[str, int]:
        """Create element hierarchy"""
        hierarchy = {}
        for elem in elements:
            importance = elem.metadata.get("importance", 0.5)
            hierarchy[elem.id] = int(importance * 5)  # 1-5 scale
        return hierarchy
    
    def _generate_interpretation(self, context: CreativeContext, elements: List[CreativeElement]) -> str:
        """Generate interpretation of the creative work"""
        interpretations = {
            "nature": "This work explores the organic rhythms and patterns found in the natural world, capturing the essence of growth, change, and harmony.",
            "technology": "This piece reflects the intersection of human creativity and technological innovation, exploring the boundaries between artificial and organic intelligence.",
            "emotions": "This creation delves into the depths of human emotion, translating feelings into visual, auditory, or conceptual forms that resonate with universal experiences.",
            "culture": "This work celebrates the rich tapestry of human culture, weaving together traditions, innovations, and shared experiences into a unified expression.",
            "abstract": "This piece transcends literal representation, exploring pure form, color, and movement to create an experience that speaks to the subconscious mind."
        }
        
        base_interpretation = interpretations.get(context.theme, interpretations["abstract"])
        
        # Add style-specific interpretation
        style_interpretations = {
            CreativeStyle.ABSTRACT: " The abstract approach allows for multiple interpretations and emotional responses.",
            CreativeStyle.REALISTIC: " The realistic style grounds the work in recognizable forms while maintaining artistic integrity.",
            CreativeStyle.SURREAL: " The surreal elements create a dreamlike quality that challenges conventional perception.",
            CreativeStyle.MINIMALIST: " The minimalist approach emphasizes essential elements, creating powerful impact through simplicity.",
            CreativeStyle.EXPRESSIVE: " The expressive style conveys raw emotion and personal vision through bold, dynamic elements."
        }
        
        style_interpretation = style_interpretations.get(context.style, "")
        
        return base_interpretation + style_interpretation
    
    def _format_creative_result(self, creative_work: CreativeWork) -> str:
        """Format the creative work result"""
        result = f"""
🎨 **Creative Work Generated**

**Title:** {creative_work.title}
**Type:** {creative_work.type.value}
**Style:** {creative_work.style.value}
**Mood:** {creative_work.mood.value}
**Medium:** {creative_work.context.medium}

**Composition:**
• Layout: {creative_work.composition['layout']['type']}
• Balance: {creative_work.composition['balance']:.2f}
• Harmony: {creative_work.composition['harmony']:.2f}
• Contrast: {creative_work.composition['contrast']:.2f}
• Rhythm: {creative_work.composition['rhythm']['pattern']}

**Elements:** {len(creative_work.elements)} creative elements
**Focus Points:** {', '.join(creative_work.composition['focus_points'])}

**Interpretation:**
{creative_work.interpretation}

**Context:**
• Theme: {creative_work.context.theme}
• Constraints: {', '.join(creative_work.context.constraints) if creative_work.context.constraints else 'None'}
• Inspiration: {', '.join(creative_work.context.inspiration) if creative_work.context.inspiration else 'None'}
• Target Audience: {creative_work.context.target_audience}
• Complexity: {creative_work.context.complexity_level}

**Status:** Created
**Timestamp:** {creative_work.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        """
        return result.strip()
    
    def _update_metrics(self, creative_work: CreativeWork):
        """Update creative metrics"""
        self.metrics.total_works += 1
        
        # Calculate complexity
        complexities = [elem.metadata.get("complexity", 0.5) for elem in creative_work.elements]
        avg_complexity = sum(complexities) / len(complexities) if complexities else 0.5
        self.metrics.average_complexity = (self.metrics.average_complexity * (self.metrics.total_works - 1) + avg_complexity) / self.metrics.total_works
        
        # Calculate innovation score
        innovation_score = self._calculate_innovation_score(creative_work)
        self.metrics.innovation_score = (self.metrics.innovation_score * (self.metrics.total_works - 1) + innovation_score) / self.metrics.total_works
        
        # Update collaboration rate
        if creative_work.context.collaboration_mode:
            self.metrics.collaboration_rate = (self.metrics.collaboration_rate * (self.metrics.total_works - 1) + 1) / self.metrics.total_works
    
    def _calculate_innovation_score(self, creative_work: CreativeWork) -> float:
        """Calculate innovation score for a creative work"""
        # Factors: style uniqueness, element diversity, composition complexity
        style_uniqueness = 0.8 if creative_work.style == CreativeStyle.EXPERIMENTAL else 0.5
        element_diversity = len(set(elem.type for elem in creative_work.elements)) / len(creative_work.elements)
        composition_complexity = creative_work.composition['contrast'] * creative_work.composition['harmony']
        
        return (style_uniqueness + element_diversity + composition_complexity) / 3
    
    def get_creative_history(self) -> List[Dict[str, Any]]:
        """Get creative work history"""
        return [asdict(work) for work in self.creative_works]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get creative metrics"""
        return asdict(self.metrics)
    
    def get_recent_works(self, limit: int = 5) -> List[CreativeWork]:
        """Get recent creative works"""
        return self.creative_works[-limit:] if self.creative_works else []
    
    def analyze_creative_patterns(self) -> Dict[str, Any]:
        """Analyze creative patterns"""
        if not self.creative_works:
            return {}
        
        patterns = {
            "style_distribution": {},
            "mood_distribution": {},
            "type_distribution": {},
            "complexity_trends": [],
            "innovation_trends": [],
            "collaboration_trends": []
        }
        
        # Analyze distributions
        for work in self.creative_works:
            style = work.style.value
            mood = work.mood.value
            work_type = work.type.value
            
            patterns["style_distribution"][style] = patterns["style_distribution"].get(style, 0) + 1
            patterns["mood_distribution"][mood] = patterns["mood_distribution"].get(mood, 0) + 1
            patterns["type_distribution"][work_type] = patterns["type_distribution"].get(work_type, 0) + 1
        
        # Analyze trends
        complexities = [work.composition['balance'] for work in self.creative_works]
        patterns["complexity_trends"] = complexities
        
        return patterns
    
    def _compose_creative_work(self, elements, context):
        """Compose creative work from elements"""
        # Placeholder for composition algorithm
        return {"composition": "balanced", "harmony": 0.8}
    
    def _transfer_style(self, source_style, target_work):
        """Transfer style from source to target"""
        # Placeholder for style transfer
        return {"transferred_style": source_style, "confidence": 0.7}
    
    def _collaborate_creative(self, contributors, context):
        """Collaborate with multiple contributors"""
        # Placeholder for collaboration
        return {"collaboration_result": "merged", "contributors": contributors}
    
    def _generate_innovations(self, base_work, constraints):
        """Generate innovative variations"""
        # Placeholder for innovation generation
        return {"innovations": ["variation_1", "variation_2"], "novelty_score": 0.8}
    
    def _interpret_creative_work(self, work, context):
        """Interpret creative work"""
        # Placeholder for interpretation
        return {"interpretation": "artistic_expression", "depth": 0.9}
    
    def _evolve_creative_work(self, work, evolution_params):
        """Evolve creative work over time"""
        # Placeholder for evolution
        return {"evolved_work": work, "evolution_stage": "advanced"} 