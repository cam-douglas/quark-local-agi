#!/usr/bin/env python3
"""
Test Suite for Pillar 31: Advanced Creative Intelligence
Tests the CreativeIntelligenceAgent functionality
"""

import os
import sys
import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.creative_intelligence_agent import (
    CreativeIntelligenceAgent,
    CreativeType,
    CreativeStyle,
    CreativeMood,
    CreativeContext,
    CreativeElement,
    CreativeWork
)

@pytest.fixture
def creative_intelligence_agent():
    """Create a test instance of CreativeIntelligenceAgent"""
    agent = CreativeIntelligenceAgent()
    agent.load_model()
    return agent

@pytest.fixture
def sample_creative_context():
    """Create a sample creative context"""
    return CreativeContext(
        theme="nature",
        style=CreativeStyle.ABSTRACT,
        mood=CreativeMood.CALM,
        medium="visual_art",
        constraints=["simplicity"],
        inspiration=["mountains", "forests"],
        target_audience="general",
        complexity_level="medium",
        collaboration_mode=False
    )

@pytest.fixture
def sample_creative_element():
    """Create a sample creative element"""
    return CreativeElement(
        id="element_1",
        type="shape",
        content="circle",
        style="abstract",
        position={"x": 50, "y": 50, "z": 0, "scale": 1.0, "rotation": 0},
        relationships=["element_2:complements"],
        metadata={"complexity": 0.5, "importance": 0.8, "mood_contribution": "calm", "style_alignment": "abstract"}
    )

class TestCreativeIntelligenceAgent:
    """Test cases for CreativeIntelligenceAgent"""
    
    def test_agent_initialization(self, creative_intelligence_agent):
        """Test agent initialization"""
        assert creative_intelligence_agent is not None
        assert creative_intelligence_agent.creative_works == []
        assert len(creative_intelligence_agent.creative_algorithms) == 6
        assert len(creative_intelligence_agent.inspiration_database) == 5
        assert len(creative_intelligence_agent.style_preferences) == 5
    
    def test_load_model(self, creative_intelligence_agent):
        """Test model loading"""
        assert "composition" in creative_intelligence_agent.creative_algorithms
        assert "style_transfer" in creative_intelligence_agent.creative_algorithms
        assert "collaboration" in creative_intelligence_agent.creative_algorithms
        assert "innovation" in creative_intelligence_agent.creative_algorithms
        assert "interpretation" in creative_intelligence_agent.creative_algorithms
        assert "evolution" in creative_intelligence_agent.creative_algorithms
        
        # Check inspiration database
        assert "nature" in creative_intelligence_agent.inspiration_database
        assert "emotions" in creative_intelligence_agent.inspiration_database
        assert "technology" in creative_intelligence_agent.inspiration_database
        
        # Check style preferences
        assert "abstract" in creative_intelligence_agent.style_preferences
        assert "realistic" in creative_intelligence_agent.style_preferences
    
    def test_parse_creative_request(self, creative_intelligence_agent):
        """Test parsing creative requests"""
        prompt = "Create an abstract painting inspired by nature with calm mood"
        kwargs = {"audience": "artists", "complexity": "high"}
        context = creative_intelligence_agent._parse_creative_request(prompt, kwargs)
        
        assert context.theme == "nature"
        assert context.style == CreativeStyle.ABSTRACT
        assert context.mood == CreativeMood.CALM
        assert context.medium == "visual_art"
        assert context.target_audience == "artists"
        assert context.complexity_level == "high"
    
    def test_extract_theme(self, creative_intelligence_agent):
        """Test theme extraction"""
        themes = ["nature", "technology", "emotions", "culture", "abstract"]
        for theme in themes:
            prompt = f"Create something about {theme}"
            extracted_theme = creative_intelligence_agent._extract_theme(prompt)
            assert extracted_theme == theme
        
        # Test default theme
        prompt = "Create something completely random"
        extracted_theme = creative_intelligence_agent._extract_theme(prompt)
        assert extracted_theme == "abstract"
    
    def test_determine_style(self, creative_intelligence_agent):
        """Test style determination"""
        styles = [
            ("abstract", CreativeStyle.ABSTRACT),
            ("realistic", CreativeStyle.REALISTIC),
            ("surreal", CreativeStyle.SURREAL),
            ("minimalist", CreativeStyle.MINIMALIST),
            ("expressive", CreativeStyle.EXPRESSIVE),
            ("technical", CreativeStyle.TECHNICAL)
        ]
        
        for style_name, expected_style in styles:
            prompt = f"Create something {style_name}"
            determined_style = creative_intelligence_agent._determine_style(prompt)
            assert determined_style == expected_style
        
        # Test experimental default
        prompt = "Create something completely new"
        determined_style = creative_intelligence_agent._determine_style(prompt)
        assert determined_style == CreativeStyle.EXPERIMENTAL
    
    def test_determine_mood(self, creative_intelligence_agent):
        """Test mood determination"""
        moods = [
            ("joyful", CreativeMood.JOYFUL),
            ("melancholic", CreativeMood.MELANCHOLIC),
            ("energetic", CreativeMood.ENERGETIC),
            ("calm", CreativeMood.CALM),
            ("mysterious", CreativeMood.MYSTERIOUS),
            ("dramatic", CreativeMood.DRAMATIC)
        ]
        
        for mood_name, expected_mood in moods:
            prompt = f"Create something {mood_name}"
            determined_mood = creative_intelligence_agent._determine_mood(prompt)
            assert determined_mood == expected_mood
        
        # Test playful default
        prompt = "Create something fun"
        determined_mood = creative_intelligence_agent._determine_mood(prompt)
        assert determined_mood == CreativeMood.PLAYFUL
    
    def test_determine_medium(self, creative_intelligence_agent):
        """Test medium determination"""
        mediums = [
            ("paint", "visual_art"),
            ("write", "literature"),
            ("music", "music"),
            ("design", "design")
        ]
        
        for medium_name, expected_medium in mediums:
            prompt = f"Create something with {medium_name}"
            determined_medium = creative_intelligence_agent._determine_medium(prompt)
            assert determined_medium == expected_medium
        
        # Test mixed_media default
        prompt = "Create something creative"
        determined_medium = creative_intelligence_agent._determine_medium(prompt)
        assert determined_medium == "mixed_media"
    
    def test_extract_constraints(self, creative_intelligence_agent):
        """Test constraint extraction"""
        prompt = "Create something simple with specific colors and size constraints"
        constraints = creative_intelligence_agent._extract_constraints(prompt)
        
        assert "simplicity" in constraints
        assert "color_specific" in constraints
        assert "size_constraint" in constraints
    
    def test_extract_inspiration(self, creative_intelligence_agent):
        """Test inspiration extraction"""
        prompt = "Create something inspired by mountains and forests"
        inspiration = creative_intelligence_agent._extract_inspiration(prompt)
        
        assert "mountains" in inspiration
        assert "forests" in inspiration
    
    def test_generate_creative_work(self, creative_intelligence_agent, sample_creative_context):
        """Test creative work generation"""
        creative_work = creative_intelligence_agent._generate_creative_work(sample_creative_context)
        
        assert creative_work is not None
        assert creative_work.id.startswith("creative_work_")
        assert creative_work.title is not None
        assert creative_work.type == CreativeType.ARTISTIC
        assert creative_work.style == sample_creative_context.style
        assert creative_work.mood == sample_creative_context.mood
        assert len(creative_work.elements) > 0
        assert creative_work.composition is not None
        assert creative_work.interpretation is not None
    
    def test_determine_creative_type(self, creative_intelligence_agent):
        """Test creative type determination"""
        mediums = [
            ("visual_art", CreativeType.ARTISTIC),
            ("literature", CreativeType.LITERARY),
            ("music", CreativeType.MUSICAL),
            ("design", CreativeType.TECHNICAL),
            ("mixed_media", CreativeType.CONCEPTUAL)
        ]
        
        for medium, expected_type in mediums:
            creative_type = creative_intelligence_agent._determine_creative_type(medium)
            assert creative_type == expected_type
    
    def test_generate_title(self, creative_intelligence_agent, sample_creative_context):
        """Test title generation"""
        title = creative_intelligence_agent._generate_title(sample_creative_context)
        assert title is not None
        assert len(title) > 0
        
        # Test different themes
        themes = ["nature", "technology", "emotions", "culture", "abstract"]
        for theme in themes:
            context = CreativeContext(
                theme=theme,
                style=CreativeStyle.ABSTRACT,
                mood=CreativeMood.CALM,
                medium="visual_art",
                constraints=[],
                inspiration=[],
                target_audience="general",
                complexity_level="medium",
                collaboration_mode=False
            )
            title = creative_intelligence_agent._generate_title(context)
            assert title is not None
    
    def test_generate_creative_elements(self, creative_intelligence_agent, sample_creative_context):
        """Test creative element generation"""
        elements = creative_intelligence_agent._generate_creative_elements(sample_creative_context)
        
        assert len(elements) >= 3
        assert len(elements) <= 8
        
        for element in elements:
            assert element.id.startswith("element_")
            assert element.type is not None
            assert element.content is not None
            assert element.style == sample_creative_context.style.value
            assert element.position is not None
            assert element.metadata is not None
    
    def test_generate_element_type(self, creative_intelligence_agent):
        """Test element type generation"""
        mediums = ["visual_art", "literature", "music", "design", "mixed_media"]
        for medium in mediums:
            context = CreativeContext(
                theme="abstract",
                style=CreativeStyle.ABSTRACT,
                mood=CreativeMood.CALM,
                medium=medium,
                constraints=[],
                inspiration=[],
                target_audience="general",
                complexity_level="medium",
                collaboration_mode=False
            )
            element_type = creative_intelligence_agent._generate_element_type(context)
            assert element_type is not None
    
    def test_generate_element_content(self, creative_intelligence_agent, sample_creative_context):
        """Test element content generation"""
        for i in range(5):
            content = creative_intelligence_agent._generate_element_content(sample_creative_context, i)
            assert content is not None
            assert len(content) > 0
    
    def test_generate_element_position(self, creative_intelligence_agent):
        """Test element position generation"""
        for i in range(5):
            position = creative_intelligence_agent._generate_element_position(i, 5)
            assert "x" in position
            assert "y" in position
            assert "z" in position
            assert "scale" in position
            assert "rotation" in position
            assert 0 <= position["x"] <= 100
            assert 0 <= position["y"] <= 100
    
    def test_generate_element_relationships(self, creative_intelligence_agent):
        """Test element relationship generation"""
        relationships = creative_intelligence_agent._generate_element_relationships(0, 5)
        assert isinstance(relationships, list)
        
        for relationship in relationships:
            assert ":" in relationship
            parts = relationship.split(":")
            assert len(parts) == 2
    
    def test_generate_element_metadata(self, creative_intelligence_agent, sample_creative_context):
        """Test element metadata generation"""
        for i in range(5):
            metadata = creative_intelligence_agent._generate_element_metadata(sample_creative_context, i)
            assert "complexity" in metadata
            assert "importance" in metadata
            assert "mood_contribution" in metadata
            assert "style_alignment" in metadata
            assert "generation_time" in metadata
    
    def test_create_composition(self, creative_intelligence_agent, sample_creative_context):
        """Test composition creation"""
        elements = creative_intelligence_agent._generate_creative_elements(sample_creative_context)
        composition = creative_intelligence_agent._create_composition(elements, sample_creative_context)
        
        assert "layout" in composition
        assert "balance" in composition
        assert "harmony" in composition
        assert "contrast" in composition
        assert "rhythm" in composition
        assert "focus_points" in composition
    
    def test_generate_layout(self, creative_intelligence_agent):
        """Test layout generation"""
        elements = [MagicMock() for _ in range(5)]
        layout = creative_intelligence_agent._generate_layout(elements)
        
        assert "type" in layout
        assert "density" in layout
        assert "spacing" in layout
        assert "hierarchy" in layout
    
    def test_calculate_balance(self, creative_intelligence_agent):
        """Test balance calculation"""
        # Test with elements
        elements = [MagicMock()]
        elements[0].metadata = {"importance": 0.8}
        balance = creative_intelligence_agent._calculate_balance(elements)
        assert 0 <= balance <= 1
        
        # Test with no elements
        balance = creative_intelligence_agent._calculate_balance([])
        assert balance == 0.5
    
    def test_calculate_harmony(self, creative_intelligence_agent, sample_creative_context):
        """Test harmony calculation"""
        elements = [MagicMock() for _ in range(3)]
        for elem in elements:
            elem.style = sample_creative_context.style.value
            elem.metadata = {"mood_contribution": sample_creative_context.mood.value}
        
        harmony = creative_intelligence_agent._calculate_harmony(elements, sample_creative_context)
        assert 0 <= harmony <= 1
    
    def test_calculate_contrast(self, creative_intelligence_agent):
        """Test contrast calculation"""
        elements = [MagicMock() for _ in range(3)]
        for i, elem in enumerate(elements):
            elem.metadata = {"complexity": 0.2 + i * 0.3}
        
        contrast = creative_intelligence_agent._calculate_contrast(elements)
        assert contrast >= 0
    
    def test_generate_rhythm(self, creative_intelligence_agent):
        """Test rhythm generation"""
        elements = [MagicMock() for _ in range(3)]
        rhythm = creative_intelligence_agent._generate_rhythm(elements)
        
        assert "pattern" in rhythm
        assert "tempo" in rhythm
        assert "repetition" in rhythm
    
    def test_identify_focus_points(self, creative_intelligence_agent):
        """Test focus point identification"""
        elements = [MagicMock() for _ in range(5)]
        for i, elem in enumerate(elements):
            elem.id = f"element_{i+1}"
            elem.metadata = {"importance": 0.5 + i * 0.1}
        
        focus_points = creative_intelligence_agent._identify_focus_points(elements)
        assert isinstance(focus_points, list)
        assert len(focus_points) <= 3
    
    def test_create_hierarchy(self, creative_intelligence_agent):
        """Test hierarchy creation"""
        elements = [MagicMock() for _ in range(3)]
        for i, elem in enumerate(elements):
            elem.id = f"element_{i+1}"
            elem.metadata = {"importance": 0.5 + i * 0.2}
        
        hierarchy = creative_intelligence_agent._create_hierarchy(elements)
        assert isinstance(hierarchy, dict)
        for elem_id, level in hierarchy.items():
            assert 1 <= level <= 5
    
    def test_generate_interpretation(self, creative_intelligence_agent, sample_creative_context):
        """Test interpretation generation"""
        elements = creative_intelligence_agent._generate_creative_elements(sample_creative_context)
        interpretation = creative_intelligence_agent._generate_interpretation(sample_creative_context, elements)
        
        assert interpretation is not None
        assert len(interpretation) > 0
    
    def test_format_creative_result(self, creative_intelligence_agent, sample_creative_context):
        """Test creative result formatting"""
        creative_work = creative_intelligence_agent._generate_creative_work(sample_creative_context)
        result = creative_intelligence_agent._format_creative_result(creative_work)
        
        assert "Creative Work Generated" in result
        assert creative_work.title in result
        assert creative_work.type.value in result
        assert creative_work.style.value in result
        assert creative_work.mood.value in result
    
    def test_update_metrics(self, creative_intelligence_agent, sample_creative_context):
        """Test metrics update"""
        initial_total = creative_intelligence_agent.metrics.total_works
        creative_work = creative_intelligence_agent._generate_creative_work(sample_creative_context)
        
        assert creative_intelligence_agent.metrics.total_works == initial_total + 1
        assert creative_intelligence_agent.metrics.average_complexity > 0
    
    def test_calculate_innovation_score(self, creative_intelligence_agent, sample_creative_context):
        """Test innovation score calculation"""
        creative_work = creative_intelligence_agent._generate_creative_work(sample_creative_context)
        innovation_score = creative_intelligence_agent._calculate_innovation_score(creative_work)
        
        assert 0 <= innovation_score <= 1
    
    def test_get_creative_history(self, creative_intelligence_agent, sample_creative_context):
        """Test creative history retrieval"""
        creative_work = creative_intelligence_agent._generate_creative_work(sample_creative_context)
        history = creative_intelligence_agent.get_creative_history()
        
        assert len(history) > 0
        assert history[0]["id"] == creative_work.id
    
    def test_get_metrics(self, creative_intelligence_agent, sample_creative_context):
        """Test metrics retrieval"""
        creative_intelligence_agent._generate_creative_work(sample_creative_context)
        metrics = creative_intelligence_agent.get_metrics()
        
        assert "total_works" in metrics
        assert "successful_works" in metrics
        assert "average_complexity" in metrics
        assert "innovation_score" in metrics
        assert "collaboration_rate" in metrics
        assert "audience_engagement" in metrics
        assert "creative_evolution" in metrics
    
    def test_get_recent_works(self, creative_intelligence_agent, sample_creative_context):
        """Test recent works retrieval"""
        for i in range(10):
            creative_intelligence_agent._generate_creative_work(sample_creative_context)
        
        recent_works = creative_intelligence_agent.get_recent_works(limit=5)
        assert len(recent_works) == 5
    
    def test_analyze_creative_patterns(self, creative_intelligence_agent, sample_creative_context):
        """Test creative pattern analysis"""
        # Generate some works first
        for i in range(5):
            creative_intelligence_agent._generate_creative_work(sample_creative_context)
        
        patterns = creative_intelligence_agent.analyze_creative_patterns()
        
        assert "style_distribution" in patterns
        assert "mood_distribution" in patterns
        assert "type_distribution" in patterns
        assert "complexity_trends" in patterns
        assert "innovation_trends" in patterns
        assert "collaboration_trends" in patterns
    
    def test_generate_method(self, creative_intelligence_agent):
        """Test the main generate method"""
        prompt = "Create an abstract painting inspired by nature"
        result = creative_intelligence_agent.generate(prompt)
        
        assert result is not None
        assert len(result) > 0
        assert "Creative Work Generated" in result
    
    def test_generate_method_error_handling(self, creative_intelligence_agent):
        """Test error handling in generate method"""
        with patch.object(creative_intelligence_agent, '_parse_creative_request', side_effect=Exception("Test error")):
            result = creative_intelligence_agent.generate("test prompt")
            assert "Creative generation error" in result
    
    def test_creative_work_storage(self, creative_intelligence_agent, sample_creative_context):
        """Test that creative works are properly stored"""
        initial_count = len(creative_intelligence_agent.creative_works)
        creative_work = creative_intelligence_agent._generate_creative_work(sample_creative_context)
        
        assert len(creative_intelligence_agent.creative_works) == initial_count + 1
        assert creative_intelligence_agent.creative_works[-1].id == creative_work.id
    
    def test_collaboration_mode(self, creative_intelligence_agent):
        """Test collaboration mode functionality"""
        context = CreativeContext(
            theme="nature",
            style=CreativeStyle.ABSTRACT,
            mood=CreativeMood.CALM,
            medium="visual_art",
            constraints=[],
            inspiration=[],
            target_audience="general",
            complexity_level="medium",
            collaboration_mode=True
        )
        
        creative_work = creative_intelligence_agent._generate_creative_work(context)
        assert creative_work.context.collaboration_mode == True
    
    def test_different_creative_types(self, creative_intelligence_agent):
        """Test different creative types"""
        mediums = ["visual_art", "literature", "music", "design", "mixed_media"]
        
        for medium in mediums:
            context = CreativeContext(
                theme="abstract",
                style=CreativeStyle.ABSTRACT,
                mood=CreativeMood.CALM,
                medium=medium,
                constraints=[],
                inspiration=[],
                target_audience="general",
                complexity_level="medium",
                collaboration_mode=False
            )
            
            creative_work = creative_intelligence_agent._generate_creative_work(context)
            assert creative_work.type in CreativeType

if __name__ == "__main__":
    pytest.main([__file__]) 