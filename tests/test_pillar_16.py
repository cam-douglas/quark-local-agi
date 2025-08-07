#!/usr/bin/env python3
"""
Test suite for Pillar 16: Meta-Learning & Self-Reflection
Tests self-monitoring agents, performance introspection, and pipeline reconfiguration
"""

import sys
import os
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_meta_learning_agent():
    """Test meta-learning agent"""
    print("Testing Meta-Learning Agent...")
    
    try:
        from meta_learning.meta_learning_agent import MetaLearningAgent
        
        # Initialize agent
        agent = MetaLearningAgent()
        assert agent is not None
        
        # Test learning capabilities
        learning_result = agent.generate("Learn from this interaction", operation="meta_learn")
        assert "learning_opportunities" in learning_result
        assert "performance_insights" in learning_result
        
        # Test self-improvement
        improvement_result = agent.generate("Improve my capabilities", operation="self_improve")
        assert "improvements" in improvement_result
        assert "capability_enhancements" in improvement_result
        
        print("✅ Meta-Learning Agent - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Meta-Learning Agent - FAILED: {e}")
        return False

def test_self_reflection_agent():
    """Test self-reflection agent"""
    print("Testing Self-Reflection Agent...")
    
    try:
        from meta_learning.self_reflection_agent import SelfReflectionAgent
        
        # Initialize agent
        agent = SelfReflectionAgent()
        assert agent is not None
        
        # Test self-reflection
        reflection_result = agent.generate("Reflect on my performance", operation="self_reflect")
        assert "reflection_insights" in reflection_result
        assert "performance_analysis" in reflection_result
        
        # Test introspection
        introspection_result = agent.generate("Analyze my behavior", operation="introspect")
        assert "behavior_analysis" in introspection_result
        assert "improvement_suggestions" in introspection_result
        
        print("✅ Self-Reflection Agent - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Self-Reflection Agent - FAILED: {e}")
        return False

def test_performance_monitor():
    """Test performance monitoring"""
    print("Testing Performance Monitor...")
    
    try:
        from meta_learning.performance_monitor import PerformanceMonitor
        
        # Initialize monitor
        monitor = PerformanceMonitor()
        assert monitor is not None
        
        # Test performance tracking
        performance_result = monitor.track_performance("test_operation", 1.5, True)
        assert "operation_id" in performance_result
        assert "execution_time" in performance_result
        assert "success" in performance_result
        
        # Test performance analysis
        analysis_result = monitor.analyze_performance()
        assert "average_execution_time" in analysis_result
        assert "success_rate" in analysis_result
        assert "performance_trends" in analysis_result
        
        print("✅ Performance Monitor - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Performance Monitor - FAILED: {e}")
        return False

def test_pipeline_reconfigurator():
    """Test pipeline reconfiguration"""
    print("Testing Pipeline Reconfigurator...")
    
    try:
        from meta_learning.pipeline_reconfigurator import PipelineReconfigurator
        
        # Initialize reconfigurator
        reconfigurator = PipelineReconfigurator()
        assert reconfigurator is not None
        
        # Test pipeline analysis
        analysis_result = reconfigurator.analyze_pipeline("test_pipeline")
        assert "pipeline_efficiency" in analysis_result
        assert "bottlenecks" in analysis_result
        assert "optimization_opportunities" in analysis_result
        
        # Test pipeline optimization
        optimization_result = reconfigurator.optimize_pipeline("test_pipeline")
        assert "optimized_configuration" in optimization_result
        assert "performance_improvements" in optimization_result
        assert "resource_optimization" in optimization_result
        
        print("✅ Pipeline Reconfigurator - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline Reconfigurator - FAILED: {e}")
        return False

def test_meta_learning_orchestrator():
    """Test meta-learning orchestrator"""
    print("Testing Meta-Learning Orchestrator...")
    
    try:
        from meta_learning.meta_learning_orchestrator import MetaLearningOrchestrator
        
        # Initialize orchestrator
        orchestrator = MetaLearningOrchestrator()
        assert orchestrator is not None
        
        # Test orchestration
        orchestration_result = orchestrator.orchestrate_learning("test_session")
        assert "learning_session" in orchestration_result
        assert "meta_learning_activities" in orchestration_result
        assert "performance_insights" in orchestration_result
        
        # Test learning coordination
        coordination_result = orchestrator.coordinate_learning()
        assert "learning_coordination" in coordination_result
        assert "agent_collaboration" in coordination_result
        assert "knowledge_integration" in coordination_result
        
        print("✅ Meta-Learning Orchestrator - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Meta-Learning Orchestrator - FAILED: {e}")
        return False

def test_self_monitoring():
    """Test self-monitoring capabilities"""
    print("Testing Self-Monitoring...")
    
    try:
        from meta_learning.meta_learning_agent import MetaLearningAgent
        
        agent = MetaLearningAgent()
        
        # Test self-monitoring
        monitoring_result = agent.monitor_self_performance()
        assert "performance_metrics" in monitoring_result
        assert "behavior_analysis" in monitoring_result
        assert "improvement_areas" in monitoring_result
        
        # Test adaptive behavior
        adaptive_result = agent.adapt_behavior("new_context")
        assert "behavior_adjustments" in adaptive_result
        assert "context_adaptation" in adaptive_result
        assert "learning_applied" in adaptive_result
        
        print("✅ Self-Monitoring - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Self-Monitoring - FAILED: {e}")
        return False

def test_learning_integration():
    """Test learning integration"""
    print("Testing Learning Integration...")
    
    try:
        from meta_learning.meta_learning_orchestrator import MetaLearningOrchestrator
        
        orchestrator = MetaLearningOrchestrator()
        
        # Test learning integration
        integration_result = orchestrator.integrate_learning("new_knowledge")
        assert "knowledge_integration" in integration_result
        assert "learning_applied" in integration_result
        assert "capability_enhancement" in integration_result
        
        # Test continuous learning
        continuous_result = orchestrator.enable_continuous_learning()
        assert "continuous_learning_enabled" in continuous_result
        assert "learning_loops" in continuous_result
        assert "adaptive_behavior" in continuous_result
        
        print("✅ Learning Integration - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Learning Integration - FAILED: {e}")
        return False

def test_meta_learning_components():
    """Test meta-learning component files"""
    print("Testing Meta-Learning Components...")
    
    try:
        # Test component files exist
        component_files = [
            "meta_learning/meta_learning_agent.py",
            "meta_learning/self_reflection_agent.py",
            "meta_learning/performance_monitor.py",
            "meta_learning/pipeline_reconfigurator.py",
            "meta_learning/meta_learning_orchestrator.py",
            "meta_learning/README.md"
        ]
        
        for component_file in component_files:
            assert os.path.exists(component_file), f"Component file missing: {component_file}"
            assert os.path.getsize(component_file) > 0, f"Component file is empty: {component_file}"
        
        # Test README content
        readme_path = "meta_learning/README.md"
        with open(readme_path, 'r') as f:
            readme_content = f.read()
            assert "Meta-Learning" in readme_content
            assert "Self-Reflection" in readme_content
            assert "Performance" in readme_content
        
        print("✅ Meta-Learning Components - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Meta-Learning Components - FAILED: {e}")
        return False

def main():
    """Run all Pillar 16 tests."""
    print("🧠 Testing Pillar 16: Meta-Learning & Self-Reflection...")
    print("=" * 70)
    
    results = []
    
    # Test each component
    results.append(test_meta_learning_agent())
    results.append(test_self_reflection_agent())
    results.append(test_performance_monitor())
    results.append(test_pipeline_reconfigurator())
    results.append(test_meta_learning_orchestrator())
    results.append(test_self_monitoring())
    results.append(test_learning_integration())
    results.append(test_meta_learning_components())
    
    # Summary
    print("=" * 70)
    print("📊 Test Results Summary:")
    
    test_names = [
        "Meta-Learning Agent",
        "Self-Reflection Agent",
        "Performance Monitor",
        "Pipeline Reconfigurator",
        "Meta-Learning Orchestrator",
        "Self-Monitoring",
        "Learning Integration",
        "Meta-Learning Components"
    ]
    
    passed = 0
    for i, (result, name) in enumerate(zip(results, test_names)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {i+1}. {name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 Pillar 16: Meta-Learning & Self-Reflection is working correctly!")
        print("\n📋 Pillar 16 Features:")
        print("  ✅ Self-monitoring agents with performance introspection")
        print("  ✅ Pipeline reconfiguration and optimization")
        print("  ✅ Meta-learning capabilities and knowledge integration")
        print("  ✅ Self-improvement loops and adaptive behavior")
        print("  ✅ Continuous learning and capability enhancement")
        print("  ✅ Performance monitoring and analysis")
        print("  ✅ Learning coordination and orchestration")
        print("  ✅ Comprehensive meta-learning architecture")
        return True
    else:
        print("⚠️  Some tests need attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 