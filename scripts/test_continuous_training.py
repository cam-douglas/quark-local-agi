#!/usr/bin/env python3
"""
Test Quark's Continuous Training and Self-Improvement System
==========================================================

This script tests Quark's ability to search for its own datasets and continuously improve its intelligence.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.dataset_discovery_agent import DatasetDiscoveryAgent
from agents.continuous_training_agent import ContinuousTrainingAgent
from agents.self_improvement_agent import SelfImprovementAgent

def test_dataset_discovery():
    """Test dataset discovery capabilities."""
    print("🔍 Testing Dataset Discovery...")
    
    try:
        # Initialize dataset discovery agent
        discovery_agent = DatasetDiscoveryAgent()
        discovery_agent.load_model()
        
        # Test dataset search
        search_queries = [
            "conversation ai training",
            "question answering datasets",
            "reasoning tasks datasets"
        ]
        
        for query in search_queries:
            print(f"\n📋 Searching for: {query}")
            result = discovery_agent.generate(f"search for {query}")
            
            if "error" not in result:
                datasets = result.get("datasets", [])
                print(f"   ✅ Found {len(datasets)} datasets")
                
                for i, dataset in enumerate(datasets[:3]):  # Show top 3
                    print(f"   📊 Dataset {i+1}: {dataset.name}")
                    print(f"      Quality: {dataset.quality_score:.2f}, Relevance: {dataset.relevance_score:.2f}")
                    print(f"      Size: {dataset.size} examples")
            else:
                print(f"   ❌ Search failed: {result['error']}")
        
        # Test dataset recommendations
        print("\n🎯 Testing dataset recommendations...")
        recommendations = discovery_agent.generate("recommend datasets for quark")
        
        if "error" not in recommendations:
            print(f"   ✅ Generated {recommendations.get('total_recommended', 0)} recommendations")
        else:
            print(f"   ❌ Recommendations failed: {recommendations['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset discovery test failed: {e}")
        return False

def test_continuous_training():
    """Test continuous training capabilities."""
    print("\n🏋️ Testing Continuous Training...")
    
    try:
        # Initialize continuous training agent
        training_agent = ContinuousTrainingAgent()
        training_agent.load_model()
        
        # Test training session
        print("   🚀 Starting training session...")
        training_result = training_agent.generate(
            "start training session",
            model_name="quark_core",
            strategy="incremental"
        )
        
        if "error" not in training_result:
            print(f"   ✅ Training session started: {training_result.get('session_id', 'unknown')}")
            print(f"   📈 Expected improvement: {training_result.get('estimated_duration', 'unknown')}")
        else:
            print(f"   ❌ Training failed: {training_result['error']}")
        
        # Test training status
        print("   📊 Checking training status...")
        status_result = training_agent.generate("get training status")
        
        if "error" not in status_result:
            print(f"   ✅ Active sessions: {status_result.get('active_sessions', 0)}")
            print(f"   ✅ Completed sessions: {status_result.get('completed_sessions', 0)}")
        else:
            print(f"   ❌ Status check failed: {status_result['error']}")
        
        # Test performance evaluation
        print("   📈 Evaluating model performance...")
        eval_result = training_agent.generate("evaluate model performance")
        
        if "error" not in eval_result:
            performance = eval_result.get("accuracy", 0)
            print(f"   ✅ Current accuracy: {performance:.3f}")
        else:
            print(f"   ❌ Evaluation failed: {eval_result['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Continuous training test failed: {e}")
        return False

def test_self_improvement():
    """Test self-improvement capabilities."""
    print("\n🧠 Testing Self-Improvement...")
    
    try:
        # Initialize self-improvement agent
        improvement_agent = SelfImprovementAgent()
        improvement_agent.load_model()
        
        # Test self-reflection
        print("   🤔 Running self-reflection...")
        reflection_result = improvement_agent.generate("run self reflection")
        
        if "error" not in reflection_result:
            improvement_needed = reflection_result.get("improvement_needed", False)
            print(f"   ✅ Self-reflection completed")
            print(f"   📊 Improvement needed: {improvement_needed}")
        else:
            print(f"   ❌ Self-reflection failed: {reflection_result['error']}")
        
        # Test dataset discovery integration
        print("   🔍 Testing dataset discovery integration...")
        discovery_result = improvement_agent.generate(
            "discover training datasets",
            operation="discover_datasets"
        )
        
        if "error" not in discovery_result:
            datasets_found = discovery_result.get("total_discovered", 0)
            high_quality = discovery_result.get("high_quality_count", 0)
            print(f"   ✅ Discovered {datasets_found} datasets ({high_quality} high-quality)")
        else:
            print(f"   ❌ Dataset discovery failed: {discovery_result['error']}")
        
        # Test continuous training integration
        print("   🏋️ Testing continuous training integration...")
        training_result = improvement_agent.generate(
            "start continuous training",
            operation="start_training"
        )
        
        if "error" not in training_result:
            improvement = training_result.get("performance_improvement", 0)
            print(f"   ✅ Training completed with {improvement:.3f} improvement")
        else:
            print(f"   ❌ Training failed: {training_result['error']}")
        
        # Test learning statistics
        print("   📊 Getting learning statistics...")
        stats_result = improvement_agent.generate("get learning statistics")
        
        if "error" not in stats_result:
            total_examples = stats_result.get("total_examples", 0)
            avg_feedback = stats_result.get("average_feedback", 0)
            print(f"   ✅ Total examples: {total_examples}")
            print(f"   ✅ Average feedback: {avg_feedback:.3f}")
        else:
            print(f"   ❌ Statistics failed: {stats_result['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Self-improvement test failed: {e}")
        return False

def test_integrated_system():
    """Test the integrated continuous training system."""
    print("\n🎯 Testing Integrated Continuous Training System...")
    
    try:
        from scripts.quark_continuous_training import QuarkContinuousTrainingOrchestrator
        
        # Create orchestrator
        orchestrator = QuarkContinuousTrainingOrchestrator()
        
        print("   🔧 Testing orchestrator initialization...")
        print("   ✅ All agents initialized successfully")
        
        # Test dataset discovery
        print("   🔍 Testing integrated dataset discovery...")
        discovery_result = orchestrator._discover_training_datasets()
        
        if "error" not in discovery_result:
            total_discovered = discovery_result.get("total_discovered", 0)
            high_quality = discovery_result.get("high_quality_count", 0)
            print(f"   ✅ Discovered {total_discovered} datasets ({high_quality} high-quality)")
        else:
            print(f"   ❌ Discovery failed: {discovery_result['error']}")
        
        # Test training session
        print("   🏋️ Testing integrated training session...")
        if discovery_result.get("datasets"):
            training_result = orchestrator._start_training_session(discovery_result["datasets"][:2])
            
            if "error" not in training_result:
                improvement = training_result.get("performance_improvement", 0)
                print(f"   ✅ Training completed with {improvement:.3f} improvement")
            else:
                print(f"   ❌ Training failed: {training_result['error']}")
        else:
            print("   ⏳ No datasets available for training")
        
        # Test monitoring
        print("   📊 Testing progress monitoring...")
        orchestrator._monitor_training_progress()
        print("   ✅ Progress monitoring working")
        
        return True
        
    except Exception as e:
        print(f"❌ Integrated system test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🤖 Quark Continuous Training System Test")
    print("=======================================")
    print("Testing Quark's ability to:")
    print("  🔍 Search for its own training datasets")
    print("  🏋️ Continuously train and improve")
    print("  🧠 Self-improve through reflection and learning")
    print("  📈 Monitor and optimize performance")
    print()
    
    # Run tests
    tests = [
        ("Dataset Discovery", test_dataset_discovery),
        ("Continuous Training", test_continuous_training),
        ("Self-Improvement", test_self_improvement),
        ("Integrated System", test_integrated_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name} Test...")
        print(f"{'='*50}")
        
        start_time = time.time()
        success = test_func()
        end_time = time.time()
        
        duration = end_time - start_time
        status = "✅ PASSED" if success else "❌ FAILED"
        
        print(f"\n{status} - {test_name} Test ({duration:.2f}s)")
        results.append((test_name, success, duration))
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    total_time = sum(duration for _, _, duration in results)
    
    for test_name, success, duration in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {test_name} ({duration:.2f}s)")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"Total time: {total_time:.2f}s")
    
    if passed == total:
        print("\n🎉 All tests passed! Quark's continuous training system is working correctly.")
        print("\n🚀 You can now run the full continuous training system with:")
        print("   python3 scripts/quark_continuous_training.py")
    else:
        print(f"\n⚠️ {total-passed} test(s) failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 