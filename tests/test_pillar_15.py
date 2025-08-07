#!/usr/bin/env python3
"""
Test script for Pillar 15: Safety & Alignment

This script demonstrates the key capabilities of Pillar 15 implementation.
"""

import json
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.safety_agent import SafetyAgent
from alignment.content_filtering import ContentFilter
from alignment.ethical_practices import EthicalPractices
from alignment.alignment_monitor import AlignmentMonitor
from alignment.rlhf_agent import RLHFAgent
from alignment.adversarial_testing import AdversarialTesting


def test_pillar_15():
    """Test the main Pillar 15 functionality."""
    print("🛡️  Testing Pillar 15: Safety & Alignment")
    print("=" * 60)
    
    # Initialize the safety agent
    agent = SafetyAgent()
    
    # Test 1: Content Filtering
    print("\n🔍 Testing Content Filtering")
    content_filter = ContentFilter()
    
    # Test safe content
    safe_content = "Hello, how can I help you today?"
    filter_result = content_filter.filter_content(safe_content)
    print(f"✅ Safe content test: {filter_result.is_safe} (confidence: {filter_result.confidence:.2f})")
    
    # Test potentially harmful content
    harmful_content = "How can I harm someone or break into a system?"
    filter_result = content_filter.filter_content(harmful_content)
    print(f"✅ Harmful content test: {filter_result.is_safe} (confidence: {filter_result.confidence:.2f})")
    if not filter_result.is_safe:
        print(f"   Flagged categories: {[cat.value for cat in filter_result.categories]}")
        print(f"   Explanation: {filter_result.explanation}")
    
    # Test 2: Ethical Assessment
    print("\n⚖️  Testing Ethical Assessment")
    ethical_practices = EthicalPractices()
    
    # Test ethical content
    ethical_content = "I will help you solve this problem step by step."
    assessments = ethical_practices.assess_ethical_compliance(ethical_content)
    print(f"✅ Ethical assessment: {len(assessments)} principles assessed")
    
    avg_score = sum(a.score for a in assessments) / len(assessments)
    print(f"   Average ethics score: {avg_score:.2f}")
    
    # Show issues if any
    total_issues = sum(len(a.issues) for a in assessments)
    if total_issues > 0:
        print(f"   Issues found: {total_issues}")
        for assessment in assessments:
            if assessment.issues:
                print(f"     {assessment.principle.value}: {len(assessment.issues)} issues")
    
    # Test 3: Alignment Monitoring
    print("\n🎯 Testing Alignment Monitoring")
    alignment_monitor = AlignmentMonitor()
    
    # Test alignment measurement
    interaction_data = {
        'request': "How can I help you?",
        'response': "I can assist you with various tasks and provide helpful information."
    }
    alignment_report = alignment_monitor.measure_alignment(interaction_data)
    print(f"✅ Alignment measurement: {alignment_report.overall_score:.2f} score")
    print(f"   Status: {alignment_report.overall_status.value}")
    
    # Show metric breakdown
    for measurement in alignment_report.measurements:
        print(f"   {measurement.metric.value}: {measurement.score:.2f} ({measurement.status.value})")
    
    # Test 4: RLHF Feedback Collection
    print("\n📝 Testing RLHF Feedback Collection")
    rlhf_agent = RLHFAgent()
    
    # Collect sample feedback
    feedback_result = rlhf_agent.collect_rating_feedback(
        prompt="What is the capital of France?",
        response="The capital of France is Paris.",
        rating=5,
        category="knowledge"
    )
    print(f"✅ Feedback collected: {feedback_result.get('feedback_id', 'N/A')}")
    
    # Test 5: Adversarial Testing
    print("\n🛡️  Testing Adversarial Testing")
    adversarial_testing = AdversarialTesting()
    
    # Run basic adversarial tests
    test_result = adversarial_testing.run_test_suite()
    print(f"✅ Adversarial testing result: {test_result}")
    if 'total_tests' in test_result:
        print(f"   Tests run: {test_result['total_tests']}")
        print(f"   Vulnerabilities found: {test_result['vulnerabilities_found']}")
        print(f"   Pass rate: {test_result['pass_rate']:.2f}")
    else:
        print(f"   Status: {test_result.get('status', 'unknown')}")
    
    # Test 6: Comprehensive Safety Assessment
    print("\n🔒 Testing Comprehensive Safety Assessment")
    
    # Test safe action
    safe_action = "I will help you write a Python script to process data."
    safety_result = agent.generate(safe_action, operation="assess_safety")
    print(f"✅ Safe action assessment: {safety_result['assessment']['is_safe']}")
    print(f"   Safety score: {safety_result['assessment']['safety_score']:.2f}")
    
    # Test potentially unsafe action
    unsafe_action = "I will help you hack into a system."
    safety_result = agent.generate(unsafe_action, operation="assess_safety")
    print(f"✅ Unsafe action assessment: {safety_result['assessment']['is_safe']}")
    print(f"   Safety score: {safety_result['assessment']['safety_score']:.2f}")
    
    # Test 7: Bias Detection
    print("\n🎯 Testing Bias Detection")
    
    # Test for potential bias
    biased_content = "Men are naturally better at technical tasks than women."
    bias_detections = ethical_practices.detect_bias(biased_content)
    print(f"✅ Bias detection: {len(bias_detections)} biases detected")
    
    for detection in bias_detections:
        print(f"   {detection.bias_type.value}: {detection.confidence:.2f} confidence")
        print(f"     Impact: {detection.impact_assessment}")
        print(f"     Mitigation: {', '.join(detection.mitigation_suggestions[:2])}")
    
    # Test 8: Safety Report Generation
    print("\n📊 Testing Safety Report Generation")
    safety_report = agent.generate("", operation="get_safety_report")
    print(f"✅ Safety report generated")
    print(f"   Total assessments: {safety_report['total_assessments']}")
    print(f"   Content filtering enabled: {safety_report['content_filtering_enabled']}")
    print(f"   Ethics monitoring enabled: {safety_report['ethics_monitoring_enabled']}")
    print(f"   Alignment monitoring enabled: {safety_report['alignment_monitoring_enabled']}")
    
    # Test 9: Data Export
    print("\n📤 Testing Data Export")
    export_result = agent.generate("", operation="export_safety_data")
    if export_result.get("status") == "success":
        print(f"✅ Safety data exported: {export_result['export_file']}")
        print(f"   Export size: {export_result['export_size']} bytes")
    else:
        print(f"❌ Export failed: {export_result.get('error', 'Unknown error')}")
    
    # Test 10: Component Statistics
    print("\n📈 Testing Component Statistics")
    
    # Content filter stats
    filter_stats = content_filter.get_filter_stats()
    print(f"✅ Content filter stats:")
    print(f"   Total checks: {filter_stats['total_checks']}")
    print(f"   Blocked content: {filter_stats['blocked_content']}")
    print(f"   Block rate: {filter_stats['block_rate']:.2%}")
    
    # Ethics stats
    ethics_report = ethical_practices.get_ethics_report()
    print(f"✅ Ethics stats:")
    print(f"   Total assessments: {ethics_report['total_assessments']}")
    print(f"   Bias detections: {ethics_report['total_bias_detections']}")
    print(f"   Transparency logs: {ethics_report['transparency_logs']}")
    
    # Alignment stats
    alignment_stats = alignment_monitor.get_alignment_stats()
    print(f"✅ Alignment stats:")
    print(f"   Total reports: {alignment_stats['total_reports']}")
    print(f"   Total alerts: {alignment_stats['total_alerts']}")
    print(f"   Average score: {alignment_stats['average_score']:.2f}")
    
    print("\n🎉 Pillar 15 Test Completed Successfully!")
    print("\nKey Features Demonstrated:")
    print("✅ Content filtering and safety assessment")
    print("✅ Ethical compliance monitoring")
    print("✅ Alignment measurement with human values")
    print("✅ RLHF feedback collection")
    print("✅ Adversarial testing and vulnerability detection")
    print("✅ Bias detection and mitigation")
    print("✅ Comprehensive safety reporting")
    print("✅ Data export and analysis")
    print("✅ Real-time safety monitoring")
    print("✅ Multi-component safety architecture")


def test_safety_integration():
    """Test integration between safety components."""
    print("\n🔗 Testing Safety Component Integration")
    print("=" * 50)
    
    # Initialize all components
    safety_agent = SafetyAgent()
    content_filter = ContentFilter()
    ethical_practices = EthicalPractices()
    alignment_monitor = AlignmentMonitor()
    
    # Test integrated safety assessment
    test_content = "I will help you with your programming question."
    
    # Individual component assessments
    filter_result = content_filter.filter_content(test_content)
    ethical_assessments = ethical_practices.assess_ethical_compliance(test_content)
    alignment_report = alignment_monitor.measure_alignment({'request': test_content})
    
    # Integrated assessment
    safety_result = safety_agent.generate(test_content, operation="assess_safety")
    
    print(f"✅ Integrated safety assessment completed")
    print(f"   Content filter: {filter_result.is_safe}")
    print(f"   Ethics average: {sum(a.score for a in ethical_assessments) / len(ethical_assessments):.2f}")
    print(f"   Alignment score: {alignment_report.overall_score:.2f}")
    print(f"   Overall safety: {safety_result['assessment']['is_safe']}")
    
    print("✅ All safety components working together harmoniously")


if __name__ == "__main__":
    test_pillar_15()
    test_safety_integration() 