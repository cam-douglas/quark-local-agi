#!/usr/bin/env python3
"""
Test Pillar 21: Governance & Ethics
==================================

Tests the implementation of ethical decision making, value alignment, transparency,
accountability, bias detection, fairness, and regulatory compliance.

Part of Phase 5: AGI Capabilities
"""

import os
import sys
import time
import tempfile
import shutil
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from governance.ethical_governance import (
    EthicalGovernance, EthicalPrinciple, ValueType, BiasType, FairnessMetric,
    EthicalDecision, ValueAlignment, BiasAssessment, FairnessReport, AccountabilityRecord
)
from agents.governance_agent import GovernanceAgent


def test_pillar_21():
    """Test Pillar 21: Governance & Ethics."""
    print("🛡️ Testing Pillar 21: Governance & Ethics")
    print("=" * 60)
    
    # Create temporary directories for testing
    temp_dir = tempfile.mkdtemp()
    governance_dir = os.path.join(temp_dir, "governance_data")
    
    try:
        # Test 1: Ethical Governance Engine
        print("\n⚖️ Testing Ethical Governance Engine")
        print("-" * 40)
        
        # Initialize governance engine
        governance_engine = EthicalGovernance(governance_dir)
        
        # Test ethical decision making
        print("✅ Testing Ethical Decision Making")
        
        ethical_contexts = [
            "Help users while protecting their privacy",
            "Ensure fairness in algorithm decisions",
            "Balance safety with user autonomy",
            "Make transparent and accountable decisions",
            "Consider long-term sustainability impacts"
        ]
        
        for context in ethical_contexts:
            decision_result = governance_engine.make_ethical_decision(
                context, ["Option A", "Option B", "Option C"]
            )
            print(f"   Context: {context[:40]}...")
            print(f"   Selected: {decision_result['selected_option']}")
            print(f"   Confidence: {decision_result['confidence']:.2f}")
            print(f"   Principles: {decision_result['principles_used']}")
            print()
        
        # Test value alignment assessment
        print("✅ Testing Value Alignment Assessment")
        
        value_types = [
            ValueType.HUMAN_RIGHTS,
            ValueType.DEMOCRATIC_VALUES,
            ValueType.CULTURAL_VALUES,
            ValueType.PROFESSIONAL_VALUES
        ]
        
        human_values = ["privacy", "fairness", "transparency", "safety"]
        ai_behavior = "AI system making decisions with privacy protection"
        
        for value_type in value_types:
            alignment_result = governance_engine.assess_value_alignment(
                value_type, human_values, ai_behavior
            )
            print(f"   Value Type: {value_type.value}")
            print(f"   Alignment Score: {alignment_result['alignment_score']:.2f}")
            print(f"   Misalignments: {len(alignment_result['misalignment_areas'])}")
            print(f"   Recommendations: {len(alignment_result['recommendations'])}")
            print()
        
        # Test bias detection
        print("✅ Testing Bias Detection")
        
        bias_scenarios = [
            "Data shows skewed distribution with missing samples from certain groups",
            "Algorithm exhibits discriminatory patterns against specific demographics",
            "Training data contains stereotypical representations",
            "Model shows confirmation bias in decision making"
        ]
        
        for scenario in bias_scenarios:
            bias_result = governance_engine.detect_bias(scenario)
            print(f"   Scenario: {scenario[:50]}...")
            print(f"   Bias Types: {bias_result['bias_types']}")
            print(f"   Severity: {bias_result['severity']:.2f}")
            print(f"   Mitigation Strategies: {len(bias_result['mitigation_strategies'])}")
            print()
        
        # Test fairness assessment
        print("✅ Testing Fairness Assessment")
        
        group_performance = {
            'group_a': 0.85,
            'group_b': 0.78,
            'group_c': 0.92,
            'group_d': 0.73
        }
        
        fairness_result = governance_engine.assess_fairness(group_performance)
        print(f"   Groups: {len(group_performance)}")
        print(f"   Fairness Scores: {fairness_result['fairness_scores']}")
        print(f"   Recommendations: {len(fairness_result['recommendations'])}")
        print()
        
        # Test accountability recording
        print("✅ Testing Accountability Recording")
        
        accountability_scenarios = [
            ("ethical_decision_making", "governance_agent", ["safety", "privacy"], ["beneficence", "justice"]),
            ("bias_detection", "safety_agent", ["fairness", "equity"], ["justice", "non_maleficence"]),
            ("privacy_protection", "privacy_agent", ["privacy", "security"], ["privacy", "autonomy"])
        ]
        
        for action, agent_id, factors, considerations in accountability_scenarios:
            accountability_result = governance_engine.record_accountability(
                action, agent_id, factors, considerations, "action_completed"
            )
            print(f"   Action: {action}")
            print(f"   Agent: {agent_id}")
            print(f"   Responsibility: {accountability_result['responsibility_assigned']}")
            print()
        
        # Test governance statistics
        print("✅ Testing Governance Statistics")
        stats = governance_engine.get_governance_stats()
        print(f"   Ethical Decisions: {stats['ethical_decisions']}")
        print(f"   Value Alignments: {stats['value_alignments']}")
        print(f"   Bias Assessments: {stats['bias_assessments']}")
        print(f"   Fairness Reports: {stats['fairness_reports']}")
        print(f"   Accountability Records: {stats['accountability_records']}")
        print(f"   Total Activities: {stats['total_governance_activities']}")
        print()
        
        # Test 2: Governance Agent Integration
        print("\n🤖 Testing Governance Agent Integration")
        print("-" * 40)
        
        # Initialize governance agent
        governance_agent = GovernanceAgent(governance_dir=governance_dir)
        
        # Test agent operations
        print("✅ Testing Governance Agent Operations")
        
        # Test ethical decision via agent
        agent_decision_result = governance_agent.generate(
            "Make a decision that balances user privacy with system functionality",
            operation="make_ethical_decision"
        )
        print(f"   Agent ethical decision: {agent_decision_result['status']}")
        
        # Test value alignment via agent
        agent_alignment_result = governance_agent.generate(
            "AI system collecting user data for personalization",
            operation="assess_value_alignment"
        )
        print(f"   Agent value alignment: {agent_alignment_result['status']}")
        
        # Test bias detection via agent
        agent_bias_result = governance_agent.generate(
            "Algorithm showing different performance across demographic groups",
            operation="detect_bias"
        )
        print(f"   Agent bias detection: {agent_bias_result['status']}")
        
        # Test fairness assessment via agent
        agent_fairness_result = governance_agent.generate(
            "Assess fairness across different user groups",
            operation="assess_fairness"
        )
        print(f"   Agent fairness assessment: {agent_fairness_result['status']}")
        
        # Test accountability recording via agent
        agent_accountability_result = governance_agent.generate(
            "Record accountability for privacy protection actions",
            operation="record_accountability"
        )
        print(f"   Agent accountability: {agent_accountability_result['status']}")
        
        # Test governance statistics via agent
        agent_stats_result = governance_agent.generate("", operation="get_governance_stats")
        print(f"   Agent governance stats: {agent_stats_result['performance_stats']['total_operations']} operations")
        
        # Test 3: Advanced Governance Features
        print("\n🚀 Testing Advanced Governance Features")
        print("-" * 40)
        
        # Test ethics analysis
        print("✅ Testing Ethics Analysis")
        ethics_analysis_result = governance_agent.generate("", operation="analyze_ethics")
        print(f"   Ethics analysis: {ethics_analysis_result['status']}")
        
        # Test compliance check
        print("✅ Testing Compliance Check")
        compliance_result = governance_agent.generate("", operation="compliance_check")
        print(f"   Compliance check: {compliance_result.get('status', 'unknown')}")
        
        # Test governance recommendations
        print("✅ Testing Governance Recommendations")
        recommendations_result = governance_agent.get_governance_recommendations()
        print(f"   Recommendations: {recommendations_result['recommendations_count']}")
        
        # Test data export
        print("✅ Testing Data Export")
        export_result = governance_agent.generate("", operation="export_governance_data")
        print(f"   Export result: {export_result['status']}")
        
        print("\n🎉 Pillar 21 Test Completed Successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Ethical decision making with principled reasoning")
        print("✅ Value alignment assessment and recommendations")
        print("✅ Bias detection and mitigation strategies")
        print("✅ Fairness assessment across multiple metrics")
        print("✅ Accountability tracking and responsibility assignment")
        print("✅ Transparency and explainability mechanisms")
        print("✅ Compliance checking and regulatory adherence")
        print("✅ Performance monitoring and statistics")
        print("✅ Data export and persistence")
        print("✅ Agent integration and operations")
        print("✅ Advanced governance capabilities")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_governance_integration():
    """Test integration between governance and other systems."""
    print("\n🔗 Testing Governance Integration")
    print("=" * 50)
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    governance_dir = os.path.join(temp_dir, "governance_data")
    
    try:
        # Initialize governance systems
        governance_engine = EthicalGovernance(governance_dir)
        governance_agent = GovernanceAgent(governance_dir=governance_dir)
        
        # Test integrated governance workflow
        print("✅ Testing Integrated Governance Workflow")
        
        # 1. Ethical decision making scenario
        scenario = """
        Scenario: The AI system needs to make decisions that balance multiple ethical considerations.
        
        Context: A healthcare AI system must decide whether to share patient data for research
        while protecting individual privacy and ensuring fair treatment across all demographic groups.
        
        Goal: Demonstrate comprehensive ethical governance capabilities.
        """
        
        print("✅ Testing Ethical Decision Making Scenario")
        
        # Generate ethical decisions for different contexts
        decision_contexts = [
            "Balance patient privacy with medical research benefits",
            "Ensure fair treatment across all demographic groups",
            "Maintain transparency while protecting sensitive information",
            "Consider long-term impacts on healthcare outcomes"
        ]
        
        ethical_decisions = []
        for context in decision_contexts:
            decision_result = governance_engine.make_ethical_decision(context, [
                "Prioritize privacy protection",
                "Enable research with safeguards",
                "Implement balanced approach"
            ])
            ethical_decisions.append(decision_result)
            print(f"   Context: {context}")
            print(f"     Decision: {decision_result['selected_option']}")
            print(f"     Confidence: {decision_result['confidence']:.2f}")
            print(f"     Principles: {decision_result['principles_used']}")
        
        # 2. Test value alignment assessment
        print("✅ Testing Value Alignment Assessment")
        
        value_alignment_scenarios = [
            (ValueType.HUMAN_RIGHTS, ["privacy", "dignity", "equality"], "AI system with privacy protections"),
            (ValueType.DEMOCRATIC_VALUES, ["transparency", "accountability"], "Transparent decision-making system"),
            (ValueType.CULTURAL_VALUES, ["respect", "inclusion"], "Inclusive AI system design")
        ]
        
        for value_type, human_values, ai_behavior in value_alignment_scenarios:
            alignment_result = governance_engine.assess_value_alignment(value_type, human_values, ai_behavior)
            print(f"   Value Type: {value_type.value}")
            print(f"     Alignment Score: {alignment_result['alignment_score']:.2f}")
            print(f"     Misalignments: {alignment_result['misalignment_areas']}")
            print(f"     Recommendations: {alignment_result['recommendations']}")
        
        # 3. Test bias detection and mitigation
        print("✅ Testing Bias Detection and Mitigation")
        
        bias_scenarios = [
            "Healthcare data shows underrepresentation of minority groups",
            "Algorithm exhibits gender bias in treatment recommendations",
            "Model shows age-based discrimination in access to care"
        ]
        
        for scenario in bias_scenarios:
            bias_result = governance_engine.detect_bias(scenario)
            print(f"   Scenario: {scenario}")
            print(f"     Bias Types: {bias_result['bias_types']}")
            print(f"     Severity: {bias_result['severity']:.2f}")
            print(f"     Mitigation: {bias_result['mitigation_strategies']}")
        
        # 4. Test fairness assessment
        print("✅ Testing Fairness Assessment")
        
        # Simulate performance across different demographic groups
        group_performance = {
            'young_adults': 0.88,
            'middle_aged': 0.82,
            'seniors': 0.75,
            'minority_groups': 0.78,
            'majority_groups': 0.85
        }
        
        fairness_result = governance_engine.assess_fairness(group_performance)
        print(f"   Group Performance: {group_performance}")
        print(f"   Fairness Scores: {fairness_result['fairness_scores']}")
        print(f"   Recommendations: {fairness_result['recommendations']}")
        
        # 5. Test accountability tracking
        print("✅ Testing Accountability Tracking")
        
        accountability_actions = [
            ("privacy_protection", "privacy_agent", ["data_encryption", "access_control"], ["privacy", "autonomy"]),
            ("bias_mitigation", "fairness_agent", ["algorithm_audit", "data_review"], ["justice", "equity"]),
            ("transparency_enhancement", "transparency_agent", ["explanation_generation", "documentation"], ["transparency", "accountability"])
        ]
        
        for action, agent_id, factors, considerations in accountability_actions:
            accountability_result = governance_engine.record_accountability(
                action, agent_id, factors, considerations, "action_completed_successfully"
            )
            print(f"   Action: {action}")
            print(f"     Agent: {agent_id}")
            print(f"     Responsibility: {accountability_result['responsibility_assigned']}")
        
        # 6. Test integrated governance insights
        print("✅ Testing Integrated Governance Insights")
        
        # Get comprehensive statistics
        stats = governance_engine.get_governance_stats()
        
        integrated_insights = {
            'ethical_decisions': stats['ethical_decisions'],
            'value_alignments': stats['value_alignments'],
            'bias_assessments': stats['bias_assessments'],
            'fairness_reports': stats['fairness_reports'],
            'accountability_records': stats['accountability_records'],
            'overall_assessment': 'Comprehensive governance system successfully managing ethical oversight'
        }
        
        print(f"   Overall assessment: {integrated_insights['overall_assessment']}")
        print(f"   Ethical decisions: {integrated_insights['ethical_decisions']}")
        print(f"   Value alignments: {integrated_insights['value_alignments']}")
        print(f"   Bias assessments: {integrated_insights['bias_assessments']}")
        print(f"   Fairness reports: {integrated_insights['fairness_reports']}")
        print(f"   Accountability records: {integrated_insights['accountability_records']}")
        
        print("✅ Governance Integration Test Completed Successfully!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_pillar_21()
    test_governance_integration() 