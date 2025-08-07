#!/usr/bin/env python3
"""
Safety CLI for Quark AI Assistant
======================================

Provides commands to view and monitor safety rules, content filtering,
ethical practices, alignment monitoring, and adversarial testing.

Part of Pillar 15: Safety & Alignment
"""

import click
import json
from datetime import datetime
from typing import Dict, Any

from core.immutable_safety_rules import get_safety_rules, verify_safety_integrity
from core.safety_enforcement import get_safety_enforcement
from agents.safety_agent import SafetyAgent
from alignment.content_filtering import ContentFilter
from alignment.ethical_practices import EthicalPractices
from alignment.alignment_monitor import AlignmentMonitor
from alignment.rlhf_agent import RLHFAgent
from alignment.adversarial_testing import AdversarialTesting


@click.group()
def safety():
    """Safety and alignment management commands for Quark AI Assistant."""
    pass


@safety.command()
def status():
    """Check the status of safety rules and enforcement."""
    click.secho("🔒 SAFETY SYSTEM STATUS", fg="blue", bold=True)
    click.echo()
    
    try:
        # Verify safety rules integrity
        verify_safety_integrity()
        click.secho("✅ Safety rules integrity verified", fg="green")
    except Exception as e:
        click.secho(f"❌ Safety rules integrity compromised: {e}", fg="red")
        return
    
    # Get safety enforcement status
    enforcement = get_safety_enforcement()
    safety_report = enforcement.get_safety_report()
    
    click.echo()
    click.secho("📊 SAFETY STATISTICS", fg="yellow", bold=True)
    click.echo(f"Total actions processed: {safety_report['total_actions']}")
    click.echo(f"Actions blocked: {safety_report['blocked_actions']}")
    
    if safety_report['blocked_actions'] > 0:
        click.secho("⚠️  Some actions have been blocked for safety reasons", fg="yellow")
    
    click.echo()
    click.secho("✅ Safety system is active and protecting users", fg="green")


@safety.command()
def rules():
    """Display all immutable safety rules."""
    click.secho("🔒 IMMUTABLE SAFETY RULES", fg="blue", bold=True)
    click.echo("These rules CANNOT be modified by the AI system.")
    click.echo()
    
    try:
        safety_rules = get_safety_rules()
        principles = safety_rules.get_safety_principles()
        
        # Display core principles
        click.secho("CORE SAFETY PRINCIPLES:", fg="yellow", bold=True)
        click.echo()
        
        for principle_name, principle_data in principles["core_principles"].items():
            click.secho(f"• {principle_data['rule']}", fg="red", bold=True)
            click.echo(f"  {principle_data['description']}")
            click.echo("  Requirements:")
            for req in principle_data['requirements']:
                click.echo(f"    - {req}")
            click.echo()
        
        # Display forbidden actions
        click.secho("FORBIDDEN ACTIONS:", fg="yellow", bold=True)
        click.echo()
        for action in principles["forbidden_actions"]:
            click.secho(f"❌ {action}", fg="red")
        click.echo()
        
        # Display required confirmations
        click.secho("REQUIRED CONFIRMATIONS:", fg="yellow", bold=True)
        click.echo()
        for category, requirements in principles["required_confirmations"].items():
            click.secho(f"• {category.replace('_', ' ').title()}:", fg="cyan")
            click.echo(f"  {requirements['description']}")
            click.echo(f"  Confirmation required: {requirements['confirmation_required']}")
            click.echo(f"  Explanation required: {requirements['explanation_required']}")
            click.echo()
            
    except Exception as e:
        click.secho(f"❌ Error accessing safety rules: {e}", fg="red")


@safety.command()
def report():
    """Generate comprehensive safety report."""
    click.secho("📊 COMPREHENSIVE SAFETY REPORT", fg="blue", bold=True)
    click.echo()
    
    try:
        # Initialize safety agent
        safety_agent = SafetyAgent()
        
        # Get safety report
        report = safety_agent.generate("", operation="get_safety_report")
        
        if report.get("status") == "success":
            click.secho("✅ Safety System Status:", fg="green")
            click.echo(f"  Safety Enabled: {report['safety_enabled']}")
            click.echo(f"  Content Filtering: {report['content_filtering_enabled']}")
            click.echo(f"  Ethics Monitoring: {report['ethics_monitoring_enabled']}")
            click.echo(f"  Alignment Monitoring: {report['alignment_monitoring_enabled']}")
            click.echo(f"  RLHF Enabled: {report['rlhf_enabled']}")
            click.echo(f"  Adversarial Testing: {report['adversarial_testing_enabled']}")
            
            click.echo()
            click.secho("📈 Statistics:", fg="yellow")
            click.echo(f"  Total Assessments: {report['total_assessments']}")
            click.echo(f"  Total Violations: {report['total_violations']}")
            click.echo(f"  Total Improvements: {report['total_improvements']}")
            
            # Content filter stats
            if 'content_filter_stats' in report:
                filter_stats = report['content_filter_stats']
                click.echo()
                click.secho("🔍 Content Filtering:", fg="cyan")
                click.echo(f"  Total Checks: {filter_stats['total_checks']}")
                click.echo(f"  Blocked Content: {filter_stats['blocked_content']}")
                click.echo(f"  Warned Content: {filter_stats['warned_content']}")
                click.echo(f"  Block Rate: {filter_stats['block_rate']:.2%}")
            
            # Ethics stats
            if 'ethics_report' in report:
                ethics_stats = report['ethics_report']
                click.echo()
                click.secho("⚖️  Ethics Monitoring:", fg="cyan")
                click.echo(f"  Total Assessments: {ethics_stats['total_assessments']}")
                click.echo(f"  Bias Detections: {ethics_stats['total_bias_detections']}")
                click.echo(f"  Transparency Logs: {ethics_stats['transparency_logs']}")
            
            # Alignment stats
            if 'alignment_stats' in report:
                align_stats = report['alignment_stats']
                click.echo()
                click.secho("🎯 Alignment Monitoring:", fg="cyan")
                click.echo(f"  Total Reports: {align_stats['total_reports']}")
                click.echo(f"  Total Alerts: {align_stats['total_alerts']}")
                click.echo(f"  Average Score: {align_stats['average_score']:.2f}")
            
        else:
            click.secho(f"❌ Error generating safety report: {report.get('error', 'Unknown error')}", fg="red")
            
    except Exception as e:
        click.secho(f"❌ Error: {e}", fg="red")


@safety.command()
@click.argument('content', required=False)
def filter(content):
    """Filter content for safety."""
    if not content:
        content = click.prompt("Enter content to filter")
    
    click.secho("🔍 CONTENT FILTERING", fg="blue", bold=True)
    click.echo()
    
    try:
        content_filter = ContentFilter()
        result = content_filter.filter_content(content)
        
        if result.is_safe:
            click.secho("✅ Content appears safe", fg="green")
        else:
            click.secho("❌ Content flagged as potentially unsafe", fg="red")
        
        click.echo()
        click.echo(f"Confidence: {result.confidence:.2f}")
        click.echo(f"Severity: {result.severity.value}")
        
        if result.categories:
            click.echo(f"Categories: {', '.join([cat.value for cat in result.categories])}")
        
        if result.flagged_terms:
            click.echo(f"Flagged terms: {', '.join(result.flagged_terms[:5])}")
        
        click.echo(f"Explanation: {result.explanation}")
        
    except Exception as e:
        click.secho(f"❌ Error filtering content: {e}", fg="red")


@safety.command()
@click.argument('content', required=False)
def ethics(content):
    """Assess ethical compliance of content."""
    if not content:
        content = click.prompt("Enter content to assess")
    
    click.secho("⚖️  ETHICAL ASSESSMENT", fg="blue", bold=True)
    click.echo()
    
    try:
        ethical_practices = EthicalPractices()
        assessments = ethical_practices.assess_ethical_compliance(content)
        
        total_score = sum(a.score for a in assessments) / len(assessments)
        click.echo(f"Overall Ethics Score: {total_score:.2f}")
        click.echo()
        
        for assessment in assessments:
            principle_name = assessment.principle.value.replace('_', ' ').title()
            score_color = "green" if assessment.score >= 0.8 else "yellow" if assessment.score >= 0.6 else "red"
            
            click.secho(f"• {principle_name}: {assessment.score:.2f}", fg=score_color)
            
            if assessment.issues:
                click.echo("  Issues:")
                for issue in assessment.issues[:3]:  # Show first 3 issues
                    click.echo(f"    - {issue}")
            
            if assessment.recommendations:
                click.echo("  Recommendations:")
                for rec in assessment.recommendations[:2]:  # Show first 2 recommendations
                    click.echo(f"    - {rec}")
            click.echo()
        
    except Exception as e:
        click.secho(f"❌ Error assessing ethics: {e}", fg="red")


@safety.command()
@click.argument('content', required=False)
def alignment(content):
    """Measure alignment with human values."""
    if not content:
        content = click.prompt("Enter content to measure alignment")
    
    click.secho("🎯 ALIGNMENT MEASUREMENT", fg="blue", bold=True)
    click.echo()
    
    try:
        alignment_monitor = AlignmentMonitor()
        interaction_data = {'request': content}
        report = alignment_monitor.measure_alignment(interaction_data)
        
        click.echo(f"Overall Alignment Score: {report.overall_score:.2f}")
        click.echo(f"Alignment Status: {report.overall_status.value}")
        click.echo()
        
        click.secho("Metric Breakdown:", fg="yellow")
        for measurement in report.measurements:
            metric_name = measurement.metric.value.replace('_', ' ').title()
            score_color = "green" if measurement.score >= 0.8 else "yellow" if measurement.score >= 0.6 else "red"
            
            click.secho(f"• {metric_name}: {measurement.score:.2f} ({measurement.status.value})", fg=score_color)
            
            if measurement.evidence:
                click.echo("  Evidence:")
                for evidence in measurement.evidence[:2]:  # Show first 2 pieces of evidence
                    click.echo(f"    - {evidence}")
            click.echo()
        
        if report.recommendations:
            click.secho("Recommendations:", fg="cyan")
            for rec in report.recommendations:
                click.echo(f"  - {rec}")
        
    except Exception as e:
        click.secho(f"❌ Error measuring alignment: {e}", fg="red")


@safety.command()
@click.argument('test_type', required=False)
def adversarial(test_type):
    """Run adversarial testing."""
    if not test_type:
        test_type = click.prompt("Enter test type (prompt_injection, jailbreak, harm_prompting, etc.)")
    
    click.secho("🛡️  ADVERSARIAL TESTING", fg="blue", bold=True)
    click.echo()
    
    try:
        adversarial_testing = AdversarialTesting()
        
        # Configure test suite
        test_config = {
            'categories': [test_type],
            'custom_prompts': []
        }
        
        result = adversarial_testing.run_test_suite(
            test_categories=test_config['categories'],
            custom_prompts=test_config['custom_prompts']
        )
        
        click.echo(f"Tests Run: {result['tests_run']}")
        click.echo(f"Vulnerabilities Found: {result['vulnerabilities_found']}")
        click.echo(f"Overall Score: {result['overall_score']:.2f}")
        click.echo()
        
        if result['test_results']:
            click.secho("Test Results:", fg="yellow")
            for test_result in result['test_results'][:5]:  # Show first 5 results
                click.echo(f"  • {test_result['test_name']}: {test_result['result']}")
        
    except Exception as e:
        click.secho(f"❌ Error running adversarial tests: {e}", fg="red")


@safety.command()
def feedback():
    """Collect human feedback for RLHF."""
    click.secho("📝 HUMAN FEEDBACK COLLECTION", fg="blue", bold=True)
    click.echo()
    
    try:
        rlhf_agent = RLHFAgent()
        
        # Get feedback from user
        prompt = click.prompt("Enter the AI prompt")
        response = click.prompt("Enter the AI response")
        feedback_type = click.prompt("Enter feedback type", type=click.Choice(['rating', 'preference', 'binary']))
        
        if feedback_type == 'rating':
            rating = click.prompt("Enter rating (1-5)", type=int)
            feedback_text = click.prompt("Enter feedback text (optional)")
        elif feedback_type == 'preference':
            rating = 0
            feedback_text = click.prompt("Enter preference feedback")
        else:  # binary
            rating = click.prompt("Enter rating (0 or 1)", type=int)
            feedback_text = click.prompt("Enter feedback text (optional)")
        
        result = rlhf_agent.collect_feedback(
            prompt=prompt,
            response=response,
            feedback_type=feedback_type,
            rating=rating,
            feedback_text=feedback_text
        )
        
        if result.get('feedback_id'):
            click.secho(f"✅ Feedback collected successfully (ID: {result['feedback_id']})", fg="green")
        else:
            click.secho("❌ Failed to collect feedback", fg="red")
        
    except Exception as e:
        click.secho(f"❌ Error collecting feedback: {e}", fg="red")


@safety.command()
def export():
    """Export safety data."""
    click.secho("📤 EXPORTING SAFETY DATA", fg="blue", bold=True)
    click.echo()
    
    try:
        safety_agent = SafetyAgent()
        result = safety_agent.generate("", operation="export_safety_data")
        
        if result.get("status") == "success":
            click.secho(f"✅ Safety data exported successfully", fg="green")
            click.echo(f"File: {result['export_file']}")
            click.echo(f"Size: {result['export_size']} bytes")
        else:
            click.secho(f"❌ Export failed: {result.get('error', 'Unknown error')}", fg="red")
        
    except Exception as e:
        click.secho(f"❌ Error exporting data: {e}", fg="red")


@safety.command()
@click.argument('action', required=False)
def test(action):
    """Test safety system with sample content."""
    if not action:
        action = click.prompt("Enter action to test")
    
    click.secho("🧪 SAFETY SYSTEM TEST", fg="blue", bold=True)
    click.echo()
    
    try:
        safety_agent = SafetyAgent()
        result = safety_agent.generate(action, operation="assess_safety")
        
        if result.get("status") == "success":
            assessment = result['assessment']
            
            if assessment['is_safe']:
                click.secho("✅ Action appears safe", fg="green")
            else:
                click.secho("❌ Action flagged as potentially unsafe", fg="red")
            
            click.echo(f"Safety Score: {assessment['safety_score']:.2f}")
            click.echo()
            
            # Content filter results
            content_filter = assessment['content_filter']
            click.secho("Content Filter:", fg="yellow")
            click.echo(f"  Safe: {content_filter['is_safe']}")
            click.echo(f"  Categories: {', '.join(content_filter['categories'])}")
            click.echo(f"  Severity: {content_filter['severity']}")
            click.echo(f"  Confidence: {content_filter['confidence']:.2f}")
            click.echo(f"  Explanation: {content_filter['explanation']}")
            click.echo()
            
            # Ethics results
            ethics = assessment['ethics']
            click.secho("Ethics Assessment:", fg="yellow")
            click.echo(f"  Total Assessments: {ethics['total_assessments']}")
            click.echo(f"  Average Score: {ethics['average_score']:.2f}")
            click.echo(f"  Issues Found: {ethics['issues_found']}")
            click.echo()
            
            # Alignment results
            alignment = assessment['alignment']
            click.secho("Alignment Measurement:", fg="yellow")
            click.echo(f"  Overall Score: {alignment['overall_score']:.2f}")
            click.echo(f"  Status: {alignment['overall_status']}")
            click.echo()
            
            # Recommendations
            if assessment['recommendations']:
                click.secho("Recommendations:", fg="cyan")
                for rec in assessment['recommendations']:
                    click.echo(f"  - {rec}")
        
        else:
            click.secho(f"❌ Test failed: {result.get('error', 'Unknown error')}", fg="red")
        
    except Exception as e:
        click.secho(f"❌ Error testing safety system: {e}", fg="red")


@safety.command()
def integrity():
    """Check safety system integrity."""
    click.secho("🔍 SAFETY SYSTEM INTEGRITY CHECK", fg="blue", bold=True)
    click.echo()
    
    try:
        # Check safety rules integrity
        verify_safety_integrity()
        click.secho("✅ Safety rules integrity verified", fg="green")
        
        # Check safety enforcement
        enforcement = get_safety_enforcement()
        click.secho("✅ Safety enforcement active", fg="green")
        
        # Check content filtering
        content_filter = ContentFilter()
        click.secho("✅ Content filtering active", fg="green")
        
        # Check ethical practices
        ethical_practices = EthicalPractices()
        click.secho("✅ Ethical practices active", fg="green")
        
        # Check alignment monitoring
        alignment_monitor = AlignmentMonitor()
        click.secho("✅ Alignment monitoring active", fg="green")
        
        # Check RLHF agent
        rlhf_agent = RLHFAgent()
        click.secho("✅ RLHF agent active", fg="green")
        
        # Check adversarial testing
        adversarial_testing = AdversarialTesting()
        click.secho("✅ Adversarial testing active", fg="green")
        
        click.echo()
        click.secho("🎉 All safety components are operational", fg="green")
        
    except Exception as e:
        click.secho(f"❌ Safety system integrity compromised: {e}", fg="red")


@safety.command()
def capabilities():
    """Show safety system capabilities."""
    click.secho("🛡️  SAFETY SYSTEM CAPABILITIES", fg="blue", bold=True)
    click.echo()
    
    capabilities = [
        "🔒 Immutable Safety Rules - Core safety principles that cannot be modified",
        "🔍 Content Filtering - Detect and block harmful or inappropriate content",
        "⚖️  Ethical Assessment - Evaluate compliance with ethical principles",
        "🎯 Alignment Monitoring - Measure alignment with human values",
        "📝 RLHF Integration - Collect and use human feedback for improvement",
        "🛡️  Adversarial Testing - Test system against safety vulnerabilities",
        "📊 Comprehensive Reporting - Generate detailed safety reports",
        "📤 Data Export - Export safety data for analysis",
        "🔍 Integrity Verification - Verify safety system integrity",
        "⚡ Real-time Monitoring - Monitor safety in real-time"
    ]
    
    for capability in capabilities:
        click.echo(f"  {capability}")
    
    click.echo()
    click.secho("These capabilities work together to ensure safe and aligned AI operation.", fg="green")


if __name__ == '__main__':
    safety() 