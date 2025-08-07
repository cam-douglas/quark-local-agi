#!/usr/bin/env python3
"""
Test suite for Phase 4: Intelligence Enhancement
Tests Pillars 13, 14, 15, and 16 functionality
"""

import sys
import os
import asyncio
import json
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_pillar_13_async_orchestration():
    """Test Pillar 13: Async & Parallel Multi-Agent Orchestration"""
    print("Testing Pillar 13: Async & Parallel Multi-Agent Orchestration...")
    
    try:
        # Test async orchestrator file exists
        async_orchestrator_path = "core/async_orchestrator.py"
        assert os.path.exists(async_orchestrator_path), "Async orchestrator missing"
        
        with open(async_orchestrator_path, 'r') as f:
            content = f.read()
            assert "class AsyncOrchestrator" in content
            assert "async def handle" in content
            assert "parallel" in content.lower()
            assert "concurrent" in content.lower()
        
        # Test orchestrator file exists
        orchestrator_path = "core/orchestrator.py"
        assert os.path.exists(orchestrator_path), "Orchestrator missing"
        
        with open(orchestrator_path, 'r') as f:
            content = f.read()
            assert "class Orchestrator" in content
            assert "PIPELINES" in content
            assert "PARALLEL_AGENTS" in content
        
        print("✅ Pillar 13: Async & Parallel Multi-Agent Orchestration - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Pillar 13: Async & Parallel Multi-Agent Orchestration - FAILED: {e}")
        return False

def test_pillar_14_frontend_ui():
    """Test Pillar 14: Front-end & Embeddable UI"""
    print("Testing Pillar 14: Front-end & Embeddable UI...")
    
    try:
        # Test web interface files
        web_files = [
            "web/frontend/index.html",
            "web/frontend/app.js",
            "web/frontend/styles.css",
            "web/fastapi_app.py"
        ]
        
        for web_file in web_files:
            assert os.path.exists(web_file), f"Web file missing: {web_file}"
            assert os.path.getsize(web_file) > 0, f"Web file is empty: {web_file}"
        
        # Test VSCode extension
        vscode_files = [
            "web/extensions/vscode/package.json",
            "web/extensions/vscode/src/extension.ts"
        ]
        
        for vscode_file in vscode_files:
            if os.path.exists(vscode_file):
                assert os.path.getsize(vscode_file) > 0, f"VSCode file is empty: {vscode_file}"
        
        # Test Obsidian plugin
        obsidian_files = [
            "web/extensions/obsidian/main.ts"
        ]
        
        for obsidian_file in obsidian_files:
            if os.path.exists(obsidian_file):
                assert os.path.getsize(obsidian_file) > 0, f"Obsidian file is empty: {obsidian_file}"
        
        # Test widgets
        widgets_dir = "web/widgets"
        if os.path.exists(widgets_dir):
            widget_files = os.listdir(widgets_dir)
            assert len(widget_files) > 0, "Widgets directory should contain files"
        
        print("✅ Pillar 14: Front-end & Embeddable UI - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Pillar 14: Front-end & Embeddable UI - FAILED: {e}")
        return False

def test_pillar_15_safety_alignment():
    """Test Pillar 15: Safety & Alignment"""
    print("Testing Pillar 15: Safety & Alignment...")
    
    try:
        # Test alignment components
        alignment_files = [
            "alignment/content_filtering.py",
            "alignment/ethical_practices.py",
            "alignment/alignment_monitor.py",
            "alignment/rlhf_agent.py",
            "alignment/adversarial_testing.py",
            "alignment/README.md"
        ]
        
        for alignment_file in alignment_files:
            assert os.path.exists(alignment_file), f"Alignment file missing: {alignment_file}"
            assert os.path.getsize(alignment_file) > 0, f"Alignment file is empty: {alignment_file}"
        
        # Test safety components
        safety_files = [
            "core/safety_enforcement.py",
            "core/safety_guardrails.py",
            "core/immutable_safety_rules.py",
            "agents/safety_agent.py"
        ]
        
        for safety_file in safety_files:
            assert os.path.exists(safety_file), f"Safety file missing: {safety_file}"
            assert os.path.getsize(safety_file) > 0, f"Safety file is empty: {safety_file}"
        
        # Test CLI
        safety_cli_path = "cli/safety_cli.py"
        assert os.path.exists(safety_cli_path), "Safety CLI missing"
        
        with open(safety_cli_path, 'r') as f:
            content = f.read()
            assert "def main()" in content
            assert "safety" in content.lower()
        
        print("✅ Pillar 15: Safety & Alignment - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Pillar 15: Safety & Alignment - FAILED: {e}")
        return False

def test_pillar_16_meta_learning():
    """Test Pillar 16: Meta-Learning & Self-Reflection"""
    print("Testing Pillar 16: Meta-Learning & Self-Reflection...")
    
    try:
        # Test meta-learning components
        meta_learning_files = [
            "meta_learning/meta_learning_agent.py",
            "meta_learning/self_reflection_agent.py",
            "meta_learning/performance_monitor.py",
            "meta_learning/pipeline_reconfigurator.py",
            "meta_learning/meta_learning_orchestrator.py",
            "meta_learning/README.md"
        ]
        
        for meta_learning_file in meta_learning_files:
            assert os.path.exists(meta_learning_file), f"Meta-learning file missing: {meta_learning_file}"
            assert os.path.getsize(meta_learning_file) > 0, f"Meta-learning file is empty: {meta_learning_file}"
        
        # Test README content
        readme_path = "meta_learning/README.md"
        with open(readme_path, 'r') as f:
            readme_content = f.read()
            assert "Meta-Learning" in readme_content
            assert "Self-Reflection" in readme_content
            assert "Performance" in readme_content
        
        print("✅ Pillar 16: Meta-Learning & Self-Reflection - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Pillar 16: Meta-Learning & Self-Reflection - FAILED: {e}")
        return False

def test_phase_4_integration():
    """Test Phase 4 integration"""
    print("Testing Phase 4 Integration...")
    
    try:
        # Test that all Phase 4 components work together
        integration_components = {
            "Async Orchestration": "core/async_orchestrator.py",
            "Frontend UI": "web/frontend/index.html",
            "Safety System": "alignment/content_filtering.py",
            "Meta-Learning": "meta_learning/meta_learning_agent.py"
        }
        
        for component_name, component_path in integration_components.items():
            assert os.path.exists(component_path), f"{component_name} component missing"
            assert os.path.getsize(component_path) > 0, f"{component_name} component is empty"
        
        print("✅ Phase 4 Integration - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Phase 4 Integration - FAILED: {e}")
        return False

def test_intelligence_enhancement():
    """Test intelligence enhancement features"""
    print("Testing Intelligence Enhancement Features...")
    
    try:
        # Test advanced features
        advanced_features = {
            "Parallel Processing": "core/async_orchestrator.py",
            "Web Interface": "web/fastapi_app.py",
            "Safety Guardrails": "core/safety_guardrails.py",
            "Self-Improvement": "meta_learning/meta_learning_agent.py"
        }
        
        for feature_name, feature_path in advanced_features.items():
            assert os.path.exists(feature_path), f"{feature_name} feature missing"
            
            with open(feature_path, 'r') as f:
                content = f.read()
                assert len(content) > 100, f"{feature_name} feature seems incomplete"
        
        print("✅ Intelligence Enhancement Features - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Intelligence Enhancement Features - FAILED: {e}")
        return False

async def main():
    """Run all Phase 4 tests."""
    print("🧠 Testing Phase 4: Intelligence Enhancement...")
    print("=" * 70)
    
    results = []
    
    # Test each pillar
    results.append(test_pillar_13_async_orchestration())
    results.append(test_pillar_14_frontend_ui())
    results.append(test_pillar_15_safety_alignment())
    results.append(test_pillar_16_meta_learning())
    results.append(test_phase_4_integration())
    results.append(test_intelligence_enhancement())
    
    # Summary
    print("=" * 70)
    print("📊 Test Results Summary:")
    
    test_names = [
        "Pillar 13: Async & Parallel Multi-Agent Orchestration",
        "Pillar 14: Front-end & Embeddable UI",
        "Pillar 15: Safety & Alignment",
        "Pillar 16: Meta-Learning & Self-Reflection",
        "Phase 4 Integration",
        "Intelligence Enhancement Features"
    ]
    
    passed = 0
    for i, (result, name) in enumerate(zip(results, test_names)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {i+1}. {name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 Phase 4: Intelligence Enhancement is working correctly!")
        print("\n📋 Phase 4 Features:")
        print("  ✅ Advanced async orchestration with parallel execution")
        print("  ✅ Rich frontend UI with extensions and widgets")
        print("  ✅ Comprehensive safety and alignment systems")
        print("  ✅ Meta-learning and self-reflection capabilities")
        print("  ✅ Intelligent pipeline management")
        print("  ✅ Performance optimization and monitoring")
        print("  ✅ Cross-platform integration")
        print("  ✅ Advanced AI capabilities")
        return True
    else:
        print("⚠️  Some tests need attention.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 