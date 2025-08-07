#!/usr/bin/env python3
"""
Test Model Development Integration
Comprehensive testing of all implemented model development components
"""

import sys
import os
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any

def test_model_scoping():
    """Test the model scoping functionality"""
    print("🔍 Testing Model Scoping...")
    
    try:
        # Add current directory to Python path
        sys.path.insert(0, os.getcwd())
        from core.model_scoping import ModelScoper
        
        scoper = ModelScoper()
        report = scoper.generate_scope_report()
        
        # Validate report structure
        required_keys = ['timestamp', 'summary', 'requirements', 'resources', 'recommendations']
        for key in required_keys:
            if key not in report:
                print(f"❌ Missing key in scope report: {key}")
                return False
        
        # Validate summary
        summary = report['summary']
        if summary['total_requirements'] > 0:
            print(f"✅ Scope report generated successfully")
            print(f"   - Requirements: {summary['total_requirements']}")
            print(f"   - High Priority: {summary['high_priority']}")
            print(f"   - Budget: ${summary['estimated_budget']}")
            return True
        else:
            print("❌ No requirements found in scope report")
            return False
            
    except Exception as e:
        print(f"❌ Error testing model scoping: {e}")
        return False

def test_architecture_selector():
    """Test the architecture selector functionality"""
    print("🏗️ Testing Architecture Selector...")
    
    try:
        # Add current directory to Python path
        sys.path.insert(0, os.getcwd())
        from model_development.architecture_selector import ArchitectureSelector
        
        selector = ArchitectureSelector()
        
        # Test architecture selection
        requirements = {
            'use_cases': ['conversational_qa', 'code_assistance'],
            'compute_budget': 'medium',
            'latency_requirements': 'medium',
            'data_availability': 'high'
        }
        
        arch_name, arch_spec = selector.select_architecture(requirements)
        
        if arch_name and arch_spec:
            print(f"✅ Architecture selection successful")
            print(f"   - Selected: {arch_name}")
            print(f"   - Type: {arch_spec.type.value}")
            print(f"   - Parameters: {arch_spec.parameters:,}")
            return True
        else:
            print("❌ Architecture selection failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing architecture selector: {e}")
        return False

async def test_web_crawler():
    """Test the web crawler functionality"""
    print("🕷️ Testing Web Crawler...")
    
    try:
        # Add current directory to Python path
        sys.path.insert(0, os.getcwd())
        from data_collection.web_crawler import WebCrawler
        
        # Test with a simple URL
        test_urls = ["https://en.wikipedia.org/wiki/Artificial_intelligence"]
        
        async with WebCrawler(max_concurrent=1, request_delay=0.1) as crawler:
            results = await crawler.crawl_urls(test_urls)
            
            if results and len(results) > 0:
                successful = [r for r in results if r.status == 'success']
                print(f"✅ Web crawler test successful")
                print(f"   - Total results: {len(results)}")
                print(f"   - Successful: {len(successful)}")
                return True
            else:
                print("❌ Web crawler test failed - no results")
                return False
                
    except Exception as e:
        print(f"❌ Error testing web crawler: {e}")
        return False

def test_configuration_files():
    """Test that configuration files exist and are valid"""
    print("📋 Testing Configuration Files...")
    
    config_files = [
        "config/model_planning.yml",
        "core/model_scoping.py",
        "docs/model_requirements.md",
        "scripts/model_assessment.py"
    ]
    
    all_exist = True
    for file_path in config_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            all_exist = False
    
    return all_exist

def test_directory_structure():
    """Test that all required directories exist"""
    print("📁 Testing Directory Structure...")
    
    required_dirs = [
        "data_collection",
        "model_development", 
        "fine_tuning",
        "retrieval",
        "orchestration",
        "optimization",
        "evaluation",
        "continuous_improvement"
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ - Missing")
            all_exist = False
    
    return all_exist

def test_integration_script():
    """Test the integration script functionality"""
    print("🔧 Testing Integration Script...")
    
    try:
        # Add current directory to Python path
        sys.path.insert(0, os.getcwd())
        from scripts.integrate_model_development import ModelDevelopmentIntegrator
        
        integrator = ModelDevelopmentIntegrator()
        
        # Test that the integrator can be instantiated
        if integrator:
            print("✅ Integration script loads successfully")
            return True
        else:
            print("❌ Integration script failed to load")
            return False
            
    except Exception as e:
        print(f"❌ Error testing integration script: {e}")
        return False

def test_documentation():
    """Test that documentation files exist"""
    print("📚 Testing Documentation...")
    
    doc_files = [
        "docs/MODEL_DEVELOPMENT_ROADMAP.md",
        "docs/MODEL_DEVELOPMENT_IMPLEMENTATION.md",
        "docs/PILLAR_MODEL_DEVELOPMENT_MAPPING.md",
        "docs/MODEL_DEVELOPMENT_QUICKSTART.md",
        "docs/MODEL_DEVELOPMENT_INTEGRATION_SUMMARY.md",
        "MODEL_DEVELOPMENT_INTEGRATION_STATUS.md"
    ]
    
    all_exist = True
    for file_path in doc_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            all_exist = False
    
    return all_exist

async def run_all_tests():
    """Run all integration tests"""
    print("🚀 Running Quark Model Development Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Configuration Files", test_configuration_files),
        ("Directory Structure", test_directory_structure),
        ("Integration Script", test_integration_script),
        ("Documentation", test_documentation),
        ("Model Scoping", test_model_scoping),
        ("Architecture Selector", test_architecture_selector),
        ("Web Crawler", test_web_crawler)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Model development integration is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
        return False

def main():
    """Main test function"""
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 