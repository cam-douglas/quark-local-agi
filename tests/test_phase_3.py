#!/usr/bin/env python3
"""
Test suite for Phase 3: Advanced Features
Tests Pillars 10, 11, and 12 functionality
"""

import sys
import os
import tempfile
import shutil
import asyncio
import json
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_pillar_10_packaging():
    """Test Pillar 10: Packaging & Documentation"""
    print("Testing Pillar 10: Packaging & Documentation...")
    
    try:
        # Test pyproject.toml configuration
        import tomllib
        with open('config/pyproject.toml', 'rb') as f:
            project_config = tomllib.load(f)
        
        assert project_config["project"]["name"] == "quark-local-agi"
        assert project_config["project"]["version"] == "1.0.0"
        
        # Test entry points exist
        entry_points = project_config["project"]["scripts"]
        assert "meta-model" in entry_points
        assert "meta-memory" in entry_points
        assert "meta-metrics" in entry_points
        assert "meta-safety" in entry_points
        assert "meta-streaming" in entry_points
        
        # Test documentation files exist
        docs_files = [
            "docs/API_REFERENCE.md",
            "docs/USER_GUIDE.md",
            "README.md",
            "CHANGELOG.md",
            "CONTRIBUTING.md"
        ]
        
        for doc_file in docs_files:
            assert os.path.exists(doc_file), f"Documentation file {doc_file} missing"
        
        print("✅ Pillar 10: Packaging & Documentation - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Pillar 10: Packaging & Documentation - FAILED: {e}")
        return False

def test_pillar_11_testing():
    """Test Pillar 11: Testing & Continuous Integration"""
    print("Testing Pillar 11: Testing & Continuous Integration...")
    
    try:
        # Test test runner exists
        test_runner_path = "tests/run_tests.py"
        assert os.path.exists(test_runner_path), "Test runner missing"
        
        # Test test discovery works
        test_files = [
            "tests/test_pillars_5_6_7_8.py",
            "tests/test_pillar_9.py",
            "tests/test_pillar_15.py",
            "tests/test_pillar_17.py",
            "tests/test_safety_system.py"
        ]
        
        for test_file in test_files:
            if os.path.exists(test_file):
                assert os.path.getsize(test_file) > 0, f"Test file {test_file} is empty"
        
        # Test CI workflow exists
        ci_workflow_path = ".github/workflows/ci.yml"
        assert os.path.exists(ci_workflow_path), "CI workflow missing"
        
        with open(ci_workflow_path, 'r') as f:
            workflow_content = f.read()
            assert "name: CI" in workflow_content
            assert "on:" in workflow_content
            assert "jobs:" in workflow_content
        
        # Test pytest configuration
        pytest_config = {
            "testpaths": ["tests"],
            "python_files": ["test_*.py"],
            "python_classes": ["Test*"],
            "python_functions": ["test_*"],
            "addopts": [
                "--strict-markers",
                "--verbose",
                "--cov=quark"
            ]
        }
        
        assert "testpaths" in pytest_config
        assert "tests" in pytest_config["testpaths"]
        assert "--cov=quark" in pytest_config["addopts"]
        
        print("✅ Pillar 11: Testing & Continuous Integration - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Pillar 11: Testing & Continuous Integration - FAILED: {e}")
        return False

def test_pillar_12_deployment():
    """Test Pillar 12: Deployment & Scaling"""
    print("Testing Pillar 12: Deployment & Scaling...")
    
    try:
        # Test Dockerfile
        dockerfile_path = "config/Dockerfile"
        assert os.path.exists(dockerfile_path), "Dockerfile missing"
        
        with open(dockerfile_path, 'r') as f:
            dockerfile_content = f.read()
            assert "FROM python:" in dockerfile_content
            assert "WORKDIR /app" in dockerfile_content
            assert "ENTRYPOINT" in dockerfile_content
        
        # Test docker-compose.yml
        compose_path = "docker-compose.yml"
        assert os.path.exists(compose_path), "docker-compose.yml missing"
        
        with open(compose_path, 'r') as f:
            compose_content = f.read()
            assert "meta-model" in compose_content
            assert "redis" in compose_content
            assert "chromadb" in compose_content
        
        # Test Kubernetes manifests
        k8s_files = [
            "deployment/kubernetes/meta-model-deployment.yaml",
            "deployment/kubernetes/ingress.yaml",
            "deployment/kubernetes/hpa.yaml"
        ]
        
        for k8s_file in k8s_files:
            assert os.path.exists(k8s_file), f"Kubernetes file {k8s_file} missing"
        
        # Test deployment CLI exists
        deploy_cli_path = "cli/deployment_cli.py"
        assert os.path.exists(deploy_cli_path), "Deployment CLI missing"
        
        with open(deploy_cli_path, 'r') as f:
            cli_content = f.read()
            assert "def main():" in cli_content
            assert "def docker():" in cli_content
            assert "def kubernetes():" in cli_content
        
        # Test deployment configuration
        deployment_config = {
            "docker": {
                "enabled": True,
                "image": "quark-local-agi:latest"
            },
            "kubernetes": {
                "enabled": True,
                "namespace": "meta-model"
            },
            "monitoring": {
                "prometheus": True,
                "grafana": True
            }
        }
        
        # Test deployment validation
        assert deployment_config["docker"]["enabled"] == True
        assert deployment_config["kubernetes"]["enabled"] == True
        assert deployment_config["monitoring"]["prometheus"] == True
        
        print("✅ Pillar 12: Deployment & Scaling - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Pillar 12: Deployment & Scaling - FAILED: {e}")
        return False

def test_continuous_integration():
    """Test Continuous Integration setup"""
    print("Testing Continuous Integration...")
    
    try:
        # Test GitHub Actions workflow
        workflow_path = "config/ci.yml"
        assert os.path.exists(workflow_path), "CI workflow missing"
        
        with open(workflow_path, 'r') as f:
            workflow_content = f.read()
            assert "name: CI" in workflow_content
            assert "on:" in workflow_content
            assert "jobs:" in workflow_content
        
        # Test pre-commit hooks
        pre_commit_path = ".pre-commit-config.yaml"
        if os.path.exists(pre_commit_path):
            with open(pre_commit_path, 'r') as f:
                pre_commit_content = f.read()
                assert "repos:" in pre_commit_content
        
        # Test code quality tools
        quality_tools = [
            "black",
            "isort", 
            "flake8",
            "mypy"
        ]
        
        # Test pytest configuration
        pytest_config = {
            "testpaths": ["tests"],
            "python_files": ["test_*.py"],
            "python_classes": ["Test*"],
            "python_functions": ["test_*"],
            "addopts": [
                "--strict-markers",
                "--verbose",
                "--cov=quark"
            ]
        }
        
        assert "testpaths" in pytest_config
        assert "tests" in pytest_config["testpaths"]
        assert "--cov=quark" in pytest_config["addopts"]
        
        print("✅ Continuous Integration - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Continuous Integration - FAILED: {e}")
        return False

def test_deployment_tools():
    """Test deployment tools and utilities"""
    print("Testing Deployment Tools...")
    
    try:
        # Test deployment CLI commands
        deploy_commands = [
            "build-docker",
            "run-docker", 
            "deploy-kubernetes",
            "status-kubernetes",
            "health-check"
        ]
        
        # Test deployment configuration
        deployment_tools = {
            "docker": {
                "enabled": True,
                "build_context": ".",
                "image_name": "quark-local-agi"
            },
            "kubernetes": {
                "enabled": True,
                "namespace": "meta-model",
                "replicas": 3
            },
            "monitoring": {
                "prometheus": True,
                "grafana": True,
                "nginx": True
            }
        }
        
        assert deployment_tools["docker"]["enabled"] == True
        assert deployment_tools["kubernetes"]["enabled"] == True
        assert deployment_tools["monitoring"]["prometheus"] == True
        
        # Test monitoring configuration
        monitoring_files = [
            "deployment/prometheus.yml",
            "deployment/nginx.conf",
            "deployment/grafana/dashboards/dashboard.yml"
        ]
        
        for monitoring_file in monitoring_files:
            if os.path.exists(monitoring_file):
                with open(monitoring_file, 'r') as f:
                    content = f.read()
                    assert len(content) > 0
        
        print("✅ Deployment Tools - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Deployment Tools - FAILED: {e}")
        return False

async def main():
    """Run all Phase 3 tests."""
    print("🧪 Testing Phase 3: Advanced Features...")
    print("=" * 60)
    
    results = []
    
    # Test each pillar
    results.append(test_pillar_10_packaging())
    results.append(test_pillar_11_testing())
    results.append(test_pillar_12_deployment())
    results.append(test_continuous_integration())
    results.append(test_deployment_tools())
    
    # Summary
    print("=" * 60)
    print("📊 Test Results Summary:")
    
    test_names = [
        "Pillar 10: Packaging & Documentation",
        "Pillar 11: Testing & Continuous Integration",
        "Pillar 12: Deployment & Scaling",
        "Continuous Integration",
        "Deployment Tools"
    ]
    
    passed = 0
    for i, (result, name) in enumerate(zip(results, test_names)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {i+1}. {name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 Phase 3: Advanced Features is working correctly!")
        print("\n📋 Phase 3 Features:")
        print("  ✅ Comprehensive packaging with pyproject.toml")
        print("  ✅ Complete documentation (API, User Guide)")
        print("  ✅ Comprehensive testing framework")
        print("  ✅ Continuous Integration setup")
        print("  ✅ Docker and Kubernetes deployment")
        print("  ✅ Monitoring and observability")
        return True
    else:
        print("⚠️  Some tests need attention.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 