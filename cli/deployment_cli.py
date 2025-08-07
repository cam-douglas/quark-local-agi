#!/usr/bin/env python3
"""
Deployment CLI for Quark AI Assistant
Handles Docker, Kubernetes, and cloud deployments
"""

import click
import subprocess
import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

@click.group()
def main():
    """Quark AI Assistant Deployment CLI"""
    pass

@main.group()
def docker():
    """Docker deployment commands"""
    pass

@docker.command()
@click.option('--tag', default='latest', help='Docker image tag')
@click.option('--no-cache', is_flag=True, help='Build without cache')
@click.option('--platform', default='linux/amd64', help='Target platform')
def build(tag: str, no_cache: bool, platform: str):
    """Build Docker image"""
    try:
        cmd = ['docker', 'build']
        if no_cache:
            cmd.append('--no-cache')
        cmd.extend(['--platform', platform, '-t', f'quark-local-agi:{tag}', '.'])
        
        click.echo(f"Building Docker image with tag: {tag}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        click.echo("✅ Docker image built successfully")
        
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Docker build failed: {e.stderr}")
        sys.exit(1)

@docker.command()
@click.option('--detach', '-d', is_flag=True, help='Run in background')
@click.option('--port', '-p', default='8000:8000', help='Port mapping')
@click.option('--env-file', help='Environment file')
def run(detach: bool, port: str, env_file: Optional[str]):
    """Run Docker container"""
    try:
        cmd = ['docker', 'run']
        if detach:
            cmd.append('-d')
        cmd.extend(['-p', port])
        
        if env_file:
            cmd.extend(['--env-file', env_file])
        
        cmd.append('quark-local-agi:latest')
        
        click.echo("Starting Docker container...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        click.echo("✅ Docker container started successfully")
        
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Docker run failed: {e.stderr}")
        sys.exit(1)

@docker.command()
def compose():
    """Run with Docker Compose"""
    try:
        click.echo("Starting services with Docker Compose...")
        subprocess.run(['docker-compose', 'up', '-d'], check=True)
        click.echo("✅ Services started successfully")
        
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Docker Compose failed: {e}")
        sys.exit(1)

@docker.command()
def stop():
    """Stop Docker services"""
    try:
        click.echo("Stopping Docker services...")
        subprocess.run(['docker-compose', 'down'], check=True)
        click.echo("✅ Services stopped successfully")
        
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Docker stop failed: {e}")
        sys.exit(1)

@main.group()
def kubernetes():
    """Kubernetes deployment commands"""
    pass

@kubernetes.command()
@click.option('--namespace', default='meta-model', help='Kubernetes namespace')
@click.option('--replicas', default=3, help='Number of replicas')
def deploy(namespace: str, replicas: int):
    """Deploy to Kubernetes"""
    try:
        # Create namespace
        subprocess.run(['kubectl', 'create', 'namespace', namespace, '--dry-run=client', '-o', 'yaml'], 
                      stdout=subprocess.PIPE, check=True)
        subprocess.run(['kubectl', 'apply', '-f', '-'], 
                      input=subprocess.run(['kubectl', 'create', 'namespace', namespace, '--dry-run=client', '-o', 'yaml'], 
                                         capture_output=True, text=True).stdout, text=True)
        
        # Apply Kubernetes manifests
        k8s_dir = Path('deployment/kubernetes')
        for manifest in k8s_dir.glob('*.yaml'):
            click.echo(f"Applying {manifest.name}...")
            subprocess.run(['kubectl', 'apply', '-f', str(manifest), '-n', namespace], check=True)
        
        # Scale deployment
        subprocess.run(['kubectl', 'scale', 'deployment', 'meta-model', '--replicas', str(replicas), '-n', namespace], check=True)
        
        click.echo("✅ Kubernetes deployment successful")
        
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Kubernetes deployment failed: {e}")
        sys.exit(1)

@kubernetes.command()
@click.option('--namespace', default='meta-model', help='Kubernetes namespace')
def status(namespace: str):
    """Check Kubernetes deployment status"""
    try:
        click.echo(f"Checking deployment status in namespace: {namespace}")
        
        # Check pods
        result = subprocess.run(['kubectl', 'get', 'pods', '-n', namespace], 
                              capture_output=True, text=True, check=True)
        click.echo(result.stdout)
        
        # Check services
        result = subprocess.run(['kubectl', 'get', 'services', '-n', namespace], 
                              capture_output=True, text=True, check=True)
        click.echo(result.stdout)
        
        # Check ingress
        result = subprocess.run(['kubectl', 'get', 'ingress', '-n', namespace], 
                              capture_output=True, text=True, check=True)
        click.echo(result.stdout)
        
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Status check failed: {e}")
        sys.exit(1)

@kubernetes.command()
@click.option('--namespace', default='meta-model', help='Kubernetes namespace')
def logs(namespace: str):
    """Get Kubernetes logs"""
    try:
        click.echo(f"Getting logs from namespace: {namespace}")
        subprocess.run(['kubectl', 'logs', '-f', 'deployment/meta-model', '-n', namespace])
        
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Log retrieval failed: {e}")
        sys.exit(1)

@kubernetes.command()
@click.option('--namespace', default='meta-model', help='Kubernetes namespace')
def delete(namespace: str):
    """Delete Kubernetes deployment"""
    try:
        click.echo(f"Deleting deployment in namespace: {namespace}")
        
        # Delete all resources in namespace
        subprocess.run(['kubectl', 'delete', 'all', '--all', '-n', namespace], check=True)
        
        # Delete namespace
        subprocess.run(['kubectl', 'delete', 'namespace', namespace], check=True)
        
        click.echo("✅ Kubernetes deployment deleted")
        
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Kubernetes deletion failed: {e}")
        sys.exit(1)

@main.group()
def monitoring():
    """Monitoring and observability commands"""
    pass

@monitoring.command()
def setup():
    """Setup monitoring stack"""
    try:
        click.echo("Setting up monitoring stack...")
        
        # Start Prometheus
        subprocess.run(['docker-compose', '-f', 'deployment/docker-compose.monitoring.yml', 'up', '-d'], check=True)
        
        click.echo("✅ Monitoring stack started")
        click.echo("Prometheus: http://localhost:9090")
        click.echo("Grafana: http://localhost:3000")
        
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Monitoring setup failed: {e}")
        sys.exit(1)

@monitoring.command()
def status():
    """Check monitoring status"""
    try:
        click.echo("Checking monitoring services...")
        
        # Check Prometheus
        result = subprocess.run(['curl', '-s', 'http://localhost:9090/-/healthy'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            click.echo("✅ Prometheus: Healthy")
        else:
            click.echo("❌ Prometheus: Unhealthy")
        
        # Check Grafana
        result = subprocess.run(['curl', '-s', 'http://localhost:3000/api/health'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            click.echo("✅ Grafana: Healthy")
        else:
            click.echo("❌ Grafana: Unhealthy")
        
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Monitoring status check failed: {e}")
        sys.exit(1)

@main.command()
def health_check():
    """Perform comprehensive health check"""
    try:
        click.echo("Performing health check...")
        
        # Check Docker
        result = subprocess.run(['docker', 'version'], capture_output=True)
        if result.returncode == 0:
            click.echo("✅ Docker: Available")
        else:
            click.echo("❌ Docker: Not available")
        
        # Check Kubernetes
        result = subprocess.run(['kubectl', 'version', '--client'], capture_output=True)
        if result.returncode == 0:
            click.echo("✅ Kubernetes: Available")
        else:
            click.echo("❌ Kubernetes: Not available")
        
        # Check Python dependencies
        try:
            import transformers
            import fastapi
            import chromadb
            click.echo("✅ Python dependencies: Available")
        except ImportError as e:
            click.echo(f"❌ Python dependencies: Missing - {e}")
        
        # Check configuration files
        config_files = [
            'config/pyproject.toml',
            'config/Dockerfile',
            'docker-compose.yml',
            'deployment/kubernetes/meta-model-deployment.yaml'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                click.echo(f"✅ {config_file}: Available")
            else:
                click.echo(f"❌ {config_file}: Missing")
        
        click.echo("✅ Health check completed")
        
    except Exception as e:
        click.echo(f"❌ Health check failed: {e}")
        sys.exit(1)

@main.command()
@click.option('--format', default='json', type=click.Choice(['json', 'yaml', 'text']))
def config(format: str):
    """Show deployment configuration"""
    try:
        config = {
            "docker": {
                "enabled": True,
                "image": "quark-local-agi:latest",
                "ports": ["8000:8000"]
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
            },
            "services": {
                "redis": True,
                "chromadb": True,
                "meta-model": True
            }
        }
        
        if format == 'json':
            click.echo(json.dumps(config, indent=2))
        elif format == 'yaml':
            click.echo(yaml.dump(config, default_flow_style=False))
        else:
            for section, settings in config.items():
                click.echo(f"\n{section.upper()}:")
                for key, value in settings.items():
                    click.echo(f"  {key}: {value}")
        
    except Exception as e:
        click.echo(f"❌ Config display failed: {e}")
        sys.exit(1)

@main.command()
@click.option('--environment', default='production', type=click.Choice(['development', 'staging', 'production']))
def deploy(environment: str):
    """Deploy to specified environment"""
    try:
        click.echo(f"Deploying to {environment} environment...")
        
        # Set environment variables
        os.environ['META_MODEL_ENV'] = environment
        
        if environment == 'development':
            # Local development
            subprocess.run(['docker-compose', 'up', '-d'], check=True)
            click.echo("✅ Development deployment complete")
            
        elif environment == 'staging':
            # Staging deployment
            subprocess.run(['docker', 'build', '-t', 'quark-local-agi:staging', '.'], check=True)
            subprocess.run(['kubectl', 'apply', '-f', 'deployment/kubernetes/', '-n', 'meta-model-staging'], check=True)
            click.echo("✅ Staging deployment complete")
            
        elif environment == 'production':
            # Production deployment
            subprocess.run(['docker', 'build', '-t', 'quark-local-agi:latest', '.'], check=True)
            subprocess.run(['kubectl', 'apply', '-f', 'deployment/kubernetes/', '-n', 'meta-model'], check=True)
            subprocess.run(['kubectl', 'scale', 'deployment', 'meta-model', '--replicas', '5', '-n', 'meta-model'], check=True)
            click.echo("✅ Production deployment complete")
        
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 