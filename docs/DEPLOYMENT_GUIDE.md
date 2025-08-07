# Deployment Guide

## Overview

This guide covers deploying the Quark AI Assistant to various environments including local development, Docker containers, Kubernetes clusters, and cloud platforms.

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -e .

# Start the AI assistant
meta-model

# Or start with web interface
meta-model --web --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
# Build Docker image
meta-deploy build-docker --tag v1.0.0

# Run with Docker Compose
meta-deploy run-docker --detach

# Check status
meta-deploy health-check
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
meta-deploy deploy-kubernetes --namespace meta-model

# Scale deployment
meta-deploy scale-kubernetes --namespace meta-model

# Check status
meta-deploy status-kubernetes --namespace meta-model
```

## Docker Deployment

### Building the Image

```bash
# Basic build
docker build -t meta-model-ai:latest .

# Multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 -t meta-model-ai:latest .

# Build with specific tag
docker build -t meta-model-ai:v1.0.0 .
```

### Docker Compose

The `docker-compose.yml` file includes:

- **Quark AI Assistant**: Main application
- **Redis**: Caching and session management
- **ChromaDB**: Vector database for memory
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboard
- **Nginx**: Load balancer and reverse proxy

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f meta-model

# Stop services
docker-compose down
```

### Environment Variables

```bash
# Core settings
META_MODEL_ENV=production
META_MODEL_SAFETY_ENABLED=true
META_MODEL_MEMORY_PATH=/app/memory_db
META_MODEL_LOG_LEVEL=INFO

# Cloud integration
META_MODEL_CLOUD_ENABLED=true
META_MODEL_WEB_BROWSER_ENABLED=true

# Database connections
REDIS_URL=redis://redis:6379
CHROMADB_HOST=chromadb
CHROMADB_PORT=8000
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- Helm (optional)

### Deployment Steps

1. **Create namespace**:
```bash
kubectl create namespace meta-model
```

2. **Apply Kubernetes manifests**:
```bash
kubectl apply -f deployment/kubernetes/meta-model-deployment.yaml -n meta-model
kubectl apply -f deployment/kubernetes/ingress.yaml -n meta-model
kubectl apply -f deployment/kubernetes/hpa.yaml -n meta-model
```

3. **Verify deployment**:
```bash
kubectl get pods -n meta-model
kubectl get services -n meta-model
kubectl get ingress -n meta-model
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment meta-model-ai --replicas=5 -n meta-model

# Using HPA (automatic scaling)
kubectl get hpa -n meta-model
```

### Monitoring

The deployment includes:

- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards
- **Custom metrics**: AI-specific metrics

Access monitoring:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

## Cloud Deployment

### Google Cloud Platform (GCP)

```bash
# Setup GCP infrastructure
meta-deploy setup-cloud --platform gcp --region us-central1

# Deploy to GKE
kubectl apply -f deployment/kubernetes/ -n meta-model
```

### Amazon Web Services (AWS)

```bash
# Setup AWS infrastructure
meta-deploy setup-cloud --platform aws --region us-west-2

# Deploy to EKS
kubectl apply -f deployment/kubernetes/ -n meta-model
```

### Azure

```bash
# Setup Azure infrastructure
meta-deploy setup-cloud --platform azure --region eastus

# Deploy to AKS
kubectl apply -f deployment/kubernetes/ -n meta-model
```

## Configuration

### Application Configuration

Create `config/settings.json`:

```json
{
  "safety": {
    "enabled": true,
    "immutable_rules": true
  },
  "memory": {
    "max_memories": 10000,
    "eviction_policy": "time_based"
  },
  "cloud": {
    "enabled": true,
    "providers": ["google_colab", "huggingface_spaces"]
  },
  "web_browser": {
    "enabled": true,
    "rate_limit": 10
  }
}
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `META_MODEL_ENV` | Environment (dev/prod) | `development` |
| `META_MODEL_SAFETY_ENABLED` | Enable safety features | `true` |
| `META_MODEL_MEMORY_PATH` | Memory database path | `./memory_db` |
| `META_MODEL_LOG_LEVEL` | Logging level | `INFO` |
| `META_MODEL_CLOUD_ENABLED` | Enable cloud integration | `true` |
| `META_MODEL_WEB_BROWSER_ENABLED` | Enable web browsing | `true` |

## Monitoring & Observability

### Metrics

The application exposes Prometheus metrics at `/metrics`:

- **Request rate**: `http_requests_total`
- **Response time**: `http_request_duration_seconds`
- **Memory usage**: `quark_memory_usage_bytes`
- **Active connections**: `quark_active_connections`
- **Model performance**: `quark_inference_duration_seconds`

### Logging

```bash
# View application logs
docker-compose logs -f meta-model

# View Kubernetes logs
kubectl logs -f deployment/meta-model-ai -n meta-model
```

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Comprehensive health check
meta-deploy health-check
```

## Security

### Docker Security

- Non-root user execution
- Read-only filesystem where possible
- Minimal base image
- Security scanning with tools like Trivy

### Kubernetes Security

- Pod Security Standards
- Network policies
- RBAC configuration
- Secrets management

### Network Security

- TLS/SSL termination
- Rate limiting
- CORS configuration
- Security headers

## Troubleshooting

### Common Issues

1. **Container won't start**:
   ```bash
   docker logs meta-model-ai
   ```

2. **Kubernetes pods in pending**:
   ```bash
   kubectl describe pod <pod-name> -n meta-model
   ```

3. **High memory usage**:
   - Check memory limits
   - Monitor memory metrics
   - Adjust resource requests

4. **Slow response times**:
   - Check CPU usage
   - Monitor network latency
   - Verify model loading

### Debug Commands

```bash
# Check container status
docker ps -a

# Check Kubernetes resources
kubectl get all -n meta-model

# Check service endpoints
kubectl get endpoints -n meta-model

# Check ingress status
kubectl describe ingress meta-model-ingress -n meta-model
```

## Performance Optimization

### Resource Allocation

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "500m"
  limits:
    memory: "8Gi"
    cpu: "2000m"
```

### Scaling Strategies

- **Horizontal Pod Autoscaler**: Automatic scaling based on CPU/memory
- **Vertical Pod Autoscaler**: Automatic resource adjustment
- **Cluster Autoscaler**: Node-level scaling

### Caching

- Redis for session management
- Model caching
- Response caching
- Memory database optimization

## Backup & Recovery

### Data Backup

```bash
# Backup memory database
docker exec meta-model-ai tar czf /backup/memory_$(date +%Y%m%d).tar.gz /app/memory_db

# Backup Kubernetes volumes
kubectl exec -n meta-model <pod-name> -- tar czf /backup/memory.tar.gz /app/memory_db
```

### Disaster Recovery

1. **Backup strategy**: Regular backups of persistent volumes
2. **Recovery procedures**: Documented recovery steps
3. **Testing**: Regular recovery testing
4. **Monitoring**: Backup success monitoring

## Support

For deployment issues:

1. Check the logs: `docker logs` or `kubectl logs`
2. Verify configuration: Check environment variables and config files
3. Test connectivity: Use health check endpoints
4. Review monitoring: Check metrics and dashboards
5. Consult documentation: Review this guide and API documentation

## Next Steps

After successful deployment:

1. **Configure monitoring**: Set up alerts and dashboards
2. **Implement CI/CD**: Automate deployment pipeline
3. **Security audit**: Review security configuration
4. **Performance tuning**: Optimize based on usage patterns
5. **Backup strategy**: Implement regular backups 