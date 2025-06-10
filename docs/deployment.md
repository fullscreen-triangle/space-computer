---
layout: default
title: "Deployment Guide"
description: "Complete guide for deploying Space Computer infrastructure"
show_toc: true
show_navigation: true
---

# Deployment Guide

This guide covers everything needed to deploy Space Computer in production environments, from single-instance setups to enterprise-scale distributed deployments.

## 🏗️ **Architecture Overview**

### **Deployment Patterns**
- **Single Instance**: Development and small-scale testing
- **Microservices**: Production-ready scalable deployment
- **Enterprise**: High-availability, multi-region setup
- **Edge**: Distributed processing for low-latency analysis

### **Infrastructure Components**
```yaml
Core_Services:
  - frontend: "Space Computer React App"
  - backend: "AI Processing Pipeline"
  - orchestration: "Meta-Orchestrator"
  - database: "PostgreSQL + Redis"
  - storage: "S3-compatible object storage"
  - queue: "Redis/RabbitMQ for async processing"

Supporting_Services:
  - monitoring: "Prometheus + Grafana"
  - logging: "ELK Stack"
  - auth: "Auth0/Keycloak"
  - cdn: "CloudFlare/AWS CloudFront"
  - load_balancer: "Nginx/HAProxy"
```

## 🚀 **Quick Start Deployment**

### **Docker Compose (Development)**

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  frontend:
    image: spacecomputer/frontend:latest
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - backend

  backend:
    image: spacecomputer/backend:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/spacecomputer
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    depends_on:
      - postgres
      - redis
    volumes:
      - ./models:/app/models
      - ./uploads:/app/uploads

  orchestration:
    image: spacecomputer/orchestration:latest
    environment:
      - BACKEND_URL=http://backend:8000
      - REDIS_URL=redis://redis:6379
    depends_on:
      - backend
      - redis

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=spacecomputer
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  worker:
    image: spacecomputer/worker:latest
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@postgres:5432/spacecomputer
    depends_on:
      - postgres
      - redis
    deploy:
      replicas: 2

volumes:
  postgres_data:
  redis_data:
```

**Deploy**:
```bash
# Set environment variables
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend
```

### **Environment Variables**
```bash
# Core Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/spacecomputer
REDIS_URL=redis://localhost:6379
SECRET_KEY=your_secret_key_here

# AI Model APIs
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
HUGGINGFACE_API_KEY=your_huggingface_key

# Storage Configuration
S3_BUCKET=spacecomputer-storage
S3_ACCESS_KEY=your_access_key
S3_SECRET_KEY=your_secret_key
S3_ENDPOINT=https://s3.amazonaws.com

# Monitoring
SENTRY_DSN=https://your_sentry_dsn
PROMETHEUS_GATEWAY=http://prometheus:9091
```

## ☁️ **Cloud Deployments**

### **AWS Deployment**

#### **ECS with Fargate**
```yaml
# ecs-task-definition.json
{
  "family": "spacecomputer",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "spacecomputer/backend:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql://user:pass@rds-endpoint:5432/spacecomputer"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/spacecomputer",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### **Terraform Configuration**
```hcl
# main.tf
provider "aws" {
  region = var.aws_region
}

# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "spacecomputer-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["us-west-2a", "us-west-2b", "us-west-2c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = true
}

# RDS Database
resource "aws_db_instance" "spacecomputer" {
  identifier = "spacecomputer-db"
  
  engine         = "postgres"
  engine_version = "15.3"
  instance_class = "db.r6g.large"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_encrypted     = true
  
  db_name  = "spacecomputer"
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.spacecomputer.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "spacecomputer-final-snapshot"
}

# ElastiCache Redis
resource "aws_elasticache_replication_group" "spacecomputer" {
  replication_group_id         = "spacecomputer-redis"
  description                  = "Redis cluster for Space Computer"
  
  node_type                    = "cache.r6g.large"
  port                         = 6379
  parameter_group_name         = "default.redis7"
  
  num_cache_clusters           = 2
  automatic_failover_enabled   = true
  multi_az_enabled            = true
  
  subnet_group_name = aws_elasticache_subnet_group.spacecomputer.name
  security_group_ids = [aws_security_group.redis.id]
}

# ECS Cluster
resource "aws_ecs_cluster" "spacecomputer" {
  name = "spacecomputer"
  
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]
  
  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight           = 1
  }
}

# Application Load Balancer
resource "aws_lb" "spacecomputer" {
  name               = "spacecomputer-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = module.vpc.public_subnets
  
  enable_deletion_protection = true
}
```

### **Google Cloud Platform**

#### **GKE Deployment**
```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spacecomputer-backend
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: spacecomputer-backend
  template:
    metadata:
      labels:
        app: spacecomputer-backend
    spec:
      containers:
      - name: backend
        image: gcr.io/your-project/spacecomputer-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: spacecomputer-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: spacecomputer-secrets
              key: redis-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: spacecomputer-backend-service
  namespace: production
spec:
  selector:
    app: spacecomputer-backend
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

#### **Helm Chart**
```yaml
# helm/values.yaml
global:
  registry: gcr.io/your-project
  tag: latest

backend:
  replicaCount: 3
  image:
    repository: spacecomputer-backend
    pullPolicy: Always
  
  service:
    type: ClusterIP
    port: 80
    targetPort: 8000
  
  resources:
    limits:
      cpu: 2000m
      memory: 4Gi
    requests:
      cpu: 1000m
      memory: 2Gi
  
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70

frontend:
  replicaCount: 2
  image:
    repository: spacecomputer-frontend
    pullPolicy: Always
  
  service:
    type: ClusterIP
    port: 80
    targetPort: 3000

orchestration:
  replicaCount: 2
  image:
    repository: spacecomputer-orchestration
    pullPolicy: Always
  
  resources:
    limits:
      cpu: 1000m
      memory: 2Gi
    requests:
      cpu: 500m
      memory: 1Gi

workers:
  replicaCount: 5
  image:
    repository: spacecomputer-worker
    pullPolicy: Always
  
  resources:
    limits:
      cpu: 4000m
      memory: 8Gi
    requests:
      cpu: 2000m
      memory: 4Gi
  
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 20
    targetCPUUtilizationPercentage: 80

postgresql:
  enabled: true
  auth:
    postgresPassword: your-secure-password
    database: spacecomputer
  primary:
    persistence:
      enabled: true
      size: 100Gi
      storageClass: ssd

redis:
  enabled: true
  auth:
    enabled: true
    password: your-redis-password
  master:
    persistence:
      enabled: true
      size: 20Gi
```

## 🔐 **Security Configuration**

### **SSL/TLS Setup**
```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name api.space-computer.ai;
    
    ssl_certificate /etc/ssl/certs/space-computer.crt;
    ssl_certificate_key /etc/ssl/private/space-computer.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS;
    ssl_prefer_server_ciphers off;
    
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    
    location / {
        proxy_pass http://backend_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

upstream backend_servers {
    least_conn;
    server backend-1:8000 max_fails=3 fail_timeout=30s;
    server backend-2:8000 max_fails=3 fail_timeout=30s;
    server backend-3:8000 max_fails=3 fail_timeout=30s;
}
```

### **Authentication & Authorization**
```python
# security/auth.py
from functools import wraps
import jwt
from flask import request, jsonify, current_app

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            # Remove 'Bearer ' prefix
            token = token.split(' ')[1]
            payload = jwt.decode(
                token, 
                current_app.config['SECRET_KEY'], 
                algorithms=['HS256']
            )
            request.user = payload
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

def require_role(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.user.get('role') != role:
                return jsonify({'error': 'Insufficient permissions'}), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator
```

## 📊 **Monitoring & Observability**

### **Prometheus Configuration**
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'spacecomputer-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'spacecomputer-workers'
    static_configs:
      - targets: ['worker-1:9090', 'worker-2:9090', 'worker-3:9090']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### **Grafana Dashboards**
```json
{
  "dashboard": {
    "title": "Space Computer Overview",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Video Analysis Queue",
        "type": "singlestat",
        "targets": [
          {
            "expr": "redis_list_length{key=\"analysis_queue\"}",
            "legendFormat": "Queue Length"
          }
        ]
      },
      {
        "title": "AI Model Response Times",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ai_model_request_duration_seconds_sum[5m]) / rate(ai_model_request_duration_seconds_count[5m])",
            "legendFormat": "{{model_name}}"
          }
        ]
      }
    ]
  }
}
```

### **Application Logging**
```python
# logging/config.py
import logging
import structlog
from pythonjsonlogger import jsonlogger

def configure_logging():
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
```

## 🔄 **CI/CD Pipeline**

### **GitHub Actions**
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run tests
        run: |
          pytest tests/ --cov=src/ --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to Registry
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          push: true
          tags: |
            ghcr.io/${{ github.repository }}/backend:latest
            ghcr.io/${{ github.repository }}/backend:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to Kubernetes
        uses: azure/k8s-deploy@v1
        with:
          manifests: |
            k8s/deployment.yaml
            k8s/service.yaml
          images: |
            ghcr.io/${{ github.repository }}/backend:${{ github.sha }}
          kubectl-version: 'latest'
```

## 💾 **Database Management**

### **Database Migrations**
```python
# migrations/env.py
from alembic import context
from sqlalchemy import engine_from_config, pool
from models import Base

config = context.config
target_metadata = Base.metadata

def run_migrations():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )
        
        with context.begin_transaction():
            context.run_migrations()
```

### **Backup Strategy**
```bash
#!/bin/bash
# backup/backup.sh

# Database backup
pg_dump $DATABASE_URL | gzip > /backups/db_$(date +%Y%m%d_%H%M%S).sql.gz

# Upload to S3
aws s3 sync /backups/ s3://spacecomputer-backups/database/

# Cleanup old backups (keep 30 days)
find /backups -name "*.sql.gz" -mtime +30 -delete

# Redis backup
redis-cli --rdb /backups/redis_$(date +%Y%m%d_%H%M%S).rdb
aws s3 sync /backups/ s3://spacecomputer-backups/redis/
```

## 🚀 **Performance Optimization**

### **Scaling Strategies**
```yaml
# Auto-scaling configuration
Horizontal_Pod_Autoscaler:
  backend:
    min_replicas: 3
    max_replicas: 20
    target_cpu: 70%
    target_memory: 80%
  
  workers:
    min_replicas: 2
    max_replicas: 50
    target_cpu: 80%
    target_memory: 85%
    custom_metrics:
      - queue_length: 100

Vertical_Pod_Autoscaler:
  enabled: true
  update_mode: "Auto"
  resource_policy:
    container_policies:
      - container_name: backend
        max_allowed:
          cpu: 4
          memory: 8Gi
```

### **Caching Configuration**
```python
# caching/redis_config.py
import redis
from functools import wraps

redis_client = redis.Redis(
    host='redis',
    port=6379,
    db=0,
    decode_responses=True
)

def cache_result(expiration=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Compute and cache result
            result = func(*args, **kwargs)
            redis_client.setex(
                cache_key, 
                expiration, 
                json.dumps(result, default=str)
            )
            
            return result
        return wrapper
    return decorator
```

## 🔧 **Troubleshooting**

### **Common Issues**

#### **High Memory Usage**
```bash
# Check memory usage
kubectl top pods -n production

# Scale up if needed
kubectl scale deployment spacecomputer-backend --replicas=5

# Check for memory leaks
kubectl exec -it pod-name -- python -c "
import gc, psutil
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Objects: {len(gc.get_objects())}')
"
```

#### **Database Connection Issues**
```python
# health_check.py
import psycopg2
import redis

def check_database():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        return True
    except Exception as e:
        print(f"Database error: {e}")
        return False

def check_redis():
    try:
        r = redis.Redis.from_url(REDIS_URL)
        r.ping()
        return True
    except Exception as e:
        print(f"Redis error: {e}")
        return False
```

## 📚 **Next Steps**

1. **[API Reference](api-reference.md)** → Integrate with your applications
2. **[Contributing](contributing.md)** → Help improve the platform
3. **[Use Cases](use-cases.md)** → See real-world applications

**Need deployment help?** Contact our DevOps team at devops@space-computer.ai for enterprise deployment assistance. 