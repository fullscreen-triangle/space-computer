# Multi-stage Docker build for Space Computer
# Stage 1: Frontend Build
FROM node:18-alpine AS frontend-builder

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY tsconfig.json ./
COPY tailwind.config.js ./
COPY postcss.config.js ./

# Install frontend dependencies
RUN npm ci --only=production

# Copy frontend source
COPY src/ ./src/
COPY public/ ./public/

# Build frontend
RUN npm run build

# Stage 2: Python Backend Build
FROM python:3.11-slim AS backend-builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    g++ \
    git \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install FFmpeg for video processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python source code
COPY src/ ./src/
COPY pyproject.toml .
COPY README.md .

# Install the package
RUN pip install -e .

# Stage 3: Production Runtime
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r spacecomputer && useradd -r -g spacecomputer spacecomputer

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=backend-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend-builder /usr/local/bin /usr/local/bin

# Copy built frontend from frontend builder
COPY --from=frontend-builder /app/out ./static/

# Copy Python application
COPY --from=backend-builder /app/src ./src/
COPY --from=backend-builder /app/pyproject.toml .

# Create necessary directories
RUN mkdir -p /app/uploads /app/models /app/logs /app/temp && \
    chown -R spacecomputer:spacecomputer /app

# Switch to non-root user
USER spacecomputer

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "space_computer.main:app", "--host", "0.0.0.0", "--port", "8000"] 