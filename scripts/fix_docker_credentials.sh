#!/bin/bash
# Script to fix Docker credential issues
# Run this if you encounter "error getting credentials" when building Docker images

echo "=========================================="
echo "Docker Credential Fix Script"
echo "=========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Check Docker credential helper
echo "Checking Docker credential configuration..."
if [ -f ~/.docker/config.json ]; then
    echo "Docker config found at ~/.docker/config.json"
    cat ~/.docker/config.json | grep -A 5 "credsStore" || echo "No credsStore configured"
else
    echo "No Docker config file found"
fi

echo ""
echo "=========================================="
echo "Solutions to try:"
echo "=========================================="
echo ""
echo "Option 1: Remove credential helper (if causing issues)"
echo "  Edit ~/.docker/config.json and remove 'credsStore' line"
echo ""
echo "Option 2: Use CPU-only Dockerfile for local testing"
echo "  docker build -f Dockerfile.training.cpu -t adaptive-stt-training:cpu ."
echo ""
echo "Option 3: Login to Docker Hub"
echo "  docker login"
echo ""
echo "Option 4: Use public registry without authentication"
echo "  The NVIDIA CUDA images are public, so credential issues shouldn't occur."
echo "  Try: docker pull nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04"
echo ""
echo "Option 5: Reset Docker credentials"
echo "  rm ~/.docker/config.json"
echo "  docker login (if needed)"
echo ""

# Try to pull the base image directly
echo "Testing direct image pull..."
if docker pull nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 2>&1 | head -5; then
    echo "✓ Successfully pulled base image"
else
    echo "⚠ Could not pull base image. Try the CPU-only version instead:"
    echo "  docker build -f Dockerfile.training.cpu -t adaptive-stt-training:cpu ."
fi

echo ""
echo "For local testing without GPU, use:"
echo "  docker build -f Dockerfile.training.cpu -t adaptive-stt-training:cpu ."
