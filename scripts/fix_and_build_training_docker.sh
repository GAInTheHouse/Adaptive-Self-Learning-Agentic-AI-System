#!/bin/bash
# Fix Docker credentials and build training containers
# Week 1: Build both CPU and GPU versions

set -e

echo "=========================================="
echo "Docker Training Container Builder"
echo "=========================================="
echo ""

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Fix credential helper issue
echo "Fixing Docker credential configuration..."
DOCKER_CONFIG="$HOME/.docker/config.json"

if [ -f "$DOCKER_CONFIG" ]; then
    # Backup original config
    cp "$DOCKER_CONFIG" "$DOCKER_CONFIG.backup"
    echo "✓ Backed up Docker config to $DOCKER_CONFIG.backup"
    
    # Remove credsStore temporarily (public images don't need it)
    # Use jq if available, otherwise use sed
    if command -v jq &> /dev/null; then
        jq 'del(.credsStore)' "$DOCKER_CONFIG" > "$DOCKER_CONFIG.tmp" && mv "$DOCKER_CONFIG.tmp" "$DOCKER_CONFIG"
        echo "✓ Removed credsStore using jq"
    else
        # Fallback: use sed to remove credsStore line
        sed -i.bak '/"credsStore"/d' "$DOCKER_CONFIG" 2>/dev/null || \
        sed -i '' '/"credsStore"/d' "$DOCKER_CONFIG" 2>/dev/null || \
        echo "⚠ Could not remove credsStore automatically. Please edit $DOCKER_CONFIG manually"
    fi
else
    echo "⚠ No Docker config found, creating one..."
    mkdir -p "$HOME/.docker"
    echo '{"auths": {}}' > "$DOCKER_CONFIG"
fi

echo ""

# Test pulling a public image
echo "Testing Docker image pull..."
if docker pull python:3.10-slim > /dev/null 2>&1; then
    echo "✓ Successfully pulled python:3.10-slim"
else
    echo "⚠ Warning: Could not pull python:3.10-slim"
    echo "  This may indicate network or Docker issues"
fi

echo ""

# Build CPU version (for local testing)
echo "=========================================="
echo "Building CPU-only training container..."
echo "=========================================="
if docker build -f Dockerfile.training.cpu -t adaptive-stt-training:cpu . 2>&1 | tee /tmp/docker_build_cpu.log; then
    echo ""
    echo "✅ CPU container built successfully!"
    echo ""
    
    # Verify CPU container
    echo "Verifying CPU container dependencies..."
    docker run --rm adaptive-stt-training:cpu \
        python3 -c "from peft import LoraConfig; print('✓ LoRA: OK')" && \
    docker run --rm adaptive-stt-training:cpu \
        python3 -c "from transformers import Wav2Vec2ForCTC; print('✓ Wav2Vec2: OK')" && \
    docker run --rm adaptive-stt-training:cpu \
        python3 -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
    
    echo ""
    echo "✅ CPU container verification complete!"
else
    echo ""
    echo "❌ CPU container build failed. Check /tmp/docker_build_cpu.log for details."
    BUILD_CPU_FAILED=true
fi

echo ""

# Try GPU version (may fail if NVIDIA image not accessible)
echo "=========================================="
echo "Attempting GPU training container build..."
echo "=========================================="
echo "Note: This requires NVIDIA CUDA base image. Will skip if not accessible."

# Test if we can pull NVIDIA image
if docker pull nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 > /dev/null 2>&1; then
    echo "✓ NVIDIA CUDA image accessible"
    echo ""
    
    if docker build -f Dockerfile.training -t adaptive-stt-training:latest . 2>&1 | tee /tmp/docker_build_gpu.log; then
        echo ""
        echo "✅ GPU container built successfully!"
        echo ""
        
        # Verify GPU container (if GPU available)
        if command -v nvidia-smi &> /dev/null; then
            echo "Verifying GPU container with GPU..."
            docker run --rm --gpus all adaptive-stt-training:latest \
                python3 -c "import torch; print(f'✓ CUDA Available: {torch.cuda.is_available()}')" || true
        else
            echo "⚠ No GPU detected locally. GPU container built but not tested."
        fi
    else
        echo ""
        echo "⚠ GPU container build failed. Check /tmp/docker_build_gpu.log"
        echo "  CPU version is sufficient for Week 1 verification."
    fi
else
    echo "⚠ NVIDIA CUDA image not accessible (credential or network issue)"
    echo "  Skipping GPU container build. CPU version is sufficient for Week 1."
    echo "  GPU version will be built on GCP in Week 2."
fi

echo ""
echo "=========================================="
echo "Build Summary"
echo "=========================================="

if [ -z "$BUILD_CPU_FAILED" ]; then
    echo "✅ CPU container: adaptive-stt-training:cpu"
    echo "   Usage: docker run --rm adaptive-stt-training:cpu bash scripts/verify_training_docker.sh"
fi

if docker images | grep -q "adaptive-stt-training.*latest"; then
    echo "✅ GPU container: adaptive-stt-training:latest"
    echo "   Usage: docker run --rm --gpus all adaptive-stt-training:latest bash scripts/verify_training_docker.sh"
fi

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo "1. CPU version is ready for Week 1 verification"
echo "2. GPU version will be built on GCP in Week 2"
echo "3. To restore Docker config: cp $DOCKER_CONFIG.backup $DOCKER_CONFIG"
echo ""
