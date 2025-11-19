#!/bin/bash
# Quick setup script - Checks prerequisites and guides through GCP setup

set -e

echo "=========================================="
echo "GCP Setup Prerequisites Check"
echo "=========================================="
echo ""

# Check if gcloud is installed
if command -v gcloud &> /dev/null; then
    echo "✅ gcloud CLI is installed"
    gcloud --version | head -1
else
    echo "❌ gcloud CLI not found"
    echo ""
    echo "Please install gcloud CLI first:"
    echo ""
    echo "Option 1 (macOS - Recommended):"
    echo "  brew install --cask google-cloud-sdk"
    echo ""
    echo "Option 2 (Direct):"
    echo "  curl https://sdk.cloud.google.com | bash"
    echo ""
    echo "See scripts/INSTALL_GCLOUD.md for detailed instructions"
    echo ""
    exit 1
fi

echo ""
echo "Checking authentication..."
if gcloud auth list --filter=status:ACTIVE --format="value(account)" &>/dev/null; then
    ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1)
    echo "✅ Authenticated as: $ACCOUNT"
else
    echo "⚠️  Not authenticated"
    echo "   Run: gcloud auth login"
    exit 1
fi

echo ""
echo "Checking project..."
PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ -n "$PROJECT" ]; then
    echo "✅ Project set to: $PROJECT"
else
    echo "⚠️  No project set"
    echo "   Run: gcloud config set project YOUR_PROJECT_ID"
    echo "   Or update PROJECT_ID in scripts/setup_gcp_gpu.sh"
    exit 1
fi

echo ""
echo "Checking required APIs..."
APIS_ENABLED=$(gcloud services list --enabled --filter="name:compute.googleapis.com OR name:storage-api.googleapis.com" --format="value(name)" 2>/dev/null | wc -l)
if [ "$APIS_ENABLED" -ge 1 ]; then
    echo "✅ Required APIs appear to be enabled"
else
    echo "⚠️  Enabling required APIs..."
    gcloud services enable compute.googleapis.com
    gcloud services enable storage-api.googleapis.com
    echo "✅ APIs enabled"
fi

echo ""
echo "=========================================="
echo "✅ All prerequisites met!"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  bash scripts/setup_gcp_gpu.sh"
echo ""
echo "Or check GPU quota first:"
echo "  gcloud compute project-info describe --project=$PROJECT | grep -i quota"
echo ""

