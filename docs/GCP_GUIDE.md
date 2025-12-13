# Complete Google Cloud Platform Guide

Comprehensive guide for setting up, deploying, and managing the Adaptive Self-Learning Agentic AI System on Google Cloud Platform.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installing gcloud CLI](#installing-gcloud-cli)
3. [Initial Setup](#initial-setup)
4. [GPU VM Setup](#gpu-vm-setup)
5. [Deployment Options](#deployment-options)
6. [Cost Optimization](#cost-optimization)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Troubleshooting](#troubleshooting)
9. [Security Best Practices](#security-best-practices)

---

## Quick Start

### 5-Minute Setup

```bash
# 1. Install gcloud CLI (if not installed)
brew install --cask google-cloud-sdk  # macOS
# Or: curl https://sdk.cloud.google.com | bash

# 2. Authenticate
gcloud auth login
gcloud config set project stt-agentic-ai-2025

# 3. Create GPU VM
bash scripts/gcp_scripts/setup_gcp_gpu.sh

# 4. Deploy API
python scripts/gcp_scripts/deploy_to_gcp.py
```

---

## Installing gcloud CLI

### For macOS (Recommended: Homebrew)

```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install gcloud CLI
brew install --cask google-cloud-sdk

# Initialize gcloud
gcloud init
```

### Alternative: Direct Download

```bash
# Download and install
curl https://sdk.cloud.google.com | bash

# Restart your shell or run:
exec -l $SHELL

# Initialize
gcloud init
```

### After Installation

1. **Authenticate**:
   ```bash
   gcloud auth login
   ```

2. **Set your project**:
   ```bash
   gcloud config set project stt-agentic-ai-2025
   ```
   (Or use your actual GCP project ID)

3. **Enable required APIs**:
   ```bash
   gcloud services enable compute.googleapis.com
   gcloud services enable storage.googleapis.com
   gcloud services enable run.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   ```

4. **Verify installation**:
   ```bash
   gcloud --version
   gcloud compute zones list
   ```

### Quick Test

```bash
# Check if gcloud works
gcloud --version

# List available zones
gcloud compute zones list | grep us-central
```

---

## Initial Setup

### Prerequisites

1. **GCP Account** with billing enabled
2. **gcloud CLI** installed (see above)
3. **Project ID**: `stt-agentic-ai-2025` (or your project ID)

### Configure GCP Project

```bash
# 1. Set project ID
export PROJECT_ID="stt-agentic-ai-2025"
gcloud config set project $PROJECT_ID

# 2. Set default region/zone
export REGION="us-central1"
export ZONE="us-central1-a"
gcloud config set compute/region $REGION
gcloud config set compute/zone $ZONE

# 3. Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com

# 4. Verify setup
gcloud config list
```

### Create GCS Buckets

```bash
# Set bucket names (must be globally unique)
export DATASETS_BUCKET="${PROJECT_ID}-stt-datasets"
export MODELS_BUCKET="${PROJECT_ID}-stt-models"
export LOGS_BUCKET="${PROJECT_ID}-stt-logs"

# Create buckets
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$DATASETS_BUCKET
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$MODELS_BUCKET
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$LOGS_BUCKET

# Verify buckets
gsutil ls -p $PROJECT_ID
```

### Setup Service Account

```bash
# 1. Create service account
gcloud iam service-accounts create stt-service-account \
    --display-name="STT System Service Account" \
    --description="Service account for STT system operations"

# 2. Grant necessary roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:stt-service-account@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:stt-service-account@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:stt-service-account@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/monitoring.metricWriter"

# 3. Download service account key
gcloud iam service-accounts keys create ~/stt-service-account-key.json \
    --iam-account=stt-service-account@${PROJECT_ID}.iam.gserviceaccount.com

# 4. Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS=~/stt-service-account-key.json
```

---

## GPU VM Setup

### Quick Setup (Automated)

```bash
# Make script executable
chmod +x scripts/gcp_scripts/setup_gcp_gpu.sh

# Run setup script
bash scripts/gcp_scripts/setup_gcp_gpu.sh
```

This will:
- Create a VM with NVIDIA T4 GPU
- Install PyTorch + CUDA
- Configure GPU drivers
- Set up project access

**Estimated Cost**: ~$0.54/hour (~$12.96/day if running 24/7)

### Manual Setup

```bash
# 1. Create GPU VM
gcloud compute instances create stt-gpu-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE \
  --scopes=https://www.googleapis.com/auth/cloud-platform

# 2. Wait for VM to be ready (2-3 minutes)
sleep 180

# 3. SSH into VM
gcloud compute ssh stt-gpu-vm --zone=us-central1-a

# 4. On VM: Clone and Setup
# (Inside VM)
cd ~
git clone <your-repo-url>
cd Adaptive-Self-Learning-Agentic-AI-System
pip install -r requirements.txt

# 5. Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 6. Exit VM
exit
```

### Deploy Code to GPU VM

```bash
# Deploy code and run evaluation
python scripts/gcp_scripts/deploy_to_gcp.py
```

This will:
- Upload your code to the VM
- Install dependencies
- Verify GPU access
- Run evaluation framework
- Download results

### GPU vs CPU Performance

Based on benchmarks:

| Metric | CPU | GPU (T4) | Speedup |
|--------|-----|----------|---------|
| Latency (per sample) | 0.72s | ~0.1-0.2s | **3-7x faster** |
| Throughput | 2.97 samples/s | ~10-20 samples/s | **3-7x faster** |
| 100 samples | 72s | 10-20s | **3-7x faster** |

### GPU Optimization

The code automatically optimizes for GPU when available:
- **TensorFloat-32 (TF32)**: Enabled for Ampere+ GPUs
- **Beam Search**: Better quality with GPU
- **KV Cache**: Faster generation on GPU
- **Half Precision**: Can be enabled for 2x speedup (with minor quality loss)

### Cost Management for GPU VMs

**VM Costs:**
- **T4 GPU + n1-standard-4**: ~$0.54/hour
- **Per day (24h)**: ~$12.96
- **Per month (730h)**: ~$394.20

**Cost-Saving Tips:**

1. **Stop VMs when not in use**:
   ```bash
   gcloud compute instances stop stt-gpu-vm --zone=us-central1-a
   gcloud compute instances start stt-gpu-vm --zone=us-central1-a
   ```

2. **Use Preemptible Instances** (60-80% cheaper):
   ```bash
   # Add --preemptible flag when creating VM
   gcloud compute instances create stt-gpu-vm-preemptible \
       --preemptible \
       --zone=us-central1-a \
       --machine-type=n1-standard-4 \
       --accelerator=type=nvidia-tesla-t4,count=1 \
       --image-family=pytorch-latest-gpu \
       --image-project=deeplearning-platform-release
   ```

3. **Use Smaller GPUs for Development**:
   - T4: Good for development (~$0.35/hour GPU)
   - V100: For training (~$2.50/hour GPU)
   - A100: For heavy training (~$3.00/hour GPU)

4. **Set Billing Alerts**:
   - Go to GCP Console → Billing → Budgets & Alerts
   - Set up alerts at 50%, 90%, 100% of budget

---

## Deployment Options

### Option A: Cloud Run (Recommended for Production)

**Best for**: Production environments, scalable applications, cost-effective for variable traffic

**Pros**:
- Auto-scaling (0 to N instances)
- Pay per request
- Managed infrastructure
- HTTPS out of the box
- Easy rollbacks

**Cons**:
- Cold start latency (~2-5s)
- No GPU support
- 300s timeout limit

**Setup**:

```bash
# 1. Build and deploy with Cloud Build
gcloud builds submit --tag gcr.io/$PROJECT_ID/stt-api

# 2. Deploy to Cloud Run
gcloud run deploy stt-api \
    --image gcr.io/$PROJECT_ID/stt-api \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --set-env-vars="GCS_DATASETS_BUCKET=$DATASETS_BUCKET,GCS_MODELS_BUCKET=$MODELS_BUCKET,GCS_LOGS_BUCKET=$LOGS_BUCKET,USE_GCS=true" \
    --service-account=stt-service-account@${PROJECT_ID}.iam.gserviceaccount.com

# 3. Get service URL
export SERVICE_URL=$(gcloud run services describe stt-api --region=$REGION --format='value(status.url)')
echo "Service URL: $SERVICE_URL"

# 4. Test deployment
curl $SERVICE_URL/api/health
```

**Estimated Cost**: $0 (free tier) to $50/month (moderate traffic)

### Option B: GPU VM for Training

**Best for**: Model training, fine-tuning, GPU-intensive workloads

**Setup**: Follow [GPU VM Setup](#gpu-vm-setup) section above

**Estimated Cost**: $0.54/hour (~$400/month if running 24/7)

**💡 Cost Tip**: Stop VM when not training!

```bash
# Start VM for training
gcloud compute instances start stt-training-vm --zone=$ZONE

# Train models
# ...

# Stop VM when done
gcloud compute instances stop stt-training-vm --zone=$ZONE
```

### Option C: App Engine

**Best for**: Simple deployment, no containerization needed

**Setup**:

```bash
# 1. Create app.yaml
cat > app.yaml << EOF
runtime: python39
instance_class: F4
automatic_scaling:
  min_instances: 0
  max_instances: 10
  target_cpu_utilization: 0.65

entrypoint: uvicorn src.control_panel_api:app --host 0.0.0.0 --port \$PORT

env_variables:
  USE_GCS: "true"
  GCS_DATASETS_BUCKET: "$DATASETS_BUCKET"
  GCS_MODELS_BUCKET: "$MODELS_BUCKET"
  GCS_LOGS_BUCKET: "$LOGS_BUCKET"
EOF

# 2. Deploy
gcloud app deploy

# 3. Open application
gcloud app browse
```

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Google Cloud Platform                 │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │  Cloud Run   │    │   Cloud      │                   │
│  │  (API)       │◄───┤   Storage    │                   │
│  │              │    │   (GCS)      │                   │
│  └──────────────┘    └──────────────┘                   │
│         │                    │                           │
│         │                    │                           │
│  ┌──────▼──────┐    ┌──────▼────────┐                  │
│  │  GPU VM     │    │  Artifact      │                  │
│  │  (Training) │    │  Registry      │                  │
│  └─────────────┘    └────────────────┘                  │
│                                                           │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │  Cloud       │    │  Cloud        │                  │
│  │  Monitoring  │    │  Logging      │                  │
│  └──────────────┘    └──────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

### Storage Structure

```
GCS Buckets:
├── stt-project-datasets/
│   ├── raw/              # Raw audio files
│   ├── processed/        # Processed audio
│   └── finetuning/       # Training datasets
├── stt-project-models/
│   ├── baseline/         # Base models
│   ├── finetuned/        # Fine-tuned models
│   └── deployed/         # Active models
└── stt-project-logs/
    ├── training/         # Training logs
    └── inference/        # Inference logs
```

---

## Cost Optimization

### Estimated Monthly Costs

| Component | Configuration | Monthly Cost |
|-----------|--------------|--------------|
| Cloud Run API | Light traffic (<100k requests) | $0-10 |
| Cloud Run API | Medium traffic (1M requests) | $50-100 |
| Cloud Storage | 100GB storage | $2-3 |
| GPU VM (T4) | Running 24/7 | ~$400 |
| GPU VM (T4) | 8 hours/day | ~$130 |
| Monitoring | Standard | Free-$5 |
| **Total (Development)** | Cloud Run + Storage | **$5-15/month** |
| **Total (Production)** | All services | **$150-500/month** |

### Cost-Saving Strategies

#### 1. Use Preemptible VMs for Training

```bash
# Create preemptible VM (60-80% cheaper)
gcloud compute instances create stt-training-vm-preemptible \
    --zone=$ZONE \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --preemptible \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB

# Note: Preemptible VMs can be terminated anytime
# Use checkpointing in your training code!
```

#### 2. Set Budget Alerts

```bash
# Create budget
gcloud billing budgets create \
    --billing-account=YOUR-BILLING-ACCOUNT-ID \
    --display-name="STT Project Budget" \
    --budget-amount=100USD \
    --threshold-rule=percent=50 \
    --threshold-rule=percent=90 \
    --threshold-rule=percent=100
```

#### 3. Lifecycle Policies for Storage

```bash
# Delete old training logs after 30 days
cat > lifecycle-logs.json << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 30}
      }
    ]
  }
}
EOF

gsutil lifecycle set lifecycle-logs.json gs://$LOGS_BUCKET
```

#### 4. Cloud Run Optimization

```bash
# Deploy with minimum instances = 0 (cold starts but cheaper)
gcloud run services update stt-api \
    --region=$REGION \
    --min-instances=0 \
    --max-instances=5

# For production with consistent traffic, use min-instances=1
gcloud run services update stt-api \
    --region=$REGION \
    --min-instances=1 \
    --max-instances=10
```

#### 5. Monitor Costs Continuously

```bash
# Use provided monitoring script
python scripts/gcp_scripts/monitor_gcp_costs.py

# Or check in console
open https://console.cloud.google.com/billing
```

---

## Monitoring & Maintenance

### Daily Monitoring

```bash
# 1. Check service health
curl $SERVICE_URL/api/health

# 2. View recent logs
gcloud logging read "resource.type=cloud_run_revision" \
    --limit 20 \
    --format="table(timestamp,jsonPayload.message)"

# 3. Check costs
python scripts/gcp_scripts/monitor_gcp_costs.py

# 4. View system stats
curl $SERVICE_URL/api/system/stats | jq '.data_management'
```

### Weekly Tasks

```bash
# 1. Review failed cases
curl $SERVICE_URL/api/data/failed-cases?limit=100 | jq

# 2. Check if ready for fine-tuning
curl $SERVICE_URL/api/data/statistics | jq

# 3. Generate performance report
curl $SERVICE_URL/api/data/report -o weekly-report.json

# 4. Review storage usage
gsutil du -sh gs://$DATASETS_BUCKET
gsutil du -sh gs://$MODELS_BUCKET
gsutil du -sh gs://$LOGS_BUCKET
```

### Monthly Maintenance

```bash
# 1. Update dependencies
pip install --upgrade -r requirements.txt

# 2. Rebuild and redeploy
gcloud builds submit --tag gcr.io/$PROJECT_ID/stt-api
gcloud run deploy stt-api --image gcr.io/$PROJECT_ID/stt-api --region=$REGION

# 3. Backup critical data
gsutil -m cp -r gs://$MODELS_BUCKET gs://${PROJECT_ID}-backup/models-$(date +%Y%m%d)

# 4. Clean up old artifacts
gcloud container images list-tags gcr.io/$PROJECT_ID/stt-api \
    --filter='-tags:*' --format='get(digest)' | \
    xargs -I {} gcloud container images delete gcr.io/$PROJECT_ID/stt-api@{} --quiet

# 5. Review and optimize costs
python scripts/gcp_scripts/monitor_gcp_costs.py
```

### Setup Monitoring & Alerts

```bash
# 1. Create notification channel (email)
gcloud alpha monitoring channels create \
    --display-name="STT Alerts" \
    --type=email \
    --channel-labels=email_address=your-email@example.com

# Get channel ID
export CHANNEL_ID=$(gcloud alpha monitoring channels list --format="value(name)")

# 2. Create uptime check
gcloud monitoring uptime create http stt-api-uptime \
    --display-name="STT API Uptime Check" \
    --resource-type=uptime-url \
    --host=$SERVICE_URL \
    --path=/api/health \
    --check-interval=5m

# 3. View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=stt-api" \
    --limit 50 \
    --format json
```

---

## Troubleshooting

### Issue 1: gcloud Not Found

**Symptoms**: `command not found: gcloud`

**Solutions**:
```bash
# Add to PATH (macOS)
echo 'export PATH="$HOME/google-cloud-sdk/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Or reinstall
brew install --cask google-cloud-sdk
```

### Issue 2: GPU Not Detected

**Symptoms**: `torch.cuda.is_available()` returns `False`

**Solutions**:
```bash
# SSH into VM
gcloud compute ssh stt-gpu-vm --zone=$ZONE

# Check NVIDIA drivers
nvidia-smi

# If not found, install drivers
sudo /opt/deeplearning/install-driver.sh

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Reboot if needed
sudo reboot
```

### Issue 3: Cloud Run Deployment Fails

**Symptoms**: Build fails or service doesn't deploy

**Solutions**:
```bash
# Check build logs
gcloud builds list --limit=5
gcloud builds log BUILD_ID

# Verify Docker image builds locally
docker build -t test-stt-api .
docker run -p 8000:8000 test-stt-api

# Check Cloud Run logs
gcloud run services describe stt-api --region=$REGION
gcloud logging read "resource.type=cloud_run_revision" --limit=50
```

### Issue 4: Permission Denied on GCS

**Symptoms**: `403 Forbidden` when accessing buckets

**Solutions**:
```bash
# Check service account permissions
gcloud projects get-iam-policy $PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:serviceAccount:stt-service-account@${PROJECT_ID}.iam.gserviceaccount.com"

# Add missing permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:stt-service-account@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# Verify bucket IAM
gsutil iam get gs://$DATASETS_BUCKET
```

### Issue 5: High Costs

**Symptoms**: Unexpected high billing

**Solutions**:
```bash
# Check running instances
gcloud compute instances list
gcloud run services list

# Stop unused VMs
gcloud compute instances stop INSTANCE_NAME --zone=$ZONE

# Check storage usage
gsutil du -sh gs://$DATASETS_BUCKET

# Delete old data
gsutil -m rm -r gs://$LOGS_BUCKET/old-logs/*

# Review detailed costs
open https://console.cloud.google.com/billing
```

### Issue 6: API Timeout Errors

**Symptoms**: 504 Gateway Timeout on long transcriptions

**Solutions**:
```bash
# Increase Cloud Run timeout (max 3600s for 2nd gen)
gcloud run services update stt-api \
    --region=$REGION \
    --timeout=900 \
    --execution-environment=gen2

# For very long files, use GPU VM instead
```

### Issue 7: Out of Memory Errors

**Symptoms**: Container crashes with OOM error

**Solutions**:
```bash
# Increase Cloud Run memory
gcloud run services update stt-api \
    --region=$REGION \
    --memory=8Gi \
    --cpu=4

# Check memory usage in logs
gcloud logging read "resource.type=cloud_run_revision AND jsonPayload.message:memory" \
    --limit=50
```

### Issue 8: VM Won't Start

**Solutions**:
```bash
# Check GPU quota
gcloud compute project-info describe

# Request quota increase if needed
# Go to: https://console.cloud.google.com/iam-admin/quotas

# Try different zone
gcloud compute instances create stt-gpu-vm \
    --zone=us-central1-b \
    # ... other flags
```

---

## Security Best Practices

### 1. Service Account Security

```bash
# Use separate service accounts for different components
gcloud iam service-accounts create stt-api-sa --display-name="API Service Account"
gcloud iam service-accounts create stt-training-sa --display-name="Training Service Account"

# Grant minimal required permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:stt-api-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"  # Read-only for API

# Rotate keys regularly
gcloud iam service-accounts keys list \
    --iam-account=stt-service-account@${PROJECT_ID}.iam.gserviceaccount.com
```

### 2. API Authentication

```bash
# For production, require authentication
gcloud run services update stt-api \
    --region=$REGION \
    --no-allow-unauthenticated

# Access with token
gcloud auth print-identity-token > token.txt
curl -H "Authorization: Bearer $(cat token.txt)" $SERVICE_URL/api/health
```

### 3. Network Security

```bash
# Use VPC for VM isolation
gcloud compute networks create stt-network --subnet-mode=custom

gcloud compute networks subnets create stt-subnet \
    --network=stt-network \
    --region=$REGION \
    --range=10.0.0.0/24

# Create firewall rules
gcloud compute firewall-rules create allow-ssh \
    --network=stt-network \
    --allow=tcp:22 \
    --source-ranges=YOUR_IP/32
```

### 4. Data Encryption

```bash
# Use customer-managed encryption keys (CMEK)
gcloud kms keyrings create stt-keyring --location=$REGION

gcloud kms keys create stt-key \
    --location=$REGION \
    --keyring=stt-keyring \
    --purpose=encryption

# Apply to bucket
gsutil kms encryption gs://$DATASETS_BUCKET \
    projects/$PROJECT_ID/locations/$REGION/keyRings/stt-keyring/cryptoKeys/stt-key
```

### 5. Audit Logging

```bash
# Enable Cloud Audit Logs
gcloud logging sinks create stt-audit-sink \
    gs://${PROJECT_ID}-audit-logs \
    --log-filter='protoPayload.serviceName="storage.googleapis.com"'
```

---

## Post-Deployment Checklist

- [ ] All APIs enabled
- [ ] Service account created with proper permissions
- [ ] GCS buckets created and configured
- [ ] Cloud Run service deployed and healthy
- [ ] GPU VM created (if needed) and tested
- [ ] Frontend accessible
- [ ] Environment variables configured
- [ ] Monitoring and alerts setup
- [ ] Budget alerts configured
- [ ] Cost monitoring script tested
- [ ] API authentication configured (if production)
- [ ] Backup strategy implemented
- [ ] Documentation updated with actual URLs and IDs
- [ ] Team trained on deployment process

---

## Next Steps

1. **Fine-tune your first model**: Follow `docs/FINETUNING_GUIDE.md`
2. **Integrate W&B**: Follow `docs/WANDB_GUIDE.md`
3. **Setup CI/CD**: Automate deployments with Cloud Build triggers
4. **Scale up**: Configure auto-scaling based on traffic
5. **Optimize costs**: Review and implement cost-saving strategies

---

## Support & Resources

- **GCP Documentation**: https://cloud.google.com/docs
- **Cloud Run Docs**: https://cloud.google.com/run/docs
- **Cost Calculator**: https://cloud.google.com/products/calculator
- **Project Documentation**: See `docs/` directory
- **API Reference**: `http://YOUR-SERVICE-URL/docs`
- **Official gcloud Install Guide**: https://cloud.google.com/sdk/docs/install

---

**Last Updated**: December 2024  
**Version**: 1.0  
**Tested on**: Google Cloud Platform

