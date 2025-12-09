# Comprehensive Google Cloud Platform Deployment Guide

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture Overview](#architecture-overview)
4. [Step-by-Step Deployment](#step-by-step-deployment)
5. [Deployment Options](#deployment-options)
6. [Cost Optimization](#cost-optimization)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Troubleshooting](#troubleshooting)
9. [Security Best Practices](#security-best-practices)

---

## 🎯 Overview

This guide provides comprehensive instructions for deploying the Adaptive Self-Learning Agentic AI System on Google Cloud Platform. The system can be deployed in multiple configurations depending on your needs:

- **Option A**: Complete production deployment with Cloud Run + Cloud Storage
- **Option B**: GPU VM for model training and fine-tuning
- **Option C**: App Engine deployment for simple hosting
- **Option D**: Development/testing deployment

---

## 📋 Prerequisites

### Required Accounts & Tools

1. **Google Cloud Platform Account**
   - Active GCP account with billing enabled
   - At least $50 in credits (recommended for testing)
   - Project created (or use default)

2. **Local Development Tools**
   ```bash
   # Install gcloud CLI (macOS)
   brew install --cask google-cloud-sdk
   
   # Or download from:
   # https://cloud.google.com/sdk/docs/install
   
   # Verify installation
   gcloud --version
   ```

3. **Required Software**
   - Python 3.8+ (`python --version`)
   - Docker Desktop (for containerized deployment)
   - Git
   - curl

4. **Optional Tools**
   - kubectl (for Kubernetes deployments)
   - Terraform (for infrastructure as code)

### Initial GCP Setup

```bash
# 1. Authenticate with GCP
gcloud auth login

# 2. Set your project ID (create one if needed)
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# 3. Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable appengine.googleapis.com

# 4. Set default region/zone
export REGION="us-central1"
export ZONE="us-central1-a"
gcloud config set compute/region $REGION
gcloud config set compute/zone $ZONE

# 5. Verify setup
gcloud config list
```

---

## 🏗️ Architecture Overview

### Component Architecture

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

## 🚀 Step-by-Step Deployment

### STEP 1: Clone and Setup Repository

```bash
# 1. Clone repository
git clone <your-repo-url>
cd Adaptive-Self-Learning-Agentic-AI-System

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify local setup
python scripts/verify_setup.py
```

### STEP 2: Configure GCP Storage

```bash
# 1. Set bucket names (must be globally unique)
export PROJECT_ID="your-project-id"
export DATASETS_BUCKET="${PROJECT_ID}-stt-datasets"
export MODELS_BUCKET="${PROJECT_ID}-stt-models"
export LOGS_BUCKET="${PROJECT_ID}-stt-logs"

# 2. Create GCS buckets
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$DATASETS_BUCKET
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$MODELS_BUCKET
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$LOGS_BUCKET

# 3. Set lifecycle policies (optional - save costs)
cat > lifecycle.json << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 90,
          "matchesPrefix": ["logs/"]
        }
      }
    ]
  }
}
EOF

gsutil lifecycle set lifecycle.json gs://$LOGS_BUCKET

# 4. Verify buckets
gsutil ls -p $PROJECT_ID

# 5. Set bucket permissions (if needed for specific service accounts)
gsutil iam ch allUsers:objectViewer gs://$MODELS_BUCKET  # Only if models should be public
```

### STEP 3: Setup Service Account

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

# 5. Verify service account
gcloud iam service-accounts list
```

### STEP 4: Upload Initial Data

```bash
# 1. Upload test audio files
gsutil -m cp -r data/test_audio gs://$DATASETS_BUCKET/test_audio/

# 2. Upload any existing processed data
gsutil -m cp -r data/processed gs://$DATASETS_BUCKET/processed/

# 3. Verify upload
gsutil ls -r gs://$DATASETS_BUCKET
```

### STEP 5: Deploy Backend API (Cloud Run - Recommended)

#### Option A: Deploy with Cloud Build

```bash
# 1. Create Dockerfile (already provided below)
# See the Dockerfile in the next section

# 2. Build and deploy with Cloud Build
gcloud builds submit --tag gcr.io/$PROJECT_ID/stt-api

# 3. Deploy to Cloud Run
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

# 4. Get service URL
export SERVICE_URL=$(gcloud run services describe stt-api --region=$REGION --format='value(status.url)')
echo "Service URL: $SERVICE_URL"

# 5. Test deployment
curl $SERVICE_URL/api/health
```

#### Option B: Deploy from Container Registry

```bash
# 1. Build Docker image locally
docker build -t gcr.io/$PROJECT_ID/stt-api:latest .

# 2. Configure Docker to use gcloud
gcloud auth configure-docker

# 3. Push to Container Registry
docker push gcr.io/$PROJECT_ID/stt-api:latest

# 4. Deploy to Cloud Run (same as above)
gcloud run deploy stt-api \
    --image gcr.io/$PROJECT_ID/stt-api:latest \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --set-env-vars="GCS_DATASETS_BUCKET=$DATASETS_BUCKET,GCS_MODELS_BUCKET=$MODELS_BUCKET,GCS_LOGS_BUCKET=$LOGS_BUCKET,USE_GCS=true"
```

### STEP 6: Setup GPU VM for Training

```bash
# 1. Use the optimized script
bash scripts/setup_gcp_gpu.sh

# Or create manually:
gcloud compute instances create stt-training-vm \
    --zone=$ZONE \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True" \
    --scopes=cloud-platform \
    --service-account=stt-service-account@${PROJECT_ID}.iam.gserviceaccount.com

# 2. Wait for VM to be ready (2-3 minutes)
sleep 180

# 3. SSH into VM
gcloud compute ssh stt-training-vm --zone=$ZONE

# 4. On VM: Setup environment
# (Inside VM)
cd ~
git clone <your-repo-url>
cd Adaptive-Self-Learning-Agentic-AI-System
pip install -r requirements.txt

# 5. Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 6. Exit VM
exit

# 7. Stop VM to save costs (start when needed)
gcloud compute instances stop stt-training-vm --zone=$ZONE
```

### STEP 7: Deploy Frontend

The frontend is automatically served by the Cloud Run API at `/app`.

```bash
# Test frontend access
curl $SERVICE_URL/app

# Or open in browser
open $SERVICE_URL/app
```

For separate frontend hosting (optional):

```bash
# Deploy to Firebase Hosting (alternative)
# 1. Install Firebase CLI
npm install -g firebase-tools

# 2. Initialize Firebase
firebase init hosting

# 3. Deploy frontend
firebase deploy --only hosting

# Or deploy to Cloud Storage as static site
gsutil cp -r frontend/* gs://${PROJECT_ID}-frontend/
gsutil web set -m index.html gs://${PROJECT_ID}-frontend
```

### STEP 8: Configure Environment Variables

```bash
# Update your local .env file
cat > .env << EOF
# GCP Configuration
PROJECT_ID=$PROJECT_ID
REGION=$REGION
ZONE=$ZONE

# Storage Buckets
GCS_DATASETS_BUCKET=$DATASETS_BUCKET
GCS_MODELS_BUCKET=$MODELS_BUCKET
GCS_LOGS_BUCKET=$LOGS_BUCKET

# API Configuration
API_URL=$SERVICE_URL
USE_GCS=true

# Service Account
GOOGLE_APPLICATION_CREDENTIALS=~/stt-service-account-key.json

# Model Configuration
MODEL_NAME=whisper
DEVICE=cpu  # Use 'cuda' on GPU VMs
EOF

# Load environment variables
source .env
```

### STEP 9: Initial System Verification

```bash
# 1. Check API health
curl $SERVICE_URL/api/health | jq

# 2. Test baseline transcription
curl -X POST $SERVICE_URL/api/transcribe/baseline \
    -F "file=@data/test_audio/test_1.wav" | jq

# 3. Test agent transcription
curl -X POST "$SERVICE_URL/api/transcribe/agent?auto_correction=true" \
    -F "file=@data/test_audio/test_1.wav" | jq

# 4. Check system stats
curl $SERVICE_URL/api/system/stats | jq

# 5. View API documentation
open $SERVICE_URL/docs
```

### STEP 10: Setup Monitoring & Alerts

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

# 3. Create alert policy for errors
cat > alert-policy.json << EOF
{
  "displayName": "STT API High Error Rate",
  "conditions": [
    {
      "displayName": "Error rate > 5%",
      "conditionThreshold": {
        "filter": "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"stt-api\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.labels.response_code_class=\"5xx\"",
        "comparison": "COMPARISON_GT",
        "thresholdValue": 0.05,
        "duration": "300s"
      }
    }
  ],
  "notificationChannels": ["$CHANNEL_ID"],
  "enabled": true
}
EOF

gcloud alpha monitoring policies create --policy-from-file=alert-policy.json

# 4. View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=stt-api" \
    --limit 50 \
    --format json
```

---

## 🎛️ Deployment Options

### Option A: Production Deployment (Cloud Run + GCS)

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

**Setup**: Follow STEP 5 Option A above

**Estimated Cost**: $0 (free tier) to $50/month (moderate traffic)

### Option B: GPU VM for Training

**Best for**: Model training, fine-tuning, GPU-intensive workloads

**Pros**:
- GPU acceleration (3-7x faster)
- Full control over environment
- No timeout limits
- Persistent storage

**Cons**:
- Higher cost when running
- Manual scaling
- Requires management

**Setup**: Follow STEP 6 above

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

### Option C: App Engine Deployment

**Best for**: Simple deployment, no containerization needed

**Pros**:
- Easy deployment (`gcloud app deploy`)
- Auto-scaling
- Integrated with GCP services

**Cons**:
- Less flexible than Cloud Run
- Higher minimum cost
- Longer deployment times

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

### Option D: Kubernetes Deployment (Advanced)

**Best for**: Complex deployments, microservices, advanced orchestration

See `docs/KUBERNETES_DEPLOYMENT.md` for detailed instructions.

---

## 💰 Cost Optimization

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
python scripts/monitor_gcp_costs.py

# Or check in console
open https://console.cloud.google.com/billing
```

---

## 📊 Monitoring & Maintenance

### Daily Monitoring

```bash
# 1. Check service health
curl $SERVICE_URL/api/health

# 2. View recent logs
gcloud logging read "resource.type=cloud_run_revision" \
    --limit 20 \
    --format="table(timestamp,jsonPayload.message)"

# 3. Check costs
python scripts/monitor_gcp_costs.py

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
python scripts/monitor_gcp_costs.py
```

### Performance Monitoring

```bash
# 1. View Cloud Run metrics
gcloud monitoring dashboards list

# 2. Check request latency
gcloud logging read "resource.type=cloud_run_revision AND jsonPayload.latency>1000" \
    --limit 50

# 3. Monitor error rates
gcloud logging read "resource.type=cloud_run_revision AND severity=ERROR" \
    --limit 50
```

---

## 🔧 Troubleshooting

### Issue 1: Cloud Run Deployment Fails

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

### Issue 2: GPU VM Not Detecting GPU

**Symptoms**: `torch.cuda.is_available()` returns `False`

**Solutions**:

```bash
# SSH into VM
gcloud compute ssh stt-training-vm --zone=$ZONE

# Check NVIDIA drivers
nvidia-smi

# If not found, install drivers
sudo /opt/deeplearning/install-driver.sh

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Reboot if needed
sudo reboot
```

### Issue 3: Permission Denied on GCS

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

### Issue 4: High Costs

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

### Issue 5: API Timeout Errors

**Symptoms**: 504 Gateway Timeout on long transcriptions

**Solutions**:

```bash
# Increase Cloud Run timeout (max 3600s for 2nd gen)
gcloud run services update stt-api \
    --region=$REGION \
    --timeout=900 \
    --execution-environment=gen2

# Add request-timeout header
curl -X POST $SERVICE_URL/api/transcribe/agent \
    -H "X-Cloud-Trace-Context: TRACE_ID" \
    -F "file=@audio.wav" \
    --max-time 300

# For very long files, use GPU VM instead
```

### Issue 6: Out of Memory Errors

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

---

## 🔒 Security Best Practices

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

## 📝 Post-Deployment Checklist

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

## 🎓 Next Steps

1. **Fine-tune your first model**: Follow `docs/FINETUNING_QUICK_START.md`
2. **Integrate W&B**: Follow `docs/WANDB_SWEEPS_GUIDE.md`
3. **Setup CI/CD**: Automate deployments with Cloud Build triggers
4. **Scale up**: Configure auto-scaling based on traffic
5. **Optimize costs**: Review and implement cost-saving strategies

---

## 📞 Support & Resources

- **GCP Documentation**: https://cloud.google.com/docs
- **Cloud Run Docs**: https://cloud.google.com/run/docs
- **Cost Calculator**: https://cloud.google.com/products/calculator
- **Project Documentation**: See `docs/` directory
- **API Reference**: `http://YOUR-SERVICE-URL/docs`

---

**Last Updated**: December 2024  
**Version**: 1.0  
**Tested on**: Google Cloud Platform

