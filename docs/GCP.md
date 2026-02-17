# Google Cloud Platform Guide

Setup and deployment for GCP resources.

## Install gcloud CLI

### macOS (Homebrew)

```bash
brew install --cask google-cloud-sdk
gcloud init
```

### Direct Download

```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

### After Installation

```bash
gcloud auth login
gcloud config set project your-project-id
gcloud services enable compute.googleapis.com storage-api.googleapis.com
```

---

## GPU VM for Training/Evaluation

### Quick Start

```bash
chmod +x scripts/setup_gcp_gpu.sh
bash scripts/setup_gcp_gpu.sh

python scripts/deploy_to_gcp.py
python scripts/monitor_gcp_costs.py
```

### Manual VM Creation

```bash
gcloud compute instances create stt-gpu-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB
```

**Cost**: ~$0.54/hour (T4 GPU). Stop when not in use: `gcloud compute instances stop stt-gpu-vm --zone=us-central1-a`

---

## Deployment Options

- **Cloud Run**: Production API hosting
- **GPU VM**: Training and fine-tuning (see above)
- **App Engine**: Simple hosting
- **Cloud Storage**: Datasets and models

### Create Storage Buckets

```bash
gsutil mb gs://your-project-datasets
gsutil mb gs://your-project-models
```

### Fine-Tuning on GCP

```bash
python scripts/deploy_finetuning_to_gcp.py \
  --create-vm --prepare-dataset --run-training \
  --dataset-id your_dataset_id
```

---

## Cost Management

- **T4 + n1-standard-4**: ~$0.54/hour
- **Preemptible**: Add `--preemptible` for 60-80% savings
- **Stop VMs**: Always stop when not in use
- **Alerts**: GCP Console → Billing → Budgets & Alerts

## Troubleshooting

- **GPU not detected**: `nvidia-smi` on VM
- **Port/auth issues**: `gcloud auth login`
- **Quota**: `gcloud compute project-info describe`
