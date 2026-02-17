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
```

Edit `scripts/setup_gcp_gpu.sh` to set your `PROJECT_ID` before running.

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

1. **Create a VM** (using `scripts/setup_gcp_gpu.sh` or manual creation above).
2. **SSH into the VM**:
   ```bash
   gcloud compute ssh stt-gpu-vm --zone=us-central1-a
   ```
3. **Prepare your dataset** locally and upload to GCS:
   ```bash
   gsutil -m cp -r data/processed/your_dataset gs://your-project-datasets/
   ```
4. **On the VM**, clone the repo, install dependencies, and run fine-tuning:
   ```bash
   git clone <repo-url>
   cd Adaptive-Self-Learning-Agentic-AI-System
   pip install -r requirements.txt
   python scripts/finetune_wav2vec2.py --audio_dir /path/to/audio
   ```

---

## Cost Management

- **T4 + n1-standard-4**: ~$0.54/hour
- **Preemptible**: Add `--preemptible` to VM creation for 60-80% savings
- **Stop VMs**: Always stop when not in use:
  ```bash
  gcloud compute instances stop stt-gpu-vm --zone=us-central1-a
  ```
- **Monitor costs**: Use GCP Console → **Billing** → **Cost breakdown** and **Reports**
- **Set budgets**: GCP Console → **Billing** → **Budgets & alerts** → Create budget

## Troubleshooting

- **GPU not detected**: `nvidia-smi` on VM
- **Port/auth issues**: `gcloud auth login`
- **Quota**: `gcloud compute project-info describe`
