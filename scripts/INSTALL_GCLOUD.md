# Installing Google Cloud SDK (gcloud CLI)

## For macOS (Your System)

### Option 1: Using Homebrew (Recommended - Easiest)

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install gcloud CLI
brew install --cask google-cloud-sdk

# Initialize gcloud
gcloud init
```

### Option 2: Direct Download

```bash
# Download and install
curl https://sdk.cloud.google.com | bash

# Restart your shell or run:
exec -l $SHELL

# Initialize
gcloud init
```

### Option 3: Using the Installer Script

```bash
# Download installer
curl https://sdk.cloud.google.com | bash

# Follow the prompts, then:
gcloud init
```

## After Installation

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
   gcloud services enable storage-api.googleapis.com
   ```

4. **Verify installation**:
   ```bash
   gcloud --version
   gcloud compute zones list
   ```

## Quick Test

```bash
# Check if gcloud works
gcloud --version

# List available zones
gcloud compute zones list | grep us-central
```

## Next Steps

Once gcloud is installed:
1. Run: `bash scripts/setup_gcp_gpu.sh`
2. Or follow the manual setup in `docs/GCP_SETUP_GUIDE.md`

## Troubleshooting

- **Permission errors**: May need to add gcloud to PATH
- **Authentication issues**: Run `gcloud auth login`
- **Project not found**: Make sure you have access to the project

## Official Documentation

- [Install Guide](https://cloud.google.com/sdk/docs/install)
- [Quick Start](https://cloud.google.com/sdk/docs/quickstart)

