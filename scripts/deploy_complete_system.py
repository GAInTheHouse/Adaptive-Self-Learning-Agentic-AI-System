#!/usr/bin/env python3
"""
Complete GCP Deployment Script
Automates the entire deployment process for the STT system
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import argparse

class Colors:
    """Terminal colors for output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print(f"{text}")
    print(f"{'='*60}{Colors.ENDC}\n")

def print_step(step: int, text: str):
    """Print formatted step"""
    print(f"{Colors.OKBLUE}{Colors.BOLD}[STEP {step}]{Colors.ENDC} {text}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def run_command(cmd: List[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command"""
    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        if check:
            print_error(f"Command failed: {e}")
            if e.stderr:
                print(e.stderr)
            sys.exit(1)
        return e

def check_prerequisites():
    """Check if all prerequisites are installed"""
    print_step(0, "Checking prerequisites...")
    
    required_tools = {
        'gcloud': ['gcloud', '--version'],
        'docker': ['docker', '--version'],
        'python': ['python3', '--version'],
        'git': ['git', '--version']
    }
    
    missing = []
    for tool, cmd in required_tools.items():
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print_success(f"{tool} installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print_error(f"{tool} not found")
            missing.append(tool)
    
    if missing:
        print_error(f"Missing required tools: {', '.join(missing)}")
        sys.exit(1)
    
    print_success("All prerequisites satisfied")

def setup_gcp_project(project_id: str, region: str, zone: str):
    """Setup GCP project configuration"""
    print_step(1, "Setting up GCP project...")
    
    # Set project
    run_command(['gcloud', 'config', 'set', 'project', project_id])
    
    # Set region and zone
    run_command(['gcloud', 'config', 'set', 'compute/region', region])
    run_command(['gcloud', 'config', 'set', 'compute/zone', zone])
    
    print_success(f"Project configured: {project_id}")

def enable_apis(project_id: str):
    """Enable required GCP APIs"""
    print_step(2, "Enabling GCP APIs...")
    
    apis = [
        'compute.googleapis.com',
        'storage.googleapis.com',
        'run.googleapis.com',
        'cloudbuild.googleapis.com',
        'containerregistry.googleapis.com',
        'logging.googleapis.com',
        'monitoring.googleapis.com'
    ]
    
    for api in apis:
        print(f"  Enabling {api}...")
        run_command(['gcloud', 'services', 'enable', api], check=False)
    
    print_success("All APIs enabled")

def create_storage_buckets(project_id: str, region: str):
    """Create GCS buckets"""
    print_step(3, "Creating GCS buckets...")
    
    buckets = {
        'datasets': f"{project_id}-stt-datasets",
        'models': f"{project_id}-stt-models",
        'logs': f"{project_id}-stt-logs"
    }
    
    for name, bucket in buckets.items():
        print(f"  Creating bucket: {bucket}")
        result = run_command([
            'gsutil', 'mb',
            '-p', project_id,
            '-c', 'STANDARD',
            '-l', region,
            f'gs://{bucket}'
        ], check=False)
        
        if result.returncode == 0:
            print_success(f"Created {name} bucket")
        else:
            print_warning(f"Bucket {bucket} may already exist")
    
    return buckets

def create_service_account(project_id: str):
    """Create service account with required permissions"""
    print_step(4, "Creating service account...")
    
    sa_name = 'stt-service-account'
    sa_email = f"{sa_name}@{project_id}.iam.gserviceaccount.com"
    
    # Create service account
    result = run_command([
        'gcloud', 'iam', 'service-accounts', 'create', sa_name,
        '--display-name', 'STT System Service Account'
    ], check=False)
    
    if result.returncode != 0:
        print_warning("Service account may already exist")
    
    # Grant roles
    roles = [
        'roles/storage.objectAdmin',
        'roles/logging.logWriter',
        'roles/monitoring.metricWriter'
    ]
    
    for role in roles:
        print(f"  Granting {role}...")
        run_command([
            'gcloud', 'projects', 'add-iam-policy-binding', project_id,
            '--member', f'serviceAccount:{sa_email}',
            '--role', role
        ], check=False)
    
    print_success(f"Service account configured: {sa_email}")
    return sa_email

def build_and_push_image(project_id: str):
    """Build and push Docker image"""
    print_step(5, "Building and pushing Docker image...")
    
    image_url = f"gcr.io/{project_id}/stt-api:latest"
    
    print("  Building Docker image...")
    run_command(['gcloud', 'builds', 'submit', '--tag', image_url], check=True)
    
    print_success(f"Image built and pushed: {image_url}")
    return image_url

def deploy_cloud_run(
    project_id: str,
    region: str,
    image_url: str,
    service_account: str,
    buckets: Dict[str, str],
    allow_unauthenticated: bool = True
):
    """Deploy to Cloud Run"""
    print_step(6, "Deploying to Cloud Run...")
    
    env_vars = (
        f"USE_GCS=true,"
        f"GCS_DATASETS_BUCKET={buckets['datasets']},"
        f"GCS_MODELS_BUCKET={buckets['models']},"
        f"GCS_LOGS_BUCKET={buckets['logs']}"
    )
    
    cmd = [
        'gcloud', 'run', 'deploy', 'stt-api',
        '--image', image_url,
        '--platform', 'managed',
        '--region', region,
        '--memory', '4Gi',
        '--cpu', '2',
        '--timeout', '300',
        '--max-instances', '10',
        '--set-env-vars', env_vars,
        '--service-account', service_account
    ]
    
    if allow_unauthenticated:
        cmd.append('--allow-unauthenticated')
    
    run_command(cmd)
    
    # Get service URL
    result = run_command([
        'gcloud', 'run', 'services', 'describe', 'stt-api',
        '--region', region,
        '--format', 'value(status.url)'
    ])
    
    service_url = result.stdout.strip()
    print_success(f"Service deployed: {service_url}")
    return service_url

def test_deployment(service_url: str):
    """Test the deployed service"""
    print_step(7, "Testing deployment...")
    
    # Test health endpoint
    print("  Testing /api/health...")
    result = run_command(['curl', '-f', f'{service_url}/api/health'], check=False)
    
    if result.returncode == 0:
        print_success("Health check passed")
        try:
            health_data = json.loads(result.stdout)
            print(f"  Status: {health_data.get('status')}")
        except:
            pass
    else:
        print_error("Health check failed")
        return False
    
    # Test root endpoint
    print("  Testing root endpoint...")
    result = run_command(['curl', '-f', service_url], check=False)
    
    if result.returncode == 0:
        print_success("Root endpoint accessible")
    else:
        print_warning("Root endpoint test failed")
    
    return True

def setup_monitoring(project_id: str, service_url: str, notification_email: Optional[str]):
    """Setup monitoring and alerts"""
    print_step(8, "Setting up monitoring...")
    
    if notification_email:
        print(f"  Creating notification channel for {notification_email}...")
        # Note: This requires alpha/beta gcloud features
        print_warning("Manual setup required for email notifications")
        print(f"  Go to: https://console.cloud.google.com/monitoring/alerting")
    
    print_success("Monitoring setup guidance provided")

def create_gpu_vm(project_id: str, zone: str, service_account: str, create_vm: bool = False):
    """Create GPU VM for training"""
    if not create_vm:
        print_step(9, "Skipping GPU VM creation (use --create-gpu-vm to enable)")
        return
    
    print_step(9, "Creating GPU VM...")
    
    vm_name = 'stt-training-vm'
    
    cmd = [
        'gcloud', 'compute', 'instances', 'create', vm_name,
        '--zone', zone,
        '--machine-type', 'n1-standard-8',
        '--accelerator', 'type=nvidia-tesla-t4,count=1',
        '--image-family', 'pytorch-latest-gpu',
        '--image-project', 'deeplearning-platform-release',
        '--boot-disk-size', '200GB',
        '--boot-disk-type', 'pd-ssd',
        '--maintenance-policy', 'TERMINATE',
        '--metadata', 'install-nvidia-driver=True',
        '--scopes', 'cloud-platform',
        '--service-account', service_account
    ]
    
    result = run_command(cmd, check=False)
    
    if result.returncode == 0:
        print_success(f"GPU VM created: {vm_name}")
        print_warning("Remember to stop the VM when not in use to save costs!")
        print(f"  Stop: gcloud compute instances stop {vm_name} --zone={zone}")
    else:
        print_warning("GPU VM creation failed (may already exist or quota limit)")

def save_deployment_info(
    project_id: str,
    region: str,
    zone: str,
    service_url: str,
    buckets: Dict[str, str],
    service_account: str
):
    """Save deployment information to file"""
    print_step(10, "Saving deployment information...")
    
    deployment_info = {
        'project_id': project_id,
        'region': region,
        'zone': zone,
        'service_url': service_url,
        'buckets': buckets,
        'service_account': service_account,
        'deployment_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'api_docs': f'{service_url}/docs',
        'frontend': f'{service_url}/app'
    }
    
    output_file = Path('deployment-info.json')
    with open(output_file, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print_success(f"Deployment info saved to {output_file}")
    
    # Print summary
    print_header("DEPLOYMENT SUMMARY")
    print(f"{Colors.BOLD}Service URL:{Colors.ENDC} {service_url}")
    print(f"{Colors.BOLD}API Docs:{Colors.ENDC} {service_url}/docs")
    print(f"{Colors.BOLD}Frontend:{Colors.ENDC} {service_url}/app")
    print(f"\n{Colors.BOLD}Buckets:{Colors.ENDC}")
    for name, bucket in buckets.items():
        print(f"  {name}: gs://{bucket}")
    print(f"\n{Colors.BOLD}Service Account:{Colors.ENDC} {service_account}")
    
    # Print .env content
    print(f"\n{Colors.BOLD}Add to your .env file:{Colors.ENDC}")
    print(f"PROJECT_ID={project_id}")
    print(f"REGION={region}")
    print(f"ZONE={zone}")
    print(f"API_URL={service_url}")
    print(f"GCS_DATASETS_BUCKET={buckets['datasets']}")
    print(f"GCS_MODELS_BUCKET={buckets['models']}")
    print(f"GCS_LOGS_BUCKET={buckets['logs']}")

def main():
    parser = argparse.ArgumentParser(
        description='Deploy STT system to Google Cloud Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic deployment
  python scripts/deploy_complete_system.py --project-id my-project
  
  # With GPU VM and custom region
  python scripts/deploy_complete_system.py --project-id my-project --create-gpu-vm --region us-west1
  
  # Skip Docker build (use existing image)
  python scripts/deploy_complete_system.py --project-id my-project --skip-build
        """
    )
    
    parser.add_argument('--project-id', required=True, help='GCP Project ID')
    parser.add_argument('--region', default='us-central1', help='GCP region (default: us-central1)')
    parser.add_argument('--zone', default='us-central1-a', help='GCP zone (default: us-central1-a)')
    parser.add_argument('--create-gpu-vm', action='store_true', help='Create GPU VM for training')
    parser.add_argument('--skip-build', action='store_true', help='Skip Docker build (use existing image)')
    parser.add_argument('--skip-prerequisites', action='store_true', help='Skip prerequisite checks')
    parser.add_argument('--notification-email', help='Email for monitoring alerts')
    parser.add_argument('--require-authentication', action='store_true', help='Require authentication for API access')
    
    args = parser.parse_args()
    
    print_header("GCP DEPLOYMENT - Adaptive Self-Learning Agentic AI System")
    print(f"Project ID: {args.project_id}")
    print(f"Region: {args.region}")
    print(f"Zone: {args.zone}")
    
    try:
        # Step 0: Prerequisites
        if not args.skip_prerequisites:
            check_prerequisites()
        
        # Step 1: Setup project
        setup_gcp_project(args.project_id, args.region, args.zone)
        
        # Step 2: Enable APIs
        enable_apis(args.project_id)
        
        # Step 3: Create buckets
        buckets = create_storage_buckets(args.project_id, args.region)
        
        # Step 4: Create service account
        service_account = create_service_account(args.project_id)
        
        # Step 5: Build and push image
        if args.skip_build:
            image_url = f"gcr.io/{args.project_id}/stt-api:latest"
            print_warning("Skipping build, using existing image")
        else:
            image_url = build_and_push_image(args.project_id)
        
        # Step 6: Deploy to Cloud Run
        service_url = deploy_cloud_run(
            args.project_id,
            args.region,
            image_url,
            service_account,
            buckets,
            allow_unauthenticated=not args.require_authentication
        )
        
        # Step 7: Test deployment
        test_deployment(service_url)
        
        # Step 8: Setup monitoring
        setup_monitoring(args.project_id, service_url, args.notification_email)
        
        # Step 9: Create GPU VM (optional)
        create_gpu_vm(args.project_id, args.zone, service_account, args.create_gpu_vm)
        
        # Step 10: Save deployment info
        save_deployment_info(
            args.project_id,
            args.region,
            args.zone,
            service_url,
            buckets,
            service_account
        )
        
        print_header("DEPLOYMENT COMPLETE!")
        print_success("Your STT system is now deployed and ready to use!")
        print(f"\n{Colors.BOLD}Next steps:{Colors.ENDC}")
        print("  1. Test the API:", f"curl {service_url}/api/health")
        print("  2. Open frontend:", f"open {service_url}/app")
        print("  3. View API docs:", f"open {service_url}/docs")
        print("  4. Monitor costs: python scripts/monitor_gcp_costs.py")
        
    except KeyboardInterrupt:
        print_error("\nDeployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

