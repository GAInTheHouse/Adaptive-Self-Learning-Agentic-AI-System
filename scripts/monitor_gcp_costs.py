#!/usr/bin/env python3
"""
Monitor GCP costs and usage for the STT project.
Shows current VM usage, storage costs, and provides cost estimates.
"""

import subprocess
import json
from datetime import datetime, timedelta
from typing import Dict, List

PROJECT_ID = "stt-agentic-ai-2025"
VM_NAME = "stt-gpu-vm"
ZONE = "us-central1-a"

def get_vm_status() -> Dict:
    """Get VM status and uptime."""
    try:
        result = subprocess.run(
            ["gcloud", "compute", "instances", "describe", VM_NAME,
             "--zone", ZONE, "--format", "json"],
            capture_output=True,
            text=True,
            check=True
        )
        vm_info = json.loads(result.stdout)
        
        status = vm_info.get("status", "UNKNOWN")
        machine_type = vm_info.get("machineType", "").split("/")[-1]
        
        # Get creation timestamp
        creation_time = vm_info.get("creationTimestamp", "")
        
        return {
            "status": status,
            "machine_type": machine_type,
            "created": creation_time,
            "exists": True
        }
    except subprocess.CalledProcessError:
        return {"exists": False}
    except FileNotFoundError:
        print("❌ gcloud CLI not found")
        return {}

def estimate_vm_cost(hours: float, machine_type: str = "n1-standard-4") -> float:
    """Estimate VM cost based on machine type."""
    # Approximate hourly costs (as of 2024)
    costs = {
        "n1-standard-4": 0.19,  # $/hour
        "n1-standard-8": 0.38,
        "n1-highmem-4": 0.24,
        "n1-highmem-8": 0.48,
    }
    
    base_cost = costs.get(machine_type, 0.19)
    gpu_cost = 0.35  # T4 GPU cost per hour
    
    return (base_cost + gpu_cost) * hours

def get_storage_usage() -> Dict:
    """Get GCS storage usage."""
    buckets = [
        "stt-project-datasets",
        "stt-project-models",
        "stt-project-logs"
    ]
    
    total_size_gb = 0
    bucket_sizes = {}
    
    for bucket in buckets:
        try:
            result = subprocess.run(
                ["gsutil", "du", "-sh", f"gs://{bucket}"],
                capture_output=True,
                text=True,
                check=True
            )
            # Parse output (format: "SIZE\tgs://bucket")
            size_str = result.stdout.split()[0]
            if "G" in size_str:
                size_gb = float(size_str.replace("G", ""))
            elif "M" in size_str:
                size_gb = float(size_str.replace("M", "")) / 1024
            else:
                size_gb = 0
            
            bucket_sizes[bucket] = size_gb
            total_size_gb += size_gb
        except (subprocess.CalledProcessError, FileNotFoundError):
            bucket_sizes[bucket] = 0
    
    # Storage cost: ~$0.02/GB/month
    monthly_storage_cost = total_size_gb * 0.02
    
    return {
        "total_gb": total_size_gb,
        "bucket_sizes": bucket_sizes,
        "monthly_cost": monthly_storage_cost
    }

def get_billing_info() -> Dict:
    """Get current billing information."""
    try:
        result = subprocess.run(
            ["gcloud", "billing", "accounts", "list", "--format", "json"],
            capture_output=True,
            text=True,
            check=True
        )
        accounts = json.loads(result.stdout)
        return {"accounts": accounts}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}

def calculate_uptime_hours(creation_time: str, status: str) -> float:
    """Calculate VM uptime in hours."""
    if status != "RUNNING":
        return 0.0
    
    try:
        from dateutil import parser
        created = parser.parse(creation_time)
        now = datetime.now(created.tzinfo)
        delta = now - created
        return delta.total_seconds() / 3600
    except:
        return 0.0

def main():
    print("="*60)
    print("GCP Cost & Usage Monitor")
    print("="*60)
    print(f"Project: {PROJECT_ID}\n")
    
    # VM Status
    print("🖥️  VM Status:")
    print("-" * 60)
    vm_status = get_vm_status()
    
    if not vm_status.get("exists"):
        print(f"  ❌ VM '{VM_NAME}' not found")
        print(f"     Run 'bash scripts/setup_gcp_gpu.sh' to create it")
    else:
        status = vm_status.get("status", "UNKNOWN")
        machine_type = vm_status.get("machine_type", "unknown")
        created = vm_status.get("created", "")
        
        print(f"  Status: {status}")
        print(f"  Machine Type: {machine_type}")
        print(f"  Created: {created}")
        
        if status == "RUNNING":
            uptime_hours = calculate_uptime_hours(created, status)
            estimated_cost = estimate_vm_cost(uptime_hours, machine_type)
            print(f"  Uptime: {uptime_hours:.2f} hours")
            print(f"  Estimated Cost: ${estimated_cost:.2f}")
            print(f"  💡 Stop VM when not in use to save costs!")
    
    # Storage Usage
    print("\n💾 Storage Usage:")
    print("-" * 60)
    storage = get_storage_usage()
    print(f"  Total Storage: {storage['total_gb']:.2f} GB")
    print(f"  Monthly Cost: ${storage['monthly_cost']:.2f}")
    print("\n  Per Bucket:")
    for bucket, size in storage['bucket_sizes'].items():
        print(f"    {bucket}: {size:.2f} GB")
    
    # Cost Estimates
    print("\n💰 Cost Estimates:")
    print("-" * 60)
    print("  VM (T4 GPU + n1-standard-4):")
    print("    - Per hour: ~$0.54")
    print("    - Per day (24h): ~$12.96")
    print("    - Per month (730h): ~$394.20")
    print("\n  Storage:")
    print(f"    - Current: ${storage['monthly_cost']:.2f}/month")
    print("\n  💡 Tips to Save:")
    print("    - Use preemptible instances: 60-80% cheaper")
    print("    - Stop VMs when not in use")
    print("    - Use smaller GPUs (T4) for development")
    print("    - Clean up old model checkpoints")
    
    # Billing
    print("\n💳 Billing:")
    print("-" * 60)
    billing = get_billing_info()
    if billing.get("accounts"):
        print(f"  Found {len(billing['accounts'])} billing account(s)")
        print("  View detailed costs: https://console.cloud.google.com/billing")
    else:
        print("  ⚠️  Could not retrieve billing info")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()

