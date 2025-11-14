#!/usr/bin/env python3
"""
Verification script to check complete setup.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.gcs_utils import get_gcs_manager

def verify_local_data():
    """Verify local data directories"""
    print("Ì≥Å Checking local data directories...")
    
    required_dirs = [
        "data/raw",
        "data/processed",
        "data/evaluation"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        exists = Path(dir_path).exists()
        status = "‚úì" if exists else "‚ùå"
        print(f"  {status} {dir_path}")
        all_exist = all_exist and exists
    
    return all_exist

def verify_gcs_data():
    """Verify data in GCS"""
    print("\n‚òÅÔ∏è  Checking GCS buckets...")
    
    try:
        gcs_manager = get_gcs_manager("datasets")
        
        prefixes = ["raw/", "processed/", "evaluation/"]
        
        for prefix in prefixes:
            files = gcs_manager.list_files(prefix)
            status = "‚úì" if files else "‚ö†Ô∏è "
            print(f"  {status} {prefix}: {len(files)} files")
        
        return True
    except Exception as e:
        print(f"  ‚ùå Error accessing GCS: {e}")
        return False

def main():
    print("="*60)
    print("Ì¥ç Setup Verification")
    print("="*60)
    
    local_ok = verify_local_data()
    gcs_ok = verify_gcs_data()
    
    print("\n" + "="*60)
    if local_ok and gcs_ok:
        print("‚úì Setup verified successfully!")
    else:
        print("‚ö†Ô∏è  Some issues detected. Review output above.")
    print("="*60)
    
    return 0 if (local_ok and gcs_ok) else 1

if __name__ == "__main__":
    sys.exit(main())
