#!/usr/bin/env python3
"""
Convenience wrapper for unified data gathering system.

This script simply forwards all arguments to scripts/data_gatherer/data_gather.py
for easier command-line access from the scripts/ directory.

Usage:
    python scripts/gather_data.py --sources all
    python scripts/gather_data.py --datasets common_voice_17_0 primock57
"""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Forward all arguments to data_gatherer/data_gather.py."""
    data_gather_script = Path(__file__).parent / "data_gatherer" / "data_gather.py"
    
    if not data_gather_script.exists():
        print(f"Error: Main script not found at: {data_gather_script}", file=sys.stderr)
        return 1
    
    # Forward all command-line arguments
    result = subprocess.run(
        [sys.executable, str(data_gather_script)] + sys.argv[1:],
        cwd=Path(__file__).parent.parent,  # Run from workspace root
    )
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
