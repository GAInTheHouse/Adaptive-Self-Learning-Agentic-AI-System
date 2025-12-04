#!/usr/bin/env python3
"""
Run Fine-Tuning Orchestration Tests

Convenience script to run all fine-tuning orchestration tests.
"""

import sys
import subprocess
from pathlib import Path

# Test files for fine-tuning orchestration
ORCHESTRATION_TESTS = [
    "test_finetuning_orchestrator.py",
    "test_model_validator.py",
    "test_model_deployer.py",
    "test_regression_tester.py",
    "test_finetuning_coordinator.py"
]


def run_tests(verbose=True, coverage=False):
    """Run orchestration tests."""
    test_dir = Path(__file__).parent
    
    print("="*80)
    print("FINE-TUNING ORCHESTRATION TESTS")
    print("="*80)
    print()
    
    # Build pytest command
    cmd = ["pytest"]
    
    # Add test files
    for test_file in ORCHESTRATION_TESTS:
        cmd.append(str(test_dir / test_file))
    
    # Add options
    if verbose:
        cmd.append("-v")
    
    cmd.append("--tb=short")
    cmd.append("-m")
    cmd.append("unit")
    
    if coverage:
        cmd.extend([
            "--cov=src/data",
            "--cov-report=term-missing",
            "--cov-report=html"
        ])
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    # Run tests
    result = subprocess.run(cmd)
    
    print()
    print("="*80)
    if result.returncode == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80)
    
    return result.returncode


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run fine-tuning orchestration tests"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode (less verbose)"
    )
    
    args = parser.parse_args()
    
    returncode = run_tests(
        verbose=not args.quiet,
        coverage=args.coverage
    )
    
    sys.exit(returncode)


if __name__ == "__main__":
    main()

