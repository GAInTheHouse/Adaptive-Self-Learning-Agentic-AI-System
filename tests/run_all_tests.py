#!/usr/bin/env python
"""
Master test runner for all test suites
Runs unit tests, integration tests, and API tests with comprehensive reporting
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess
import argparse
from datetime import datetime
import json


class TestRunner:
    """Master test runner"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "test_suites": {},
            "summary": {
                "total_passed": 0,
                "total_failed": 0,
                "total_skipped": 0
            }
        }
    
    def run_pytest(self, test_path, description, markers=None):
        """Run pytest on a specific path"""
        print(f"\n{'=' * 70}")
        print(f"Running: {description}")
        print(f"{'=' * 70}\n")
        
        cmd = ["pytest", str(test_path), "-v", "--tb=short"]
        
        if markers:
            cmd.extend(["-m", markers])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse output for pass/fail counts (simple parsing)
            output = result.stdout
            
            passed = output.count(" PASSED")
            failed = output.count(" FAILED")
            skipped = output.count(" SKIPPED")
            
            self.results["test_suites"][description] = {
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "exit_code": result.returncode
            }
            
            self.results["summary"]["total_passed"] += passed
            self.results["summary"]["total_failed"] += failed
            self.results["summary"]["total_skipped"] += skipped
            
            print(output)
            
            if result.returncode == 0:
                print(f"\n✅ {description}: PASSED")
            else:
                print(f"\n❌ {description}: FAILED")
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ Error running {description}: {e}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        for suite, results in self.results["test_suites"].items():
            status = "✅ PASSED" if results["exit_code"] == 0 else "❌ FAILED"
            print(f"\n{suite}: {status}")
            print(f"  Passed: {results['passed']}")
            print(f"  Failed: {results['failed']}")
            print(f"  Skipped: {results['skipped']}")
        
        print(f"\n{'=' * 70}")
        print("OVERALL SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total Passed:  {self.results['summary']['total_passed']}")
        print(f"Total Failed:  {self.results['summary']['total_failed']}")
        print(f"Total Skipped: {self.results['summary']['total_skipped']}")
        
        total_tests = (
            self.results['summary']['total_passed'] +
            self.results['summary']['total_failed']
        )
        
        if total_tests > 0:
            pass_rate = (self.results['summary']['total_passed'] / total_tests) * 100
            print(f"Pass Rate:     {pass_rate:.1f}%")
        
        all_passed = all(
            suite["exit_code"] == 0
            for suite in self.results["test_suites"].values()
        )
        
        if all_passed and self.results['summary']['total_failed'] == 0:
            print(f"\n{'🎉' * 35}")
            print("ALL TESTS PASSED!")
            print(f"{'🎉' * 35}")
        else:
            print(f"\n⚠️  Some tests failed. Please review the output above.")
    
    def save_results(self, output_path="test_results.json"):
        """Save test results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📊 Test results saved to: {output_path}")


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Run test suites")
    parser.add_argument(
        "--suite",
        choices=["all", "unit", "integration", "api", "quick"],
        default="all",
        help="Test suite to run"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save test results to JSON"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    tests_dir = Path(__file__).parent
    
    print("=" * 70)
    print("ADAPTIVE STT SYSTEM - TEST SUITE RUNNER")
    print("=" * 70)
    print(f"Test Suite: {args.suite.upper()}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    all_passed = True
    
    if args.suite in ["all", "unit", "quick"]:
        # Run unit tests
        all_passed &= runner.run_pytest(
            tests_dir / "test_metrics.py",
            "Unit Tests: Metrics (WER/CER)"
        )
        
        all_passed &= runner.run_pytest(
            tests_dir / "test_error_detector.py",
            "Unit Tests: Error Detector"
        )
        
        all_passed &= runner.run_pytest(
            tests_dir / "test_benchmark.py",
            "Unit Tests: Benchmark"
        )
    
    if args.suite in ["all", "integration"]:
        # Run integration tests
        all_passed &= runner.run_pytest(
            tests_dir / "test_integration.py",
            "Integration Tests: Complete Workflow"
        )
    
    if args.suite in ["all", "api"]:
        # Run API tests
        print("\n⚠️  Note: API tests require the server to be running:")
        print("   uvicorn src.agent_api:app --port 8000\n")
        
        all_passed &= runner.run_pytest(
            tests_dir / "test_api_comprehensive.py",
            "API Tests: Comprehensive Endpoint Testing"
        )
    
    # Print summary
    runner.print_summary()
    
    # Save results if requested
    if args.save_results:
        runner.save_results()
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

