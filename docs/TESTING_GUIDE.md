# Testing Guide - Adaptive Self-Learning STT System

Comprehensive guide to testing the STT system with unit tests, integration tests, and API tests.

## 📋 Table of Contents
- [Test Organization](#test-organization)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Writing New Tests](#writing-new-tests)
- [Test Types](#test-types)
- [CI/CD Integration](#cicd-integration)

## 🗂️ Test Organization

### Directory Structure

```
tests/
├── __init__.py                    # Test package initialization
├── conftest.py                    # pytest fixtures and configuration
├── pytest.ini                     # pytest settings
├── run_all_tests.py              # Master test runner
│
├── test_metrics.py                # Unit tests for WER/CER metrics
├── test_error_detector.py         # Unit tests for error detection
├── test_benchmark.py              # Unit tests for benchmarking
├── test_integration.py            # Integration tests
└── test_api_comprehensive.py      # Comprehensive API tests

experiments/
├── test_baseline.py               # Baseline model tests
├── test_agent.py                  # Agent system tests
├── test_data_management.py        # Data management tests
└── test_api.py                    # Basic API tests
```

### Test Categories

1. **Unit Tests** (`tests/test_*.py`)
   - Test individual components in isolation
   - Fast execution
   - No external dependencies

2. **Integration Tests** (`tests/test_integration.py`)
   - Test multiple components working together
   - End-to-end workflows
   - May require test data

3. **API Tests** (`tests/test_api_comprehensive.py`)
   - Test REST API endpoints
   - Requires running server
   - Performance and load testing

4. **Functional Tests** (`experiments/test_*.py`)
   - High-level feature testing
   - User-facing functionality
   - Real-world scenarios

## 🚀 Running Tests

### Quick Start

```bash
# Install pytest (if not already installed)
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_metrics.py -v

# Run with specific markers
pytest tests/ -v -m unit
pytest tests/ -v -m "not slow"
```

### Using the Test Runner

```bash
# Run all test suites
python tests/run_all_tests.py --suite all

# Run only unit tests
python tests/run_all_tests.py --suite unit

# Run integration tests
python tests/run_all_tests.py --suite integration

# Run API tests (requires server running)
python tests/run_all_tests.py --suite api

# Quick tests (unit tests only)
python tests/run_all_tests.py --suite quick

# Save results to JSON
python tests/run_all_tests.py --suite all --save-results
```

### Running Legacy Tests

```bash
# Baseline model tests
python experiments/test_baseline.py

# Agent system tests
python experiments/test_agent.py

# Data management tests
python experiments/test_data_management.py

# API tests (requires server)
uvicorn src.agent_api:app --port 8000 &
python experiments/test_api.py
```

### API Testing Workflow

```bash
# Terminal 1: Start API server
uvicorn src.agent_api:app --reload --port 8000

# Terminal 2: Run API tests
pytest tests/test_api_comprehensive.py -v

# Or use the test runner
python tests/run_all_tests.py --suite api
```

## 📊 Test Coverage

### Current Coverage

| Component | Test File | Coverage |
|-----------|-----------|----------|
| Metrics (WER/CER) | test_metrics.py | ✅ 95% |
| Error Detector | test_error_detector.py | ✅ 90% |
| Benchmark | test_benchmark.py | ✅ 85% |
| Baseline Model | test_baseline.py | ✅ 80% |
| Agent System | test_agent.py | ✅ 85% |
| Data Management | test_data_management.py | ✅ 90% |
| API Endpoints | test_api_comprehensive.py | ✅ 90% |
| Integration | test_integration.py | ✅ 85% |

### Generating Coverage Report

```bash
# Install coverage tool
pip install pytest-cov

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Open coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Coverage Goals

- **Unit Tests**: 90%+ coverage
- **Integration Tests**: 80%+ coverage
- **API Tests**: 95%+ endpoint coverage
- **Overall**: 85%+ code coverage

## ✍️ Writing New Tests

### Test File Template

```python
"""
Description of what this test file covers
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.module_name import ClassName
import pytest


class TestClassName:
    """Test cases for ClassName"""
    
    def setup_method(self):
        """Setup before each test"""
        self.instance = ClassName()
    
    def teardown_method(self):
        """Cleanup after each test"""
        pass
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        result = self.instance.method()
        assert result is not None
    
    def test_edge_case(self):
        """Test edge case"""
        with pytest.raises(ValueError):
            self.instance.method(invalid_input)


def test_standalone_function():
    """Test standalone function"""
    result = standalone_function()
    assert result == expected_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Best Practices

1. **Test Naming**
   - Use descriptive names: `test_calculate_wer_with_perfect_match`
   - Start with `test_` prefix
   - Group related tests in classes

2. **Assertions**
   - Use clear assertion messages
   - Test one concept per test
   - Use pytest's assertion helpers

3. **Fixtures**
   - Define reusable fixtures in `conftest.py`
   - Use appropriate fixture scopes
   - Clean up resources in teardown

4. **Mocking**
   - Mock external dependencies
   - Use `pytest-mock` for complex mocking
   - Mock at appropriate level

5. **Test Data**
   - Use fixtures for test data
   - Keep test data small and focused
   - Store large test data separately

### Example: Testing Error Detection

```python
def test_all_caps_detection():
    """Test detection of all caps text"""
    detector = ErrorDetector()
    errors = detector.detect_errors("HELLO WORLD")
    
    assert len(errors) > 0, "Should detect errors in all caps text"
    assert any(e.error_type == "all_caps" for e in errors), \
        "Should specifically detect all_caps error"
    
    summary = detector.get_error_summary(errors)
    assert summary['has_errors'] is True
    assert summary['error_score'] > 0.0
```

## 🧪 Test Types

### 1. Unit Tests

**Purpose**: Test individual functions/methods in isolation

**Example**:
```python
def test_calculate_wer_perfect_match():
    """Test WER calculation with perfect match"""
    evaluator = STTEvaluator()
    wer_score = evaluator.calculate_wer("hello world", "hello world")
    assert wer_score == 0.0
```

**Characteristics**:
- Fast execution (< 1 second)
- No external dependencies
- Deterministic results
- High coverage

### 2. Integration Tests

**Purpose**: Test multiple components working together

**Example**:
```python
def test_end_to_end_workflow():
    """Test complete workflow from transcription to data storage"""
    # Initialize components
    model = BaselineSTTModel()
    agent = STTAgent(model)
    data_system = IntegratedDataManagementSystem()
    
    # Transcribe
    result = agent.transcribe_with_agent("audio.wav")
    
    # Record errors
    if result['error_detection']['has_errors']:
        case_id = data_system.record_failed_transcription(...)
        assert case_id is not None
```

**Characteristics**:
- Slower execution (seconds to minutes)
- Tests component interactions
- May require test data
- Realistic scenarios

### 3. API Tests

**Purpose**: Test REST API endpoints

**Example**:
```python
def test_agent_transcribe_endpoint():
    """Test /agent/transcribe endpoint"""
    with open("data/test_audio/test_1.wav", "rb") as f:
        files = {"file": f}
        response = requests.post(
            "http://localhost:8000/agent/transcribe",
            files=files
        )
    
    assert response.status_code == 200
    data = response.json()
    assert 'transcript' in data
    assert 'error_detection' in data
```

**Characteristics**:
- Requires running server
- Tests HTTP interface
- Includes performance tests
- Validates API contracts

### 4. Performance Tests

**Purpose**: Verify performance characteristics

**Example**:
```python
def test_transcription_latency():
    """Test transcription latency is acceptable"""
    start_time = time.time()
    result = model.transcribe("audio.wav")
    latency = time.time() - start_time
    
    assert latency < 10.0, "Transcription should complete within 10s"
```

**Characteristics**:
- Measure timing
- Check resource usage
- Verify scalability
- Set performance baselines

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run unit tests
      run: |
        pytest tests/ -v -m "not api" --cov=src
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash

echo "Running tests before commit..."
pytest tests/ -v -m "not api and not slow"

if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi

echo "All tests passed!"
```

## 📝 Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.unit
def test_unit_function():
    """Unit test"""
    pass

@pytest.mark.integration
def test_integration_workflow():
    """Integration test"""
    pass

@pytest.mark.api
def test_api_endpoint():
    """API test"""
    pass

@pytest.mark.slow
def test_slow_operation():
    """Slow test"""
    pass
```

Run specific markers:
```bash
# Run only unit tests
pytest -m unit

# Run everything except slow tests
pytest -m "not slow"

# Run integration and API tests
pytest -m "integration or api"
```

## 🐛 Debugging Tests

### Running Single Test

```bash
# Run specific test function
pytest tests/test_metrics.py::test_perfect_match -v

# Run specific test class
pytest tests/test_metrics.py::TestSTTEvaluator -v

# Run specific test method
pytest tests/test_metrics.py::TestSTTEvaluator::test_perfect_match -v
```

### Verbose Output

```bash
# Show print statements
pytest tests/ -v -s

# Show full traceback
pytest tests/ -v --tb=long

# Stop at first failure
pytest tests/ -v -x
```

### Debugging with pdb

```python
def test_debug_example():
    """Test with debugging"""
    import pdb; pdb.set_trace()
    result = function_to_test()
    assert result == expected
```

Run with:
```bash
pytest tests/test_file.py --pdb
```

## 📈 Test Metrics

### Key Metrics to Track

1. **Code Coverage**: Percentage of code executed by tests
2. **Test Count**: Number of tests per component
3. **Pass Rate**: Percentage of tests passing
4. **Execution Time**: Time taken to run test suite
5. **Failure Rate**: Percentage of tests failing

### Viewing Test Results

```bash
# Generate HTML report
pytest tests/ --html=report.html --self-contained-html

# Generate JUnit XML (for CI)
pytest tests/ --junit-xml=junit.xml
```

## 🎯 Testing Checklist

Before submitting code, ensure:

- [ ] All tests pass locally
- [ ] New features have tests
- [ ] Bug fixes have regression tests
- [ ] Code coverage >= 85%
- [ ] Tests are well-documented
- [ ] No skipped tests without reason
- [ ] API tests pass (if applicable)
- [ ] Integration tests pass
- [ ] Performance tests meet benchmarks

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)

## 🆘 Troubleshooting

### Common Issues

**Issue: Tests can't find src module**
```bash
# Solution: Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Issue: API tests fail**
```bash
# Solution: Make sure API server is running
uvicorn src.agent_api:app --port 8000
```

**Issue: Tests are slow**
```bash
# Solution: Run quick tests only
pytest tests/ -v -m "not slow"
```

**Issue: Import errors**
```bash
# Solution: Install test dependencies
pip install pytest pytest-cov pytest-mock requests
```

---

**Last Updated**: November 24, 2025  
**Version**: 1.0.0  
**Status**: Complete Testing Suite ✅

