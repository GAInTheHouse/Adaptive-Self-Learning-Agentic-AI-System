# GitHub Actions Workflows

This directory contains GitHub Actions workflows for automated testing and CI/CD.

## 📋 Available Workflows

### 1. Tests (`tests.yml`)
**Trigger**: Push to main/develop branches, Pull Requests  
**Purpose**: Comprehensive test suite for all commits

**What it does**:
- Runs on Python 3.8, 3.9, 3.10 (matrix testing)
- Executes unit tests
- Executes integration tests
- Executes functional tests
- Generates coverage reports
- Uploads coverage to Codecov
- Runs linting checks (flake8, black)
- Tests API endpoints

**Jobs**:
1. `test`: Main test suite
2. `lint`: Code quality checks
3. `api-test`: API endpoint testing

### 2. Quick Tests (`quick-test.yml`)
**Trigger**: Every push to any branch  
**Purpose**: Fast feedback on basic tests

**What it does**:
- Runs only fast unit tests
- Uses CPU-only PyTorch for speed
- Provides quick pass/fail feedback
- Completes in ~2-3 minutes

### 3. Weekly Full Test Suite (`weekly-full-test.yml`)
**Trigger**: Weekly (Sunday at 00:00 UTC) or manual  
**Purpose**: Comprehensive testing with coverage reports

**What it does**:
- Runs all tests (including slow tests)
- Generates HTML test reports
- Generates comprehensive coverage reports
- Checks coverage threshold (>80%)
- Uploads artifacts for review
- Parallel test execution

## 🚀 Workflow Triggers

### Automatic Triggers

```yaml
# On every push to specific branches
on:
  push:
    branches: [ main, master, develop ]

# On pull requests
on:
  pull_request:
    branches: [ main, master ]

# On schedule (weekly)
on:
  schedule:
    - cron: '0 0 * * 0'
```

### Manual Triggers

All workflows can be triggered manually from the GitHub Actions tab:
1. Go to repository → Actions tab
2. Select workflow from left sidebar
3. Click "Run workflow" button

## 📊 Understanding Test Results

### Status Badges

Add these to your README.md:

```markdown
![Tests](https://github.com/YOUR_USERNAME/YOUR_REPO/workflows/Tests/badge.svg)
![Quick Tests](https://github.com/YOUR_USERNAME/YOUR_REPO/workflows/Quick%20Tests/badge.svg)
```

### Viewing Results

**In Pull Requests**:
- Test status appears as checks at the bottom
- Click "Details" to see full logs
- Review coverage changes

**In Actions Tab**:
- View all workflow runs
- Download test artifacts
- See coverage reports
- Review detailed logs

## 🔧 Configuration

### Modifying Workflows

**Change Python versions**:
```yaml
strategy:
  matrix:
    python-version: ['3.8', '3.9', '3.10', '3.11']
```

**Change test scope**:
```yaml
- name: Run tests
  run: |
    pytest tests/ -v -m "not slow and not integration"
```

**Add more checks**:
```yaml
- name: Security check
  run: |
    pip install bandit
    bandit -r src/
```

### Secrets and Environment Variables

**Add secrets** (for API keys, credentials):
1. Go to Settings → Secrets and variables → Actions
2. Add new repository secret
3. Use in workflow:

```yaml
env:
  API_KEY: ${{ secrets.API_KEY }}
```

## 📝 Workflow Files

### tests.yml - Main Test Suite

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
      - name: Run tests
        run: pytest tests/ -v
```

### Quick Test Pattern

```yaml
name: Quick Tests
on:
  push:
    branches: [ '*' ]
jobs:
  quick-test:
    runs-on: ubuntu-latest
    steps:
      - name: Run quick tests
        run: pytest -m "not slow" -v
```

## 🎯 Best Practices

### 1. Fast Feedback
- Keep quick tests under 5 minutes
- Run expensive tests weekly or on release

### 2. Parallel Testing
```yaml
- name: Run tests in parallel
  run: pytest -n auto
```

### 3. Caching
```yaml
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
```

### 4. Conditional Execution
```yaml
- name: Run if Python files changed
  if: contains(github.event.head_commit.modified, '.py')
  run: pytest tests/
```

### 5. Matrix Strategy
```yaml
strategy:
  matrix:
    python-version: [3.8, 3.9, 3.10]
    os: [ubuntu-latest, macos-latest, windows-latest]
```

## 🔍 Debugging Failed Workflows

### Check Logs
1. Go to Actions tab
2. Click failed workflow run
3. Expand failed step
4. Review error messages

### Run Locally
```bash
# Simulate GitHub Actions environment
act -j test  # requires 'act' tool

# Or run tests directly
pytest tests/ -v
```

### Common Issues

**1. Dependency Installation Fails**
```yaml
- name: Install dependencies
  run: |
    pip install --upgrade pip
    pip install wheel setuptools
    pip install -r requirements.txt
```

**2. Tests Timeout**
```yaml
- name: Run tests
  timeout-minutes: 30
  run: pytest tests/
```

**3. Import Errors**
```yaml
- name: Setup Python path
  run: echo "PYTHONPATH=$PYTHONPATH:$(pwd)" >> $GITHUB_ENV
```

## 📈 Coverage Reporting

### Codecov Integration

```yaml
- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
    fail_ci_if_error: false
```

### Coverage Badge
```markdown
[![codecov](https://codecov.io/gh/USERNAME/REPO/branch/main/graph/badge.svg)](https://codecov.io/gh/USERNAME/REPO)
```

## 🔐 Security Scanning

### Add Dependency Scanning

```yaml
- name: Check dependencies
  run: |
    pip install safety
    safety check
```

### Add Code Security Scan

```yaml
- name: Run Bandit
  run: |
    pip install bandit
    bandit -r src/
```

## 📦 Artifacts

### Upload Test Results

```yaml
- name: Upload test report
  uses: actions/upload-artifact@v3
  with:
    name: test-report
    path: test-report.html
```

### Download Artifacts
1. Go to workflow run
2. Scroll to "Artifacts" section
3. Click to download

## 🎓 Tips

1. **Start Simple**: Begin with basic test workflow, add complexity gradually
2. **Use Matrix**: Test multiple Python versions
3. **Cache Dependencies**: Speed up workflows with caching
4. **Fail Fast**: Use `fail-fast: true` to stop on first failure
5. **Manual Triggers**: Add `workflow_dispatch` for manual runs
6. **Status Checks**: Require tests to pass before merging PRs

## 🆘 Getting Help

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [GitHub Community Forum](https://github.community/)

---

**Last Updated**: November 24, 2025  
**Maintained by**: Development Team

