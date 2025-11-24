"""
pytest configuration and fixtures
"""

import pytest
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def test_audio_path():
    """Provide path to test audio file"""
    return "test_audio/test_1.wav"


@pytest.fixture(scope="session")
def test_audio_exists(test_audio_path):
    """Check if test audio exists"""
    return Path(test_audio_path).exists()


@pytest.fixture
def temp_directory():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_transcripts():
    """Provide sample transcripts for testing"""
    return {
        "perfect": "This is a perfect transcript.",
        "all_caps": "THIS IS ALL CAPS TEXT",
        "repeated_chars": "Hellllllo world",
        "empty": "",
        "no_punctuation": "this is a long text without any punctuation marks which might indicate an error",
        "normal": "This is a normal transcript with proper formatting.",
    }


@pytest.fixture
def sample_audio_transcripts():
    """Provide sample pairs of ground truth and hypotheses"""
    return [
        ("hello world", "hello world"),  # Perfect match
        ("hello world", "hello earth"),  # Single word error
        ("this is a test", "this is test"),  # Deletion
        ("hello", "hello world"),  # Insertion
        ("good morning", "good evening"),  # Substitution
    ]


def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require API server running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Mark API tests
        if "test_api" in item.nodeid:
            item.add_marker(pytest.mark.api)
        
        # Mark integration tests
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

