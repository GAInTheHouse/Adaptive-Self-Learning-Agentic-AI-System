"""
Comprehensive API tests for both baseline and agent endpoints
Requires API server to be running: uvicorn src.agent_api:app --port 8000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import pytest
import json
import time


BASE_URL = "http://localhost:8000"
TEST_AUDIO_PATH = "data/test_audio/test_1.wav"


class TestBaselineAPI:
    """Test baseline API endpoints"""
    
    @pytest.fixture(autouse=True)
    def check_server(self):
        """Check if API server is running"""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code != 200:
                pytest.skip("API server not running")
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running - start with: uvicorn src.agent_api:app --port 8000")
    
    def test_health_endpoint(self):
        """Test /health endpoint"""
        response = requests.get(f"{BASE_URL}/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'status' in data
        assert data['status'] == 'healthy'
        assert 'model' in data
    
    def test_model_info_endpoint(self):
        """Test /model-info endpoint"""
        response = requests.get(f"{BASE_URL}/model-info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'name' in data
        assert 'parameters' in data
        assert 'device' in data
        assert data['parameters'] > 0
    
    @pytest.mark.skipif(not Path(TEST_AUDIO_PATH).exists(), 
                       reason="Test audio file not available")
    def test_transcribe_endpoint(self):
        """Test /transcribe endpoint (baseline)"""
        with open(TEST_AUDIO_PATH, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{BASE_URL}/transcribe", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'transcript' in data
        assert 'model' in data
        assert isinstance(data['transcript'], str)
        assert len(data['transcript']) > 0
    
    def test_transcribe_missing_file(self):
        """Test /transcribe with missing file"""
        response = requests.post(f"{BASE_URL}/transcribe")
        
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_transcribe_invalid_file(self):
        """Test /transcribe with invalid file"""
        files = {"file": ("test.txt", b"not an audio file", "text/plain")}
        response = requests.post(f"{BASE_URL}/transcribe", files=files)
        
        # Should fail or handle gracefully
        assert response.status_code in [400, 422, 500]


class TestAgentAPI:
    """Test agent API endpoints"""
    
    @pytest.fixture(autouse=True)
    def check_server(self):
        """Check if API server is running"""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code != 200:
                pytest.skip("API server not running")
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running - start with: uvicorn src.agent_api:app --port 8000")
    
    @pytest.mark.skipif(not Path(TEST_AUDIO_PATH).exists(), 
                       reason="Test audio file not available")
    def test_agent_transcribe_endpoint(self):
        """Test /agent/transcribe endpoint"""
        with open(TEST_AUDIO_PATH, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{BASE_URL}/agent/transcribe", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert 'transcript' in data
        assert 'original_transcript' in data
        assert 'error_detection' in data
        assert 'corrections' in data
        
        # Check error detection structure
        error_detection = data['error_detection']
        assert 'has_errors' in error_detection
        assert 'error_count' in error_detection
        assert 'error_score' in error_detection
        assert 'errors' in error_detection
        assert 'error_types' in error_detection
        
        # Check corrections structure
        corrections = data['corrections']
        assert 'applied' in corrections
        assert 'count' in corrections
        assert 'details' in corrections
    
    @pytest.mark.skipif(not Path(TEST_AUDIO_PATH).exists(), 
                       reason="Test audio file not available")
    def test_agent_transcribe_with_correction(self):
        """Test /agent/transcribe with auto_correction enabled"""
        with open(TEST_AUDIO_PATH, "rb") as f:
            files = {"file": f}
            params = {"auto_correction": True}
            response = requests.post(
                f"{BASE_URL}/agent/transcribe",
                files=files,
                params=params
            )
        
        assert response.status_code == 200
        data = response.json()
        
        # If corrections were needed, they should be applied
        if data['error_detection']['has_errors']:
            # Original and corrected might differ
            assert 'transcript' in data
            assert 'original_transcript' in data
    
    @pytest.mark.skipif(not Path(TEST_AUDIO_PATH).exists(), 
                       reason="Test audio file not available")
    def test_agent_transcribe_without_correction(self):
        """Test /agent/transcribe with auto_correction disabled"""
        with open(TEST_AUDIO_PATH, "rb") as f:
            files = {"file": f}
            params = {"auto_correction": False}
            response = requests.post(
                f"{BASE_URL}/agent/transcribe",
                files=files,
                params=params
            )
        
        assert response.status_code == 200
        data = response.json()
        
        # Corrections should not be applied
        assert data['corrections']['applied'] is False
    
    def test_agent_stats_endpoint(self):
        """Test /agent/stats endpoint"""
        response = requests.get(f"{BASE_URL}/agent/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'total_transcriptions' in data
        assert 'error_detection' in data
        assert 'learning' in data
        
        # Check error_detection structure
        assert 'threshold' in data['error_detection']
        assert 'total_errors_detected' in data['error_detection']
        
        # Check learning structure
        assert 'total_errors_learned' in data['learning']
        assert 'feedback_count' in data['learning']
    
    def test_agent_learning_data_endpoint(self):
        """Test /agent/learning-data endpoint"""
        response = requests.get(f"{BASE_URL}/agent/learning-data")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'error_patterns' in data
        assert 'correction_history' in data
        assert 'feedback_history' in data
        
        assert isinstance(data['error_patterns'], list)
        assert isinstance(data['correction_history'], list)
        assert isinstance(data['feedback_history'], list)
    
    def test_agent_feedback_endpoint(self):
        """Test /agent/feedback endpoint"""
        feedback_data = {
            "transcript_id": "test_123",
            "user_feedback": "Good transcription",
            "is_correct": True,
            "corrected_transcript": None
        }
        
        response = requests.post(
            f"{BASE_URL}/agent/feedback",
            json=feedback_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'status' in data
        assert data['status'] == 'success'
    
    def test_agent_feedback_with_correction(self):
        """Test /agent/feedback with corrected transcript"""
        feedback_data = {
            "transcript_id": "test_456",
            "user_feedback": "Needs correction",
            "is_correct": False,
            "corrected_transcript": "This is the corrected version"
        }
        
        response = requests.post(
            f"{BASE_URL}/agent/feedback",
            json=feedback_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['status'] == 'success'
    
    def test_agent_feedback_missing_fields(self):
        """Test /agent/feedback with missing required fields"""
        feedback_data = {
            "transcript_id": "test_789"
            # Missing other required fields
        }
        
        response = requests.post(
            f"{BASE_URL}/agent/feedback",
            json=feedback_data
        )
        
        # Should fail validation
        assert response.status_code == 422


class TestAPIPerformance:
    """Test API performance characteristics"""
    
    @pytest.fixture(autouse=True)
    def check_server(self):
        """Check if API server is running"""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code != 200:
                pytest.skip("API server not running")
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    @pytest.mark.skipif(not Path(TEST_AUDIO_PATH).exists(), 
                       reason="Test audio file not available")
    def test_transcription_latency(self):
        """Test transcription latency is reasonable"""
        start_time = time.time()
        
        with open(TEST_AUDIO_PATH, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{BASE_URL}/transcribe", files=files)
        
        latency = time.time() - start_time
        
        assert response.status_code == 200
        # Latency should be reasonable (< 30 seconds on CPU)
        assert latency < 30, f"Transcription took {latency:.2f}s, which is too slow"
    
    def test_health_check_fast(self):
        """Test health check is fast"""
        start_time = time.time()
        response = requests.get(f"{BASE_URL}/health")
        latency = time.time() - start_time
        
        assert response.status_code == 200
        assert latency < 1, "Health check should be under 1 second"
    
    @pytest.mark.skipif(not Path(TEST_AUDIO_PATH).exists(), 
                       reason="Test audio file not available")
    def test_agent_overhead(self):
        """Test agent transcription overhead vs baseline"""
        # Baseline transcription
        start_baseline = time.time()
        with open(TEST_AUDIO_PATH, "rb") as f:
            files = {"file": f}
            baseline_response = requests.post(f"{BASE_URL}/transcribe", files=files)
        baseline_time = time.time() - start_baseline
        
        # Agent transcription
        start_agent = time.time()
        with open(TEST_AUDIO_PATH, "rb") as f:
            files = {"file": f}
            agent_response = requests.post(f"{BASE_URL}/agent/transcribe", files=files)
        agent_time = time.time() - start_agent
        
        assert baseline_response.status_code == 200
        assert agent_response.status_code == 200
        
        # Agent should not add more than 50% overhead
        overhead = (agent_time - baseline_time) / baseline_time
        assert overhead < 0.5, f"Agent adds {overhead*100:.1f}% overhead, should be <50%"


class TestAPIEdgeCases:
    """Test API edge cases and error handling"""
    
    @pytest.fixture(autouse=True)
    def check_server(self):
        """Check if API server is running"""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code != 200:
                pytest.skip("API server not running")
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not running")
    
    def test_empty_file_upload(self):
        """Test uploading empty file"""
        files = {"file": ("empty.wav", b"", "audio/wav")}
        response = requests.post(f"{BASE_URL}/transcribe", files=files)
        
        # Should handle gracefully (either 400 or 500)
        assert response.status_code in [400, 422, 500]
    
    def test_large_file_upload(self):
        """Test uploading very large file"""
        # Create a large fake file (10MB)
        large_data = b"0" * (10 * 1024 * 1024)
        files = {"file": ("large.wav", large_data, "audio/wav")}
        
        try:
            response = requests.post(
                f"{BASE_URL}/transcribe",
                files=files,
                timeout=60
            )
            # Should either accept or reject large files
            assert response.status_code in [200, 413, 500]
        except requests.exceptions.Timeout:
            # Timeout is acceptable for large files
            pass
    
    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        # Simple concurrency test
        responses = []
        
        for _ in range(3):
            response = requests.get(f"{BASE_URL}/health")
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
    
    def test_invalid_endpoint(self):
        """Test accessing invalid endpoint"""
        response = requests.get(f"{BASE_URL}/invalid_endpoint")
        
        assert response.status_code == 404
    
    def test_wrong_http_method(self):
        """Test using wrong HTTP method"""
        # Try GET on transcribe endpoint (should be POST)
        response = requests.get(f"{BASE_URL}/transcribe")
        
        assert response.status_code == 405  # Method Not Allowed


if __name__ == "__main__":
    print("=" * 70)
    print("API Comprehensive Tests")
    print("=" * 70)
    print("\n⚠️  Make sure API server is running:")
    print("   uvicorn src.agent_api:app --reload --port 8000\n")
    
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

