"""
Unit tests for Model Validator
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.data.model_validator import (
    ModelValidator,
    ValidationConfig,
    ValidationResult
)


@pytest.fixture
def validator(temp_directory):
    """Create validator instance."""
    config = ValidationConfig(
        min_wer_improvement=0.0,
        require_significance=False  # Skip for faster tests
    )
    
    with patch('src.data.model_validator.GCSManager'):
        validator = ModelValidator(
            config=config,
            storage_dir=str(Path(temp_directory) / "validation"),
            use_gcs=False
        )
    
    return validator


@pytest.fixture
def mock_evaluation_set():
    """Create mock evaluation set."""
    return [
        {
            'audio_path': 'test1.wav',
            'reference': 'this is a test'
        },
        {
            'audio_path': 'test2.wav',
            'reference': 'hello world'
        },
        {
            'audio_path': 'test3.wav',
            'reference': 'testing one two three'
        }
    ]


def mock_baseline_transcribe(audio_path):
    """Mock baseline transcription function."""
    return "this is a test"  # Simple mock


def mock_model_transcribe(audio_path):
    """Mock fine-tuned model transcription function."""
    return "this is a test"  # Same for simplicity


class TestValidationConfig:
    """Test ValidationConfig."""
    
    def test_config_creation(self):
        """Test config creation."""
        config = ValidationConfig()
        assert config.min_wer_improvement == 0.0
        assert config.require_significance is True
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = ValidationConfig(min_wer_improvement=0.05)
        config_dict = config.to_dict()
        assert config_dict['min_wer_improvement'] == 0.05


class TestValidationResult:
    """Test ValidationResult."""
    
    def test_result_creation(self):
        """Test result creation."""
        result = ValidationResult(
            model_id="test_model",
            baseline_id="baseline",
            model_wer=0.10,
            model_cer=0.05,
            baseline_wer=0.15,
            baseline_cer=0.08,
            wer_improvement=0.05,
            cer_improvement=0.03,
            is_significant=True,
            num_samples=100,
            passed=True
        )
        
        assert result.model_id == "test_model"
        assert result.passed is True
        assert result.wer_improvement == 0.05
    
    def test_result_to_dict(self):
        """Test result serialization."""
        result = ValidationResult(
            model_id="test_model",
            baseline_id="baseline",
            model_wer=0.10,
            model_cer=0.05,
            baseline_wer=0.15,
            baseline_cer=0.08,
            wer_improvement=0.05,
            cer_improvement=0.03,
            is_significant=True,
            num_samples=100,
            passed=True
        )
        
        result_dict = result.to_dict()
        assert result_dict['model_id'] == "test_model"
        assert result_dict['passed'] is True


class TestModelValidator:
    """Test ModelValidator."""
    
    def test_initialization(self, validator):
        """Test validator initialization."""
        assert validator is not None
        assert validator.config.min_wer_improvement == 0.0
    
    def test_load_evaluation_set_jsonl(self, temp_directory):
        """Test loading evaluation set from JSONL."""
        # Create test JSONL file
        test_file = Path(temp_directory) / "test_eval.jsonl"
        with open(test_file, 'w') as f:
            f.write(json.dumps({'audio_path': 'test1.wav', 'reference': 'hello'}) + '\n')
            f.write(json.dumps({'audio_path': 'test2.wav', 'reference': 'world'}) + '\n')
        
        with patch('src.data.model_validator.GCSManager'):
            validator = ModelValidator(
                config=ValidationConfig(),
                storage_dir=str(Path(temp_directory) / "validation"),
                use_gcs=False
            )
        
        samples = validator.load_evaluation_set(str(test_file))
        
        assert len(samples) == 2
        assert samples[0]['audio_path'] == 'test1.wav'
    
    def test_load_evaluation_set_json(self, temp_directory):
        """Test loading evaluation set from JSON."""
        # Create test JSON file
        test_file = Path(temp_directory) / "test_eval.json"
        data = [
            {'audio_path': 'test1.wav', 'reference': 'hello'},
            {'audio_path': 'test2.wav', 'reference': 'world'}
        ]
        with open(test_file, 'w') as f:
            json.dump(data, f)
        
        with patch('src.data.model_validator.GCSManager'):
            validator = ModelValidator(
                config=ValidationConfig(),
                storage_dir=str(Path(temp_directory) / "validation"),
                use_gcs=False
            )
        
        samples = validator.load_evaluation_set(str(test_file))
        
        assert len(samples) == 2
    
    def test_validate_model(self, validator, mock_evaluation_set):
        """Test model validation."""
        result = validator.validate_model(
            model_id="test_model",
            model_transcribe_fn=mock_model_transcribe,
            baseline_id="baseline",
            baseline_transcribe_fn=mock_baseline_transcribe,
            evaluation_set=mock_evaluation_set
        )
        
        assert result is not None
        assert result.model_id == "test_model"
        assert result.baseline_id == "baseline"
        assert result.num_samples > 0
    
    def test_validate_model_with_improvement(self, validator, mock_evaluation_set):
        """Test validation with model improvement."""
        def better_model_transcribe(audio_path):
            # Returns exact match for first sample
            if 'test1' in audio_path:
                return "this is a test"
            return "hello world"
        
        def worse_baseline_transcribe(audio_path):
            # Returns worse transcripts
            return "this is wrong"
        
        result = validator.validate_model(
            model_id="better_model",
            model_transcribe_fn=better_model_transcribe,
            baseline_id="worse_baseline",
            baseline_transcribe_fn=worse_baseline_transcribe,
            evaluation_set=mock_evaluation_set
        )
        
        assert result.wer_improvement > 0  # Model should be better
    
    def test_validate_model_dict_return(self, validator, mock_evaluation_set):
        """Test validation with dict return from transcribe functions."""
        def dict_transcribe(audio_path):
            return {'transcript': 'this is a test'}
        
        result = validator.validate_model(
            model_id="dict_model",
            model_transcribe_fn=dict_transcribe,
            baseline_id="dict_baseline",
            baseline_transcribe_fn=dict_transcribe,
            evaluation_set=mock_evaluation_set
        )
        
        assert result is not None
    
    def test_validation_passed_criteria(self, validator):
        """Test validation pass criteria."""
        # Configure strict validator
        validator.config.min_wer_improvement = 0.05
        validator.config.max_wer_degradation_rate = 0.1
        
        eval_set = [
            {'audio_path': 'test1.wav', 'reference': 'hello world'}
        ]
        
        def good_model(audio_path):
            return "hello world"  # Perfect match
        
        def bad_baseline(audio_path):
            return "goodbye earth"  # Bad match
        
        result = validator.validate_model(
            model_id="good_model",
            model_transcribe_fn=good_model,
            baseline_id="bad_baseline",
            baseline_transcribe_fn=bad_baseline,
            evaluation_set=eval_set
        )
        
        assert result.passed is True
    
    def test_get_best_model(self, validator, mock_evaluation_set):
        """Test getting best model."""
        # Validate multiple models
        for i in range(3):
            def model_fn(audio_path):
                return "this is a test"
            
            validator.validate_model(
                model_id=f"model_{i}",
                model_transcribe_fn=model_fn,
                baseline_id="baseline",
                baseline_transcribe_fn=mock_baseline_transcribe,
                evaluation_set=mock_evaluation_set
            )
        
        best = validator.get_best_model(metric='wer', only_passed=False)
        
        assert best is not None
        assert hasattr(best, 'model_wer')
    
    def test_generate_validation_report(self, validator, mock_evaluation_set):
        """Test validation report generation."""
        # Run some validations
        validator.validate_model(
            model_id="test_model",
            model_transcribe_fn=mock_model_transcribe,
            baseline_id="baseline",
            baseline_transcribe_fn=mock_baseline_transcribe,
            evaluation_set=mock_evaluation_set
        )
        
        report = validator.generate_validation_report()
        
        assert 'generated_at' in report
        assert 'total_validations' in report
        assert report['total_validations'] > 0
    
    def test_compare_multiple_models(self, validator, mock_evaluation_set):
        """Test comparing multiple models."""
        model_configs = [
            {'id': 'model1', 'transcribe_fn': mock_model_transcribe},
            {'id': 'model2', 'transcribe_fn': mock_model_transcribe}
        ]
        baseline_config = {
            'id': 'baseline',
            'transcribe_fn': mock_baseline_transcribe
        }
        
        results = validator.compare_multiple_models(
            model_configs=model_configs,
            baseline_config=baseline_config,
            evaluation_set=mock_evaluation_set
        )
        
        assert len(results) == 2
        assert all(isinstance(r, ValidationResult) for r in results)


@pytest.mark.unit
def test_validator_integration(temp_directory):
    """Integration test for validator."""
    # Create test evaluation file
    eval_file = Path(temp_directory) / "eval_set.jsonl"
    with open(eval_file, 'w') as f:
        for i in range(10):
            f.write(json.dumps({
                'audio_path': f'test_{i}.wav',
                'reference': f'test transcript {i}'
            }) + '\n')
    
    config = ValidationConfig(
        min_wer_improvement=0.0,
        require_significance=False
    )
    
    with patch('src.data.model_validator.GCSManager'):
        validator = ModelValidator(
            config=config,
            evaluation_data_path=str(eval_file),
            storage_dir=str(Path(temp_directory) / "validation"),
            use_gcs=False
        )
    
    # Define transcription functions
    def model_transcribe(audio_path):
        idx = int(audio_path.split('_')[1].split('.')[0])
        return f'test transcript {idx}'
    
    def baseline_transcribe(audio_path):
        return 'wrong transcript'
    
    # Run validation
    result = validator.validate_model(
        model_id="integration_model",
        model_transcribe_fn=model_transcribe,
        baseline_id="integration_baseline",
        baseline_transcribe_fn=baseline_transcribe,
        evaluation_set_path=str(eval_file)
    )
    
    assert result is not None
    assert result.passed is True
    assert result.wer_improvement > 0

