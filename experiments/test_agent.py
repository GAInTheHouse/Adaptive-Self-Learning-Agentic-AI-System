"""
Agent Testing Script - Week 2
Test the agent integration system with sample audio files
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.baseline_model import BaselineSTTModel
from src.agent import STTAgent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_agent_basic():
    """Test basic agent functionality"""
    print("\n" + "="*60)
    print("Testing STT Agent - Basic Functionality")
    print("="*60)
    
    # Initialize components
    baseline_model = BaselineSTTModel(model_name="whisper")
    agent = STTAgent(baseline_model=baseline_model)
    
    # Test with sample audio
    test_audio_path = "data/test_audio/test_1.wav"
    
    if not Path(test_audio_path).exists():
        print(f"⚠️  Test audio not found: {test_audio_path}")
        print("   Skipping transcription test...")
        return
    
    print(f"\n📝 Transcribing: {test_audio_path}")
    
    # Test baseline transcription
    print("\n1. Baseline Transcription:")
    baseline_result = baseline_model.transcribe(test_audio_path)
    print(f"   Transcript: {baseline_result['transcript']}")
    print(f"   Inference time: {baseline_result.get('inference_time_seconds', 'N/A')}s")
    
    # Test agent transcription
    print("\n2. Agent Transcription (with error detection):")
    agent_result = agent.transcribe_with_agent(
        audio_path=test_audio_path,
        enable_auto_correction=True
    )
    
    print(f"   Original: {agent_result['original_transcript']}")
    print(f"   Corrected: {agent_result['transcript']}")
    print(f"   Has errors: {agent_result['error_detection']['has_errors']}")
    print(f"   Error count: {agent_result['error_detection']['error_count']}")
    print(f"   Error score: {agent_result['error_detection']['error_score']:.3f}")
    
    if agent_result['error_detection']['errors']:
        print("\n   Detected Errors:")
        for error in agent_result['error_detection']['errors']:
            print(f"     - {error['type']}: {error['description']} (confidence: {error['confidence']:.2f})")
    
    if agent_result['corrections']['applied']:
        print(f"\n   Corrections applied: {agent_result['corrections']['count']}")
        for correction in agent_result['corrections']['details']:
            print(f"     - {correction['error_type']}: {correction['original'][:50]}...")
    
    # Test agent stats
    print("\n3. Agent Statistics:")
    stats = agent.get_agent_stats()
    print(f"   Total errors detected: {stats['error_detection']['total_errors_detected']}")
    print(f"   Corrections made: {stats['learning']['corrections_made']}")
    print(f"   Feedback count: {stats['learning']['feedback_count']}")
    
    print("\n✅ Basic agent test completed!")


def test_error_detection():
    """Test error detection with various scenarios"""
    print("\n" + "="*60)
    print("Testing Error Detection")
    print("="*60)
    
    from src.agent.error_detector import ErrorDetector
    
    detector = ErrorDetector()
    
    test_cases = [
        ("", "Empty transcript"),
        ("HELLO WORLD THIS IS ALL CAPS", "All caps"),
        ("aaaaa", "Repeated characters"),
        ("This is a normal transcript.", "Normal transcript"),
        ("a b c d e f", "Too many short words"),
        ("This is a very long transcript without any punctuation which might indicate an error", "No punctuation"),
    ]
    
    print("\nTesting error detection on various transcripts:\n")
    
    for transcript, description in test_cases:
        errors = detector.detect_errors(transcript)
        error_summary = detector.get_error_summary(errors)
        
        print(f"Test: {description}")
        print(f"  Transcript: {transcript[:50]}...")
        print(f"  Errors detected: {error_summary['error_count']}")
        print(f"  Error score: {error_summary['error_score']:.3f}")
        
        if errors:
            print("  Error types:")
            for error_type, count in error_summary['error_types'].items():
                print(f"    - {error_type}: {count}")
        print()


def test_self_learning():
    """Test self-learning component (in-memory tracking)"""
    print("\n" + "="*60)
    print("Testing Self-Learning Component")
    print("="*60)
    
    from src.agent.self_learner import SelfLearner
    
    learner = SelfLearner()  # In-memory only
    
    # Record some errors
    print("\n1. Recording errors...")
    learner.record_error(
        error_type="repeated_chars",
        transcript="Helllllo world",
        context={"audio_length": 2.0},
        correction="Hello world"
    )
    
    learner.record_error(
        error_type="all_caps",
        transcript="HELLO WORLD",
        context={"audio_length": 1.5},
        correction="Hello world"
    )
    
    learner.record_error(
        error_type="empty_transcript",
        transcript="",
        context={"audio_length": 3.0}
    )
    
    print("   ✅ Recorded 3 errors")
    
    # Record feedback
    print("\n2. Recording feedback...")
    learner.record_feedback(
        transcript_id="test_001",
        user_feedback="Good transcription",
        is_correct=True
    )
    
    learner.record_feedback(
        transcript_id="test_002",
        user_feedback="Needs correction",
        is_correct=False,
        corrected_transcript="Corrected version"
    )
    
    print("   ✅ Recorded 2 feedback entries")
    
    # Get statistics
    print("\n3. Learning Statistics:")
    stats = learner.get_error_statistics()
    print(f"   Total errors: {stats['total_errors']}")
    print(f"   Error type counts: {stats['error_type_counts']}")
    print(f"   Corrections made: {stats['corrections_made']}")
    print(f"   Feedback count: {stats['feedback_count']}")
    
    # Get in-memory data (for external persistence)
    print("\n4. Getting in-memory data...")
    data = learner.get_in_memory_data()
    print(f"   ✅ Retrieved {len(data['error_patterns'])} error patterns")
    print(f"   ✅ Retrieved {len(data['correction_history'])} corrections")
    print(f"   ✅ Retrieved {len(data['feedback_history'])} feedback entries")
    print("   Note: Data persistence handled by data management layer")
    
    print("\n✅ Self-learning test completed!")


def main():
    """Run all agent tests"""
    print("\n" + "="*60)
    print("STT Agent Integration Tests - Week 2")
    print("="*60)
    
    try:
        # Test error detection
        test_error_detection()
        
        # Test self-learning
        test_self_learning()
        
        # Test basic agent functionality
        test_agent_basic()
        
        print("\n" + "="*60)
        print("✅ All agent tests completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

