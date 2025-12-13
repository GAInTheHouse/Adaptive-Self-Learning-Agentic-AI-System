#!/usr/bin/env python3
"""
Test script to verify LLM (Ollama with Llama 2/3) connection and functionality.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.llm_corrector import LlamaLLMCorrector
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_llm_connection():
    """Test LLM connection and basic functionality"""
    
    print("=" * 60)
    print("LLM Connection Test (Ollama with Llama 2/3)")
    print("=" * 60)
    
    # Initialize LLM
    print("\n1. Initializing Ollama LLM...")
    try:
        llm = LlamaLLMCorrector(
            model_name="llama3.2:3b",  # Use Ollama Llama 3.2 3B
            use_quantization=False,  # Not used for Ollama
            fast_mode=True
        )
        print("   ✓ LLM corrector initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize LLM: {e}")
        print(f"   Make sure Ollama is installed and running:")
        print(f"   1. Install Ollama: https://ollama.ai/download")
        print(f"   2. Pull the model: ollama pull llama3.2:3b")
        print(f"   3. Ensure Ollama server is running: ollama serve")
        return False
    
    # Check availability
    print("\n2. Checking LLM availability...")
    is_available = llm.is_available()
    if is_available:
        print("   ✓ LLM is available and loaded")
    else:
        print("   ✗ LLM is not available")
        return False
    
    # Test correction
    print("\n3. Testing transcript correction...")
    test_transcript = "HIS LATRPAR AS USUALLY FORE"
    print(f"   Input: {test_transcript}")
    
    correction_success = False
    correction_times = []
    try:
        start_time = time.time()
        result = llm.correct_transcript(
            test_transcript,
            errors=[{"type": "garbled", "description": "nonsense words"}],
            context={}  # General conversational transcripts
        )
        inference_time = time.time() - start_time
        correction_times.append(inference_time)
        
        corrected = result.get("corrected_transcript", "")
        print(f"   Output: {corrected}")
        print(f"   Inference time: {inference_time:.2f}s")
        
        # Check if LLM was actually used and produced a correction
        if not result.get("llm_used", False):
            print("   ✗ LLM was not used (check for errors)")
            correction_success = False
        elif not corrected or corrected == test_transcript:
            print("   ⚠ LLM returned same or empty transcript")
            correction_success = False
        else:
            print("   ✓ LLM successfully corrected the transcript")
            correction_success = True
    
    except Exception as e:
        print(f"   ✗ Error during correction: {e}")
        import traceback
        traceback.print_exc()
        correction_success = False
        return False
    
    # Test improvement (tests general quality improvement for conversational text)
    print("\n4. Testing transcript improvement...")
    print("   (This tests if LLM can improve readability, fix punctuation, and capitalization)")
    test_transcript2 = "i wrote a book it was really good"
    print(f"   Input: {test_transcript2}")
    
    improvement_success = False
    try:
        start_time = time.time()
        improved = llm.improve_transcript(test_transcript2, improvement_type="general")
        inference_time = time.time() - start_time
        correction_times.append(inference_time)
        print(f"   Output: {improved}")
        print(f"   Inference time: {inference_time:.2f}s")
        
        # Check if improvement was made (capitalization, punctuation, etc.)
        if improved and improved != test_transcript2:
            # Check if it actually improved (capitalization, punctuation added)
            has_improvement = (
                improved[0].isupper() != test_transcript2[0].isupper() or  # Capitalization changed
                '.' in improved or ',' in improved or '!' in improved or '?' in improved  # Punctuation added
            )
            if has_improvement:
                print("   ✓ LLM successfully improved the transcript (capitalization/punctuation)")
                improvement_success = True
            else:
                print("   ⚠ LLM changed text but no clear improvement detected")
                improvement_success = False
        else:
            print("   ⚠ LLM returned same transcript (may be acceptable for already-correct text)")
            improvement_success = False
    
    except Exception as e:
        print(f"   ✗ Error during improvement: {e}")
        import traceback
        traceback.print_exc()
        improvement_success = False
        return False
    
    # Final summary
    print("\n" + "=" * 60)
    print("Summary:")
    if correction_times:
        avg_time = sum(correction_times) / len(correction_times)
        print(f"  Average inference time: {avg_time:.2f}s")
        print(f"  Min inference time: {min(correction_times):.2f}s")
        print(f"  Max inference time: {max(correction_times):.2f}s")
    
    if correction_success and improvement_success:
        print("✓ All tests passed! LLM is working correctly.")
        print("=" * 60)
        return True
    else:
        print("✗ Some tests failed or did not produce expected results.")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = test_llm_connection()
    sys.exit(0 if success else 1)

