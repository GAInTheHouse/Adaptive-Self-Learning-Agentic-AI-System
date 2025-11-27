#!/usr/bin/env python
"""
Comprehensive Demo Script
Demonstrates all major functionalities of the Adaptive Self-Learning STT System
"""

import sys
from pathlib import Path
import json
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def demo_baseline_model():
    """Demo 1: Baseline STT Model"""
    print_section("DEMO 1: Baseline STT Model")
    
    from src.baseline_model import BaselineSTTModel
    
    print("Initializing Whisper baseline model...")
    model = BaselineSTTModel(model_name="whisper")
    
    # Get model info
    info = model.get_model_info()
    print(f"✅ Model: {info['name']}")
    print(f"✅ Parameters: {info['parameters']:,}")
    print(f"✅ Device: {info['device']}")
    print(f"✅ Trainable params: {info['trainable_params']:,}")
    
    # Test transcription
    test_audio = "data/test_audio/test_1.wav"
    if Path(test_audio).exists():
        print(f"\n🎤 Transcribing: {test_audio}")
        result = model.transcribe(test_audio)
        print(f"📝 Transcript: {result['transcript']}")
    else:
        print(f"⚠️  Test audio not found: {test_audio}")
    
    return model

def demo_agent_system(baseline_model):
    """Demo 2: Agent with Error Detection"""
    print_section("DEMO 2: Agent System with Error Detection")
    
    from src.agent import STTAgent
    
    print("Initializing STT Agent...")
    agent = STTAgent(
        baseline_model=baseline_model,
        error_threshold=0.3
    )
    
    # Create test case with known errors
    test_audio = "data/test_audio/test_1.wav"
    
    if Path(test_audio).exists():
        print(f"\n🎤 Transcribing with agent: {test_audio}")
        
        # Transcribe without auto-correction
        print("\n--- Without Auto-Correction ---")
        result_no_correction = agent.transcribe_with_agent(
            test_audio,
            enable_auto_correction=False
        )
        
        print(f"📝 Original transcript: {result_no_correction['transcript']}")
        print(f"🔍 Errors detected: {result_no_correction['error_detection']['error_count']}")
        print(f"📊 Error score: {result_no_correction['error_detection']['error_score']:.2f}")
        print(f"🏷️  Error types: {list(result_no_correction['error_detection']['error_types'].keys())}")
        
        # Transcribe with auto-correction
        print("\n--- With Auto-Correction ---")
        result_corrected = agent.transcribe_with_agent(
            test_audio,
            enable_auto_correction=True
        )
        
        print(f"📝 Corrected transcript: {result_corrected['transcript']}")
        print(f"✅ Corrections applied: {result_corrected['corrections']['count']}")
        
        # Show agent statistics
        print("\n--- Agent Statistics ---")
        stats = agent.get_agent_stats()
        print(f"Total transcriptions: {stats['total_transcriptions']}")
        print(f"Error detection rate: {stats['error_detection']['error_rate']:.1%}")
        print(f"Total errors learned: {stats['learning']['total_errors_learned']}")
        
        return result_corrected
    else:
        print(f"⚠️  Test audio not found: {test_audio}")
        return None

def demo_data_management(agent_result):
    """Demo 3: Data Management System"""
    print_section("DEMO 3: Data Management System")
    
    from src.data.integration import IntegratedDataManagementSystem
    
    print("Initializing data management system...")
    system = IntegratedDataManagementSystem(
        base_dir="data/demo",
        use_gcs=False  # Set to True if GCP is configured
    )
    
    # Record a failed case
    if agent_result and agent_result['error_detection']['has_errors']:
        print("\n📝 Recording failed transcription...")
        case_id = system.record_failed_transcription(
            audio_path="data/test_audio/test_1.wav",
            original_transcript=agent_result['original_transcript'],
            corrected_transcript=None,  # Will add later
            error_types=list(agent_result['error_detection']['error_types'].keys()),
            error_score=agent_result['error_detection']['error_score'],
            inference_time=agent_result.get('inference_time_seconds', 0),
            model_confidence=0.85
        )
        print(f"✅ Recorded case: {case_id}")
        
        # Add correction
        print("\n✏️  Adding correction...")
        corrected_text = "This is a properly formatted transcript."
        system.add_correction(case_id, corrected_text)
        print(f"✅ Correction added: {corrected_text}")
    else:
        print("⚠️  No errors to record (creating sample data)...")
        case_id = system.record_failed_transcription(
            audio_path="test_audio/sample.wav",
            original_transcript="THIS IS ALL CAPS NO PUNCTUATION",
            corrected_transcript="This is all caps, no punctuation.",
            error_types=["all_caps", "missing_punctuation"],
            error_score=0.75,
            inference_time=0.5,
            model_confidence=0.85
        )
        print(f"✅ Created sample case: {case_id}")
    
    # Get statistics
    print("\n--- System Statistics ---")
    stats = system.get_system_statistics()
    print(f"Total failed cases: {stats['data_management']['total_failed_cases']}")
    print(f"Corrected cases: {stats['data_management']['corrected_cases']}")
    print(f"Correction rate: {stats['data_management']['correction_rate']:.1%}")
    print(f"Error type distribution: {stats['data_management']['error_type_distribution']}")
    
    # Track performance
    print("\n--- Recording Performance Metrics ---")
    system.record_training_performance(
        model_version="whisper_base_v1",
        wer=0.12,
        cer=0.06,
        training_metadata={
            "epochs": 10,
            "batch_size": 16,
            "learning_rate": 0.0001
        }
    )
    print("✅ Performance metrics recorded")
    
    # Record inference stats
    system.record_inference_stats(
        model_version="whisper_base_v1",
        latency=0.5,
        throughput=2.0,
        batch_size=1
    )
    print("✅ Inference stats recorded")
    
    # Generate report
    print("\n--- Generating Comprehensive Report ---")
    report_path = "data/demo/demo_report.json"
    report = system.generate_comprehensive_report(output_path=report_path)
    print(f"✅ Report generated: {report_path}")
    print(f"   Report timestamp: {report['report_metadata']['timestamp']}")
    
    return system

def demo_fine_tuning_pipeline(data_system):
    """Demo 4: Fine-tuning Dataset Preparation"""
    print_section("DEMO 4: Fine-tuning Dataset Preparation")
    
    # Check if we have enough data
    stats = data_system.get_system_statistics()
    corrected_cases = stats['data_management']['corrected_cases']
    
    print(f"Corrected cases available: {corrected_cases}")
    
    if corrected_cases >= 10:  # Lowered threshold for demo
        print("\n📦 Preparing fine-tuning dataset...")
        
        try:
            dataset_info = data_system.prepare_finetuning_dataset(
                min_error_score=0.3,  # Lower threshold for demo
                max_samples=1000,
                balance_error_types=True,
                create_version=True,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15
            )
            
            print(f"✅ Dataset created: {dataset_info['local_path']}")
            print(f"\n--- Dataset Statistics ---")
            print(f"Total samples: {dataset_info['stats']['total_samples']}")
            print(f"Train samples: {dataset_info['stats']['train_size']}")
            print(f"Val samples: {dataset_info['stats']['val_size']}")
            print(f"Test samples: {dataset_info['stats']['test_size']}")
            print(f"Error type distribution: {dataset_info['stats']['error_type_distribution']}")
            
            if dataset_info.get('version_id'):
                print(f"\n✅ Dataset version: {dataset_info['version_id']}")
                print(f"   Quality score: {dataset_info.get('quality_score', 'N/A')}")
        
        except Exception as e:
            print(f"⚠️  Could not prepare dataset: {e}")
    else:
        print(f"⚠️  Need at least 10 corrected cases (have {corrected_cases})")
        print("   In production, you'd need 500+ cases for good fine-tuning")

def demo_evaluation_framework():
    """Demo 5: Evaluation Framework"""
    print_section("DEMO 5: Evaluation Framework")
    
    print("The evaluation framework can be run with:")
    print("  python experiments/kavya_evaluation_framework.py")
    print("\nIt provides:")
    print("  ✅ WER/CER calculation")
    print("  ✅ Error analysis")
    print("  ✅ Performance benchmarking")
    print("  ✅ Visualization generation")
    print("\nOutput files:")
    print("  - experiments/evaluation_outputs/evaluation_report.json")
    print("  - experiments/evaluation_outputs/evaluation_summary.json")
    print("  - experiments/evaluation_outputs/benchmark_report.json")
    print("  - experiments/evaluation_outputs/visualizations/*.png")

def demo_api_server():
    """Demo 6: API Server"""
    print_section("DEMO 6: API Server")
    
    print("To start the Agent API server:")
    print("  uvicorn src.agent_api:app --reload --port 8000")
    print("\nAvailable endpoints:")
    print("  POST   /agent/transcribe        - Transcribe with error detection")
    print("  POST   /agent/feedback          - Submit feedback")
    print("  GET    /agent/stats             - Get agent statistics")
    print("  GET    /agent/learning-data     - Get learning data")
    print("  POST   /transcribe              - Simple transcription (baseline)")
    print("  GET    /model-info              - Model information")
    print("  GET    /health                  - Health check")
    print("\nExample API calls:")
    print('  curl -X POST "http://localhost:8000/agent/transcribe" \\')
    print('    -F "file=@data/test_audio/test_1.wav"')
    print('\n  curl "http://localhost:8000/agent/stats"')

def demo_gcp_integration():
    """Demo 7: GCP Integration"""
    print_section("DEMO 7: Google Cloud Platform Integration")
    
    print("GCP Setup Commands:")
    print("  gcloud auth login")
    print("  gcloud config set project stt-agentic-ai-2025")
    print("  gcloud services enable compute.googleapis.com storage-api.googleapis.com")
    print("  gsutil mb gs://stt-project-datasets")
    print("\nGCP Scripts:")
    print("  bash scripts/setup_gcp_gpu.sh      - Create GPU VM")
    print("  python scripts/deploy_to_gcp.py    - Deploy to GCP")
    print("  python scripts/monitor_gcp_costs.py - Monitor costs")
    print("\nWith GCP enabled, data is automatically synced to cloud storage")
    print("for backup, collaboration, and disaster recovery.")

def demo_production_workflow():
    """Demo 8: Production Workflow"""
    print_section("DEMO 8: Production Workflow Example")
    
    print("A typical production workflow:")
    print("\n1️⃣  Initialize System")
    print("   from src.baseline_model import BaselineSTTModel")
    print("   from src.agent import STTAgent")
    print("   from src.data.integration import IntegratedDataManagementSystem")
    print("   model = BaselineSTTModel()")
    print("   agent = STTAgent(model)")
    print("   data_system = IntegratedDataManagementSystem()")
    
    print("\n2️⃣  Process Audio")
    print("   result = agent.transcribe_with_agent('audio.wav')")
    
    print("\n3️⃣  Record Failures")
    print("   if result['error_detection']['has_errors']:")
    print("       case_id = data_system.record_failed_transcription(...)")
    
    print("\n4️⃣  Collect Corrections")
    print("   data_system.add_correction(case_id, corrected_text)")
    
    print("\n5️⃣  Monitor Progress")
    print("   stats = data_system.get_system_statistics()")
    
    print("\n6️⃣  Prepare Fine-tuning Dataset")
    print("   if stats['corrected_cases'] >= 500:")
    print("       dataset = data_system.prepare_finetuning_dataset()")
    
    print("\n7️⃣  Track Performance")
    print("   data_system.record_training_performance()")
    
    print("\n8️⃣  Generate Reports")
    print("   report = data_system.generate_comprehensive_report()")

def main():
    """Run all demos"""
    print("\n" + "🎯" * 35)
    print("  COMPREHENSIVE DEMO - Adaptive Self-Learning STT System")
    print("🎯" * 35)
    
    print("\nThis demo will showcase all major functionalities:")
    print("  1. Baseline STT Model")
    print("  2. Agent with Error Detection")
    print("  3. Data Management System")
    print("  4. Fine-tuning Pipeline")
    print("  5. Evaluation Framework")
    print("  6. API Server")
    print("  7. GCP Integration")
    print("  8. Production Workflow")
    
    try:
        # Demo 1: Baseline Model
        baseline_model = demo_baseline_model()
        time.sleep(1)
        
        # Demo 2: Agent System
        agent_result = demo_agent_system(baseline_model)
        time.sleep(1)
        
        # Demo 3: Data Management
        data_system = demo_data_management(agent_result)
        time.sleep(1)
        
        # Demo 4: Fine-tuning Pipeline
        demo_fine_tuning_pipeline(data_system)
        time.sleep(1)
        
        # Demo 5: Evaluation Framework
        demo_evaluation_framework()
        time.sleep(1)
        
        # Demo 6: API Server
        demo_api_server()
        time.sleep(1)
        
        # Demo 7: GCP Integration
        demo_gcp_integration()
        time.sleep(1)
        
        # Demo 8: Production Workflow
        demo_production_workflow()
        
        # Final summary
        print_section("DEMO COMPLETE ✅")
        print("All major functionalities have been demonstrated!")
        print("\nNext steps:")
        print("  1. Run individual test scripts: python experiments/test_*.py")
        print("  2. Start API server: uvicorn src.agent_api:app --reload")
        print("  3. Process your own audio files")
        print("  4. Check documentation: README.md, SETUP_INSTRUCTIONS.md")
        print("\n📚 Documentation locations:")
        print("  - README.md                    - Full documentation")
        print("  - SETUP_INSTRUCTIONS.md        - Detailed setup guide")
        print("  - QUICK_REFERENCE.md           - Quick command reference")
        print("  - DATA_MANAGEMENT_README.md    - Data management guide")
        print("  - docs/                        - Component-specific docs")
        print("\n🧪 Test scripts:")
        print("  - experiments/test_baseline.py")
        print("  - experiments/test_agent.py")
        print("  - experiments/test_data_management.py")
        print("  - experiments/test_api.py")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Tip: Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()

