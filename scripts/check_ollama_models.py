#!/usr/bin/env python3
"""
Quick script to check what Ollama models are available.
"""

try:
    import ollama
    
    print("Checking Ollama models...")
    print("=" * 60)
    
    models_response = ollama.list()
    print(f"Response type: {type(models_response)}")
    print(f"Response: {models_response}")
    print("=" * 60)
    
    # Handle ListResponse object
    if hasattr(models_response, 'models'):
        models_list = models_response.models
    elif isinstance(models_response, dict):
        models_list = models_response.get('models', [])
    elif isinstance(models_response, list):
        models_list = models_response
    else:
        models_list = []
    
    if not models_list:
        print("\n❌ No models found. You need to pull a model first:")
        print("   ollama pull llama3.2:3b")
    else:
        print(f"\n✅ Found {len(models_list)} model(s):")
        for i, m in enumerate(models_list, 1):
            # Handle Model objects
            if hasattr(m, 'model'):
                name = m.model
                size = getattr(m, 'size', 'unknown')
                modified = getattr(m, 'modified_at', 'unknown')
                print(f"  {i}. {name}")
                print(f"     Size: {size}")
                print(f"     Modified: {modified}")
            elif isinstance(m, dict):
                name = m.get('model') or m.get('name', 'unknown')
                size = m.get('size', 'unknown')
                modified = m.get('modified_at', 'unknown')
                print(f"  {i}. {name}")
                print(f"     Size: {size}")
                print(f"     Modified: {modified}")
            else:
                print(f"  {i}. {m}")
        
        print("\n📝 To use in the system, you can use:")
        first_model = models_list[0]
        if hasattr(first_model, 'model'):
            model_name = first_model.model
        elif isinstance(first_model, dict):
            model_name = first_model.get('model') or first_model.get('name', 'unknown')
        else:
            model_name = str(first_model)
        
        print(f"   - Exact name: {model_name}")
        if ':' in model_name:
            base_name = model_name.split(':')[0]
            print(f"   - Base name: {base_name}")
            
except ImportError:
    print("❌ Ollama package not installed.")
    print("   Install with: pip install ollama")
except Exception as e:
    print(f"❌ Error: {e}")
    print("   Make sure Ollama is installed and running:")
    print("   1. Install: https://ollama.ai/download")
    print("   2. Start server: ollama serve")
    print("   3. Pull model: ollama pull llama3.2:3b")

