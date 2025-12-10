#!/bin/bash

# Start Control Panel Script for STT System
# This script starts the control panel API server

echo "=================================================="
echo "  STT System Control Panel Startup Script"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "src/control_panel_api.py" ]; then
    echo "❌ Error: control_panel_api.py not found!"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Check if we're already in a virtual environment
if [ -n "$VIRTUAL_ENV" ] || [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "✅ Already in a virtual environment"
    if [ -n "$VIRTUAL_ENV" ]; then
        echo "   Using venv: $VIRTUAL_ENV"
    elif [ -n "$CONDA_DEFAULT_ENV" ]; then
        echo "   Using conda: $CONDA_DEFAULT_ENV"
    fi
    ENV_ACTIVATED=true
else
    ENV_ACTIVATED=false
    
    # Try to activate venv first (preferred for this project)
    if [ -d "venv" ]; then
        echo "🔄 Activating venv..."
        source venv/bin/activate
        if [ $? -eq 0 ]; then
            echo "✅ Venv activated"
            ENV_ACTIVATED=true
        fi
    elif [ -d ".venv" ]; then
        echo "🔄 Activating .venv..."
        source .venv/bin/activate
        if [ $? -eq 0 ]; then
            echo "✅ .venv activated"
            ENV_ACTIVATED=true
        fi
    fi
    
    # If venv not found, try conda as fallback
    if [ "$ENV_ACTIVATED" = false ]; then
        echo "🔄 Venv not found, trying conda..."
        
        # Try to find conda.sh in common locations
        CONDA_INIT_FOUND=false
        
        # Check common conda installation paths
        for CONDA_PATH in \
            "/opt/homebrew/anaconda3/etc/profile.d/conda.sh" \
            "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" \
            "$HOME/anaconda3/etc/profile.d/conda.sh" \
            "$HOME/miniconda3/etc/profile.d/conda.sh" \
            "/opt/conda/etc/profile.d/conda.sh" \
            "/usr/local/anaconda3/etc/profile.d/conda.sh" \
            "/usr/local/miniconda3/etc/profile.d/conda.sh"
        do
            if [ -f "$CONDA_PATH" ]; then
                source "$CONDA_PATH"
                CONDA_INIT_FOUND=true
                echo "✅ Found conda at: $CONDA_PATH"
                break
            fi
        done
        
        # If not found in standard locations, try to use conda from PATH
        if [ "$CONDA_INIT_FOUND" = false ]; then
            if command -v conda &> /dev/null; then
                echo "✅ Using conda from PATH"
                # Initialize conda for this shell
                eval "$(conda shell.bash hook)"
                CONDA_INIT_FOUND=true
            fi
        fi
        
        # Activate conda environment if conda is available
        if [ "$CONDA_INIT_FOUND" = true ]; then
            echo "🔄 Activating conda environment: stt-genai..."
            conda activate stt-genai 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "✅ Conda environment 'stt-genai' activated"
                ENV_ACTIVATED=true
            else
                echo "⚠️  Conda environment 'stt-genai' not found, but continuing..."
            fi
        fi
    fi
    
    # If still no environment activated, warn but continue
    if [ "$ENV_ACTIVATED" = false ]; then
        echo "⚠️  Warning: No virtual environment activated!"
        echo "   It's recommended to use a virtual environment."
        echo "   You can create one with: python -m venv venv"
    fi
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python -c "import fastapi, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing required packages!"
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if port 8000 is available
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port 8000 is already in use!"
    echo "Do you want to kill the existing process? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "🔪 Killing process on port 8000..."
        kill -9 $(lsof -ti:8000)
        sleep 2
    else
        echo "Please free port 8000 and try again."
        exit 1
    fi
fi

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/production/{failed_cases,metadata,finetuning,versions,reports}
mkdir -p frontend

echo ""
echo "=================================================="
echo "  🚀 Starting Control Panel API Server"
echo "=================================================="
echo ""
echo "  📡 API Server: http://localhost:8000"
echo "  📊 Control Panel: http://localhost:8000/app"
echo "  📚 API Docs: http://localhost:8000/docs"
echo ""
echo "  Press Ctrl+C to stop the server"
echo "=================================================="
echo ""

# Start the server
uvicorn src.control_panel_api:app --reload --port 8000 --host 0.0.0.0

