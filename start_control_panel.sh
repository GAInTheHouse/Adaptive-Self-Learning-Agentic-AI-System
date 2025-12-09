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

# Initialize conda
echo "🔄 Initializing conda..."
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "⚠️  Conda not found. Trying to use conda from PATH..."
fi

# Activate conda environment
echo "🔄 Activating conda environment: stt-genai..."
conda activate stt-genai 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Could not activate conda environment 'stt-genai'!"
    echo "Please create the environment first:"
    echo "  conda create -n stt-genai python=3.8"
    echo "  conda activate stt-genai"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo "✅ Conda environment 'stt-genai' activated"

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

