#!/bin/bash

# Sonar Real-time Translator Setup Script
# This script sets up the environment for the real-time video and audio translator

set -e  # Exit on any error

echo "🎯 Setting up Sonar Real-time Translator..."
echo "==========================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d" " -f2 | cut -d"." -f1,2)
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python 3.8 or later is required. Current version: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python version: $PYTHON_VERSION"

# Check if ffmpeg is installed (required for audio processing)
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  FFmpeg not found. Installing..."
    
    # Check the operating system
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ffmpeg
        else
            echo "❌ Homebrew not found. Please install FFmpeg manually."
            echo "   Visit: https://ffmpeg.org/download.html"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y ffmpeg
        elif command -v yum &> /dev/null; then
            sudo yum install -y ffmpeg
        elif command -v pacman &> /dev/null; then
            sudo pacman -S ffmpeg
        else
            echo "❌ Package manager not found. Please install FFmpeg manually."
            exit 1
        fi
    else
        echo "❌ Unsupported operating system: $OSTYPE"
        echo "   Please install FFmpeg manually: https://ffmpeg.org/download.html"
        exit 1
    fi
else
    echo "✅ FFmpeg is installed"
fi

# Check if PortAudio is installed (required for PyAudio)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if ! brew list portaudio &> /dev/null; then
        echo "⚠️  PortAudio not found. Installing..."
        brew install portaudio
    else
        echo "✅ PortAudio is installed"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if ! dpkg -l | grep -q portaudio19-dev; then
        echo "⚠️  PortAudio development files not found. Installing..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y portaudio19-dev python3-pyaudio
        elif command -v yum &> /dev/null; then
            sudo yum install -y portaudio-devel
        elif command -v pacman &> /dev/null; then
            sudo pacman -S portaudio
        fi
    else
        echo "✅ PortAudio development files are installed"
    fi
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (CPU version)
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install fairseq2 for Sonar
echo "🎯 Installing fairseq2 (Sonar)..."
pip install fairseq2

# Install other dependencies
echo "📚 Installing other dependencies..."
pip install -r requirements.txt

# Verify installations
echo "🔍 Verifying installations..."

# Test PyTorch
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} installed successfully')"

# Test fairseq2
python3 -c "import fairseq2; print('✅ fairseq2 installed successfully')"

# Test other key packages
python3 -c "import cv2; print(f'✅ OpenCV {cv2.__version__} installed successfully')"
python3 -c "import pyaudio; print('✅ PyAudio installed successfully')"
python3 -c "import streamlit; print(f'✅ Streamlit {streamlit.__version__} installed successfully')"

echo ""
echo "🎉 Setup completed successfully!"
echo "==========================================="
echo ""
echo "📋 Next steps:"
echo "1. Get a Gemini API key from Google AI Studio"
echo "2. Run the Streamlit app: streamlit run streamlit_app.py"
echo "3. Or run the command-line version: python realtime_video_translator.py"
echo ""
echo "💡 Tips:"
echo "- Make sure your microphone and camera permissions are enabled"
echo "- For better performance, consider using a GPU if available"
echo "- The first run will download Sonar models (may take a few minutes)"
echo ""
echo "🚀 Happy translating!"

# Check if we should download models now
read -p "📥 Do you want to download Sonar models now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 Downloading Sonar models..."
    python3 -c "
import asyncio
from sonar_translator import SonarRealtimeTranslator, TranslationConfig

async def download_models():
    config = TranslationConfig(device='cpu')
    translator = SonarRealtimeTranslator(config)
    await translator.initialize_models()
    print('✅ Models downloaded successfully!')

asyncio.run(download_models())
    "
fi

echo "✨ All done! Enjoy your real-time translator!"

