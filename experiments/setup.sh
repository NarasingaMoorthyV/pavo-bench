#!/bin/bash
# PAVO Experiment Setup for Lambda Labs H100
# Run this first: bash setup.sh

set -e
echo "=== PAVO Experiment Setup ==="

# Install Python dependencies
pip install torch numpy scipy scikit-learn huggingface_hub datasets tqdm

# Install ollama
if ! command -v ollama &> /dev/null; then
    echo "Installing ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
    sleep 5
fi

# Start ollama server
echo "Starting ollama server..."
ollama serve &
sleep 10

# Pull models
echo "Pulling Llama 3.1 8B..."
ollama pull llama3.1:8b
echo "Pulling Gemma2 2B..."
ollama pull gemma2:2b

# Install whisper
pip install openai-whisper faster-whisper

echo "=== Setup complete ==="
echo "Now run: python run_all_experiments.py"
