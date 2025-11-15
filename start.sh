#!/bin/bash
# Startup script for Ding-Bot chess engine
# Sets up environment and starts the server

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if HUGGINGFACE_MODEL_REPO is set
if [ -z "$HUGGINGFACE_MODEL_REPO" ]; then
    echo "ERROR: HUGGINGFACE_MODEL_REPO environment variable is not set!"
    echo ""
    echo "Please set it using one of these methods:"
    echo "  1. Create a .env file with: HUGGINGFACE_MODEL_REPO='your-username/chess-bot-model'"
    echo "  2. Export it: export HUGGINGFACE_MODEL_REPO='your-username/chess-bot-model'"
    echo "  3. Pass it inline: HUGGINGFACE_MODEL_REPO='your-username/chess-bot-model' ./start.sh"
    echo ""
    exit 1
fi

echo "Starting Ding-Bot chess engine..."
echo "Model repository: $HUGGINGFACE_MODEL_REPO"
echo ""

# Start the server
python3 serve.py

