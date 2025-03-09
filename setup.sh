#!/bin/bash

# Detect OS
OS="$(uname -s)"

echo "🚀 Installing Ollama..."

if [[ "$OS" == "Darwin" ]]; then
    # MacOS installation
    brew install ollama
elif [[ "$OS" == "Linux" ]]; then
    # Linux installation
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "❌ Unsupported OS: $OS"
    exit 1
fi

echo "📥 Pulling Llama 3.2 model..."
ollama pull llama3.2

echo "🐳 Starting Qdrant in Docker..."
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant

echo "✅ Setup Complete! Ollama, Qdrant, and LlamaIndex are installed and running."
