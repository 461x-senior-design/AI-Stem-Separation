#!/bin/bash

# Ollama Setup and gpt-oss:20b Runner Script
# This script automates the setup and execution of ollama with gpt-oss:20b

set -e

echo "================================================"
echo "Ollama + gpt-oss:20b Setup Script"
echo "================================================"
echo ""

# Check if ollama is already installed
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is already installed: $(ollama --version 2>&1 | head -1)"
else
    echo "→ Downloading ollama..."
    curl -L https://github.com/ollama/ollama/releases/download/v0.1.32/ollama-linux-amd64 -o /tmp/ollama
    chmod +x /tmp/ollama
    echo "✓ Ollama downloaded to /tmp/ollama"
fi

# Set ollama binary path
OLLAMA_BIN="/tmp/ollama"
if command -v ollama &> /dev/null; then
    OLLAMA_BIN="ollama"
fi

# Check if ollama server is running
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "→ Starting ollama server..."
    $OLLAMA_BIN serve > /tmp/ollama-server.log 2>&1 &
    OLLAMA_PID=$!
    echo "✓ Ollama server started (PID: $OLLAMA_PID)"
    
    # Wait for server to be ready
    echo "→ Waiting for server to start..."
    sleep 10
    
    # Verify server is responding
    if curl -s http://localhost:11434/ > /dev/null; then
        echo "✓ Ollama server is responding"
    else
        echo "✗ Ollama server is not responding"
        exit 1
    fi
else
    echo "✓ Ollama server is already running"
fi

# Pull gpt-oss:20b model
echo "→ Pulling gpt-oss:20b model (this may take several minutes)..."
if $OLLAMA_BIN pull gpt-oss:20b; then
    echo "✓ Model gpt-oss:20b successfully pulled"
else
    echo "✗ Failed to pull gpt-oss:20b model"
    echo "  Error: Network restrictions may be blocking access to registry.ollama.ai"
    echo "  Please check network connectivity and DNS resolution"
    exit 1
fi

# List available models
echo ""
echo "Available models:"
$OLLAMA_BIN list

# Run brainstorm about PR reviews
echo ""
echo "================================================"
echo "Running PR Review Brainstorm with gpt-oss:20b"
echo "================================================"
echo ""

# Create prompt for brainstorming
PROMPT="You are an expert code reviewer. Please brainstorm and provide insights on the following aspects of pull request reviews:

1. What are the key things to look for in a code review?
2. What are common mistakes developers make in pull requests?
3. How can we improve the quality of code reviews?
4. What tools and techniques can enhance the PR review process?
5. What are best practices for writing PR descriptions?

Please provide detailed, actionable insights for each point."

# Run the model with the prompt
echo "Sending prompt to gpt-oss:20b..."
echo ""
$OLLAMA_BIN run gpt-oss:20b "$PROMPT" > /tmp/pr-review-brainstorm.txt

# Display results
echo "================================================"
echo "Brainstorm Results"
echo "================================================"
cat /tmp/pr-review-brainstorm.txt

echo ""
echo "✓ Brainstorm complete. Results saved to /tmp/pr-review-brainstorm.txt"
echo ""
echo "================================================"
echo "Setup and execution complete!"
echo "================================================"
