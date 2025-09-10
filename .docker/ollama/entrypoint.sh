#!/bin/bash
set -e

PORT=${OLLAMA_PORT:-11434}

export OLLAMA_HOST="0.0.0.0:$PORT"
ollama serve &
SERVER_PID=$!

echo "Waiting for Ollama server to be healthy..."
until ollama ps >/dev/null 2>&1; do
    sleep 1
done

echo "Server is up. Starting model downloads..."

MODELS=${MODELS:-llama2}
for model in $(echo $MODELS | tr ',' ' '); do
    if ! ollama list | grep -q "$model"; then
        echo "Pulling model $model..."
        ollama pull "$model"
    else
        echo "Model $model already exists."
    fi
done

wait $SERVER_PID