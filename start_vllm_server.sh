#!/bin/bash
# Start vLLM OpenAI-compatible API server
# This is the recommended production deployment method

# Load configuration
MODEL_PATH="model"
PORT=8000
GPU_MEMORY_UTIL=0.90
MAX_MODEL_LEN=4096

echo "=========================================="
echo "Starting vLLM OpenAI-Compatible Server"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "GPU Memory: ${GPU_MEMORY_UTIL}%"
echo "Max Context: $MAX_MODEL_LEN tokens"
echo "=========================================="

# Start vLLM server with optimized settings for RTX 4080
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tokenizer "$MODEL_PATH" \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --enable-prefix-caching \
    --disable-log-requests

echo ""
echo "Server started on http://localhost:$PORT"
echo "OpenAI API endpoint: http://localhost:$PORT/v1"
echo ""
echo "Test with:"
echo "  curl http://localhost:$PORT/v1/models"
