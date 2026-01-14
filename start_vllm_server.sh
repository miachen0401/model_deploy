#!/bin/bash
# Start vLLM OpenAI-compatible API server
# This is the recommended production deployment method
# Optimized for RTX 4080 (16GB VRAM)

# Configuration for RTX 4080
MODEL_PATH="model"
PORT=8000
GPU_MEMORY_UTIL=0.90          # Use 90% of 16GB = ~14.4GB
MAX_MODEL_LEN=4096            # Maximum context length
MAX_NUM_SEQS=128              # Max concurrent sequences (optimized for 4080)
DTYPE="bfloat16"              # Best precision for Ampere+ GPUs

echo "=========================================="
echo "Starting vLLM OpenAI-Compatible Server"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "GPU: RTX 4080 (16GB)"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTIL} (~14.4GB)"
echo "Max Context Length: $MAX_MODEL_LEN tokens"
echo "Max Concurrent Sequences: $MAX_NUM_SEQS"
echo "Data Type: $DTYPE"
echo "=========================================="

# Start vLLM server with optimized settings for RTX 4080
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tokenizer "$MODEL_PATH" \
    --trust-remote-code \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
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
