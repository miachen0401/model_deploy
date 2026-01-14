# vLLM Deployment Guide

## Overview

This project now supports **vLLM** for high-performance inference of your fine-tuned Qwen 2.5B model. vLLM provides:

- **10-20x faster inference** compared to standard HuggingFace Transformers
- **PagedAttention** for efficient memory management
- **Continuous batching** for higher throughput
- **Streaming support** (via OpenAI-compatible server)
- **Optimized for RTX 4080** with bfloat16 precision

## Prerequisites

- RTX 4080 GPU (or any NVIDIA GPU with Compute Capability 8.0+)
- WSL Ubuntu (or native Linux)
- CUDA 12.1+ installed
- Python 3.9+

## Installation

### 1. Install vLLM

```bash
# Sync and install dependencies including vLLM
uv sync
```

This will install vLLM v0.11.0 and all required dependencies.

### 2. Verify Installation

```bash
# Check if vLLM is installed
uv run python -c "import vllm; print(vllm.__version__)"
```

## Deployment Options

You have **two deployment options**:

### Option 1: vLLM OpenAI-Compatible Server (Recommended)

**Best for production** - Full OpenAI API compatibility with streaming support.

### Option 2: Custom FastAPI Server

**Best for custom endpoints** - Full control over API design, but no streaming.

---

## Option 1: vLLM OpenAI-Compatible Server (RECOMMENDED)

This is the **recommended production deployment** method.

### Start the Server

```bash
# Start vLLM's built-in OpenAI-compatible server
bash start_vllm_server.sh
```

The server will:
1. Initialize vLLM engine with optimal settings for RTX 4080
2. Load your fine-tuned model from the `model/` directory
3. Start listening on `http://localhost:8000/v1`

### Test the Server

```bash
# Run OpenAI-compatible examples
uv run python example_openai_compatible.py
```

### Features

- ✅ **Full OpenAI API compatibility** - Drop-in replacement for OpenAI API
- ✅ **Streaming support** - Real-time token generation
- ✅ **Batch processing** - Handle multiple requests efficiently
- ✅ **Production-ready** - Built-in monitoring and metrics
- ✅ **Best performance** - Optimized vLLM implementation

### API Endpoints

All OpenAI endpoints are available at `http://localhost:8000/v1`:

```bash
# List models
curl http://localhost:8000/v1/models

# Text completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "prompt": "What is machine learning?",
    "max_tokens": 100
  }'

# Streaming completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "prompt": "Explain quantum computing:",
    "max_tokens": 200,
    "stream": true
  }'
```

---

## Option 2: Custom FastAPI Server

Use this if you need custom API endpoints or want full control over the server.

### Start the Server

```bash
# Start custom FastAPI server with vLLM
uv run python app_vllm.py
```

The server will:
1. Initialize vLLM LLM engine
2. Load your fine-tuned model with optimized settings
3. Start listening on `http://localhost:8000`

### Test the Server

```bash
# Test custom FastAPI endpoints
uv run python example_usage_vllm.py
```

### Features

- ✅ **Custom API design** - Define your own endpoints and response formats
- ✅ **High performance** - vLLM-powered inference
- ✅ **Parallel sequences** - Generate multiple completions simultaneously
- ⚠️  **No streaming** - Streaming requires OpenAI-compatible server

### API Endpoints

Custom endpoints at `http://localhost:8000`:

```bash
# Health check
curl http://localhost:8000/health

# Text generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning?",
    "max_new_tokens": 100,
    "temperature": 0.7
  }'
```

---

## Monitor GPU Usage

For both deployment options, monitor your RTX 4080:

```bash
watch -n 1 nvidia-smi
```

You should see ~90% GPU memory utilization (14-15GB on RTX 4080).

### Expected Performance

On RTX 4080, you should see:
- **Throughput**: 80-150 tokens/second (depending on batch size)
- **Latency**: ~10-50ms time to first token
- **GPU Memory**: ~12-14GB usage (90% of 16GB)

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Generate Text
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning?",
    "max_new_tokens": 100,
    "temperature": 0.7
  }'
```

### Streaming Generation
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing:",
    "max_new_tokens": 200,
    "stream": true
  }'
```

## Configuration

Edit `config.yml` to tune vLLM settings:

```yaml
VLLM:
  GPU_MEMORY_UTILIZATION: 0.90  # Use 90% of GPU memory
  MAX_MODEL_LEN: 4096           # Maximum context length
  DTYPE: "bfloat16"             # Data type (bfloat16 for RTX 4080)
  TENSOR_PARALLEL_SIZE: 1       # Number of GPUs
  MAX_NUM_SEQS: 256             # Max parallel sequences
  ENABLE_PREFIX_CACHING: true   # Enable prefix caching
```

### Performance Tuning

For **maximum throughput**:
- Increase `GPU_MEMORY_UTILIZATION` to 0.95
- Increase `MAX_NUM_SEQS` to 512
- Enable `ENABLE_PREFIX_CACHING`

For **lower latency**:
- Decrease `MAX_NUM_SEQS` to 64
- Reduce `GPU_MEMORY_UTILIZATION` to 0.80

## Comparison: Deployment Options

| Feature | Standard (app.py) | Custom FastAPI (app_vllm.py) | OpenAI-Compatible (start_vllm_server.sh) |
|---------|-------------------|------------------------------|-------------------------------------------|
| Throughput | 10-20 tokens/sec | 80-150 tokens/sec | 80-150 tokens/sec |
| Concurrent requests | 1 | 256+ | 256+ |
| Memory efficiency | Low | High (PagedAttention) | High (PagedAttention) |
| Streaming | No | No | **Yes** ✅ |
| API compatibility | Custom | Custom | **OpenAI** ✅ |
| Production-ready | Basic | Advanced | **Production** ✅ |
| Batch processing | Sequential | Continuous | Continuous |
| **Recommendation** | Development only | Custom endpoints | **Production use** ⭐ |

## Troubleshooting

### Issue: "Cannot re-initialize CUDA in forked subprocess"

**Problem**: This error occurs with vLLM v0.11.0+ when using AsyncLLMEngine with FastAPI.

**Solution**: The custom FastAPI server (`app_vllm.py`) has been fixed to use the synchronous LLM class with proper multiprocessing setup. Alternatively, use the OpenAI-compatible server which doesn't have this issue:

```bash
bash start_vllm_server.sh
```

### Issue: CUDA Out of Memory

**Solution**: Reduce GPU memory utilization in the startup script:

Edit `start_vllm_server.sh`:
```bash
GPU_MEMORY_UTIL=0.80  # Reduce from 0.90
```

Or for custom FastAPI, edit `app_vllm.py`:
```python
llm = LLM(
    ...
    gpu_memory_utilization=0.80,  # Reduce from 0.90
)
```

### Issue: vLLM Import Error

**Solution**: Reinstall vLLM:
```bash
uv sync --reinstall-package vllm
```

### Issue: Model Loading Fails

**Solution**: Check if model path is correct:
```bash
ls -la model/
```

The model should be in the `model/` directory.

### Issue: Slow Performance

**Solution**:
1. Check GPU utilization with `nvidia-smi`
2. Ensure CUDA graphs are enabled (default)
3. Increase `MAX_NUM_SEQS` for better batching

## Advanced Features

### Multi-GPU Support

If you have multiple GPUs (future upgrade):

```yaml
VLLM:
  TENSOR_PARALLEL_SIZE: 2  # Use 2 GPUs
```

### Quantization

For even faster inference with minimal quality loss:

```yaml
VLLM:
  QUANTIZATION: "awq"  # or "gptq", "squeezellm"
```

## Monitoring

### Prometheus Metrics

vLLM exposes metrics on `http://localhost:8000/metrics` for monitoring.

### Logging

Check logs for performance stats:
```bash
tail -f /var/log/vllm.log
```

## Production Deployment

For production use:

1. **Use systemd service**:
   ```bash
   sudo systemctl start vllm-api
   ```

2. **Behind reverse proxy** (nginx):
   ```nginx
   location /api {
       proxy_pass http://localhost:8000;
   }
   ```

3. **Enable authentication** with API keys

4. **Monitor with Grafana** + Prometheus

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
