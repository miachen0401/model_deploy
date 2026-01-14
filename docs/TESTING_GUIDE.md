# Quick Testing Guide for vLLM Deployment

## 📋 Prerequisites

1. **Start the vLLM server** (must be running for tests):
   ```bash
   bash start_vllm_server.sh
   ```

2. **Install test dependencies** (one-time):
   ```bash
   uv sync --group dev
   ```

## 🚀 Quick Start

### Run All Tests
```bash
uv run pytest -v
```

### Run Specific Test Categories

```bash
# Health checks only (fast)
uv run pytest tests/test_vllm_server.py::TestVLLMServerHealth -v

# Basic functionality
uv run pytest tests/test_vllm_server.py::TestBasicGeneration -v

# Performance benchmarks
uv run pytest tests/test_vllm_server.py::TestPerformance -v

# Concurrency tests
uv run pytest tests/test_vllm_concurrency.py::TestConcurrency -v

# Load testing
uv run pytest tests/test_vllm_concurrency.py::TestLoadTesting -v
```

## 📊 Test Results Explained

### Error Handling Tests

#### ✅ `test_empty_prompt` - EXPECTED BEHAVIOR
- **What it tests**: Empty prompt rejection
- **Expected**: Should fail with 400 Bad Request
- **Why**: vLLM correctly rejects empty prompts (this is good!)

#### ✅ `test_very_long_prompt` - FIXED
- **What it tests**: Long prompts within context window
- **Prompt length**: ~800 tokens (200 repetitions × ~4 tokens each)
- **Context limit**: 4096 tokens
- **Expected**: Should succeed (well within limits)

#### ✅ `test_exceeding_context_window` - NEW
- **What it tests**: Prompts exceeding the 4096 token limit
- **Prompt length**: ~4800 tokens (exceeds limit)
- **Expected**: Either truncated or rejected with 400 error

## 📈 Performance Expectations (RTX 4080)

Based on your configuration:
- **MAX_NUM_SEQS**: 128 concurrent sequences
- **MAX_MODEL_LEN**: 4096 tokens
- **GPU_MEMORY_UTIL**: 90% (~14.4GB)

### Expected Metrics

| Test | Expected Result |
|------|----------------|
| Response time (100 tokens) | < 5 seconds |
| Throughput | > 50 tokens/sec |
| 5 concurrent requests | >90% success rate |
| 20 concurrent requests | >80% success rate |
| 50 concurrent requests | >70% success rate |

## 🔍 Monitoring During Tests

Open another terminal and run:
```bash
watch -n 1 nvidia-smi
```

Expected during tests:
- **GPU Memory**: ~14-15GB (90% of 16GB)
- **GPU Utilization**: 70-100%
- **Power Draw**: 200-320W

## ⚡ Quick Smoke Test

Fastest way to verify everything works:

```bash
# Start server
bash start_vllm_server.sh

# In another terminal, run smoke tests (~10 seconds)
uv run pytest tests/test_vllm_server.py::TestVLLMServerHealth -v
uv run pytest tests/test_vllm_server.py::TestBasicGeneration::test_simple_completion -v
```

## 🐛 Troubleshooting

### Test Fails: "vLLM server not available"
**Solution**: Make sure the server is running:
```bash
# Check if server is up
curl http://localhost:8000/v1/models

# If not, start it
bash start_vllm_server.sh
```

### Test Fails: Timeout
**Solution**: Some tests take longer under load. Increase timeout:
```bash
# Run with longer timeout
uv run pytest --timeout=600
```

### Test Fails: "CUDA out of memory"
**Solution**: Reduce concurrent sequences in `start_vllm_server.sh`:
```bash
MAX_NUM_SEQS=64  # Reduce from 128
```

Then restart the server.

### Test Fails: Low success rate in concurrency tests
**Solution**: This is normal under very high load. The tests check for:
- 5 concurrent: >90% success
- 20 concurrent: >80% success
- 50 concurrent: >70% success

If lower, reduce `MAX_NUM_SEQS` or increase GPU memory allocation.

## 📝 Understanding Test Results

### All Tests Pass ✅
Your vLLM deployment is working correctly with optimal performance!

### Some Concurrency Tests Fail ⚠️
- **70-80% success rate**: Normal under extreme load
- **<70% success rate**: Consider tuning (reduce MAX_NUM_SEQS)

### Performance Tests Below Threshold ⚠️
- Check GPU utilization (`nvidia-smi`)
- Verify no other processes using GPU
- Consider reducing MAX_MODEL_LEN for higher throughput

## 🎯 Recommended Test Workflow

1. **Start server**: `bash start_vllm_server.sh`
2. **Quick smoke test**: `uv run pytest tests/test_vllm_server.py::TestVLLMServerHealth -v`
3. **Full functional tests**: `uv run pytest tests/test_vllm_server.py -v`
4. **Concurrency tests**: `uv run pytest tests/test_vllm_concurrency.py::TestConcurrency -v`
5. **Load tests** (optional): `uv run pytest tests/test_vllm_concurrency.py::TestLoadTesting -v`

## 📚 More Information

- Full test documentation: `tests/README.md`
- vLLM deployment guide: `VLLM_DEPLOYMENT.md`
- Configuration tuning: Edit `start_vllm_server.sh`

## 🎓 Common Test Scenarios

### Development: Quick validation
```bash
uv run pytest tests/test_vllm_server.py::TestBasicGeneration -v
```

### Pre-deployment: Full validation
```bash
uv run pytest tests/test_vllm_server.py -v
```

### Production: Load testing
```bash
uv run pytest tests/test_vllm_concurrency.py -v
```

### CI/CD: Fast smoke tests
```bash
uv run pytest -m smoke -v
```
