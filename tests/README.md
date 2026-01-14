# vLLM Model Deployment Tests

Comprehensive test suite for vLLM-deployed Qwen 2.5B model, including functional tests, performance benchmarks, and concurrency/load tests.

## Test Structure

```
tests/
├── conftest.py                    # Pytest fixtures and configuration
├── test_vllm_server.py           # Functional tests for vLLM OpenAI-compatible server
├── test_vllm_concurrency.py      # Concurrency and load tests
└── test_api.py                   # Legacy API tests (not pytest-based)
```

## Prerequisites

### 1. Install Test Dependencies

```bash
# Install dev dependencies including pytest
uv sync --group dev
```

### 2. Start the vLLM Server

**IMPORTANT**: Tests require the vLLM OpenAI-compatible server to be running.

```bash
# In one terminal, start the vLLM server
bash start_vllm_server.sh
```

Wait for the server to fully initialize (you'll see "Uvicorn running on...").

## Running Tests

### Run All Tests

```bash
# Run all tests
uv run pytest

# Run with detailed output
uv run pytest -v

# Run with output from tests (print statements)
uv run pytest -s
```

### Run Specific Test Files

```bash
# Run only functional tests
uv run pytest tests/test_vllm_server.py

# Run only concurrency tests
uv run pytest tests/test_vllm_concurrency.py
```

### Run Specific Test Classes or Functions

```bash
# Run specific test class
uv run pytest tests/test_vllm_server.py::TestBasicGeneration

# Run specific test function
uv run pytest tests/test_vllm_server.py::TestBasicGeneration::test_simple_completion
```

### Run Tests by Marker

```bash
# Run only performance tests
uv run pytest -m performance

# Run only concurrency tests
uv run pytest -m concurrency

# Run smoke tests (quick validation)
uv run pytest -m smoke
```

### Parallel Test Execution

```bash
# Run tests in parallel using multiple CPUs
uv run pytest -n auto

# Run with 4 workers
uv run pytest -n 4
```

## Test Categories

### 1. Server Health Tests (`TestVLLMServerHealth`)

Validates that the vLLM server is running and the model is loaded.

```bash
uv run pytest tests/test_vllm_server.py::TestVLLMServerHealth -v
```

### 2. Basic Generation Tests (`TestBasicGeneration`)

Tests fundamental text generation with various prompts and configurations.

```bash
uv run pytest tests/test_vllm_server.py::TestBasicGeneration -v
```

### 3. Sampling Parameters Tests (`TestSamplingParameters`)

Tests different sampling parameters (temperature, top_p, max_tokens).

```bash
uv run pytest tests/test_vllm_server.py::TestSamplingParameters -v
```

### 4. Streaming Tests (`TestStreamingGeneration`)

Tests streaming text generation capabilities.

```bash
uv run pytest tests/test_vllm_server.py::TestStreamingGeneration -v
```

### 5. Performance Tests (`TestPerformance`)

Benchmarks response time and throughput.

```bash
uv run pytest tests/test_vllm_server.py::TestPerformance -v
```

**Expected Results on RTX 4080:**
- Response time: < 5 seconds for 100 tokens
- Throughput: > 50 tokens/second

### 6. Concurrency Tests (`TestConcurrency`)

Tests parallel request handling with varying concurrency levels.

```bash
uv run pytest tests/test_vllm_concurrency.py::TestConcurrency -v
```

**Test scenarios:**
- 5 concurrent requests
- 10 concurrent requests
- 20 concurrent requests
- 50 concurrent requests (burst test)

### 7. Load Testing (`TestLoadTesting`)

Tests sustained load and burst scenarios.

```bash
uv run pytest tests/test_vllm_concurrency.py::TestLoadTesting -v
```

**Test scenarios:**
- Sustained low load (5 concurrent for 10 seconds)
- Burst load (50 requests simultaneously)

### 8. Concurrency Limits Tests (`TestConcurrencyLimits`)

Tests behavior at various concurrency limits (5, 10, 20, 50, 100).

```bash
uv run pytest tests/test_vllm_concurrency.py::TestConcurrencyLimits -v
```

### 9. Throughput Optimization Tests (`TestThroughputOptimization`)

Compares throughput with different configurations.

```bash
uv run pytest tests/test_vllm_concurrency.py::TestThroughputOptimization -v
```

## Performance Expectations

### RTX 4080 (16GB VRAM) - Optimized Settings

**Configuration:**
- Model: Qwen 2.5B
- GPU Memory Utilization: 90% (~14.4GB)
- Max Concurrent Sequences: 128
- Data Type: bfloat16
- Max Context Length: 4096 tokens

**Expected Performance:**
- **Throughput**: 80-150 tokens/second
- **Latency** (p50): 0.5-2.0 seconds for 50 tokens
- **Latency** (p95): 1.0-4.0 seconds for 50 tokens
- **Concurrent requests**: Handle 50+ simultaneously with >80% success rate
- **Sustained load**: 5-10 concurrent requests continuously

## Test Configuration

### Adjusting Test Parameters

Edit `tests/conftest.py` to modify test configurations:

```python
@pytest.fixture
def concurrency_test_config() -> dict:
    return {
        "concurrent_requests": [1, 5, 10, 20, 50],  # Modify these
        "max_tokens": 50,
        "temperature": 0.7,
        "timeout": 60,
    }
```

### Adjusting Server Configuration

Edit `start_vllm_server.sh` to modify server settings:

```bash
GPU_MEMORY_UTIL=0.90      # Reduce if OOM errors
MAX_NUM_SEQS=128          # Increase for more concurrency
MAX_MODEL_LEN=4096        # Adjust context length
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: vLLM Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install uv
          uv sync --group dev

      - name: Start vLLM server
        run: |
          bash start_vllm_server.sh &
          sleep 30  # Wait for server to start

      - name: Run tests
        run: uv run pytest -v
```

## Troubleshooting

### Issue: Tests fail with "vLLM server not available"

**Solution**: Make sure the vLLM server is running:
```bash
bash start_vllm_server.sh
```

Wait ~30 seconds for the model to load.

### Issue: Tests timeout

**Solution**: Increase timeout in `pytest.ini`:
```ini
timeout = 600  # Increase from 300
```

Or adjust individual test timeouts.

### Issue: Concurrency tests fail with low success rate

**Solution**: Reduce max concurrent sequences in `start_vllm_server.sh`:
```bash
MAX_NUM_SEQS=64  # Reduce from 128
```

### Issue: OOM (Out of Memory) errors

**Solution**: Reduce GPU memory utilization:
```bash
GPU_MEMORY_UTIL=0.80  # Reduce from 0.90
```

Or reduce max concurrent sequences:
```bash
MAX_NUM_SEQS=64  # Reduce from 128
```

### Issue: Tests are too slow

**Solution**: Run tests in parallel:
```bash
uv run pytest -n auto
```

Or run only smoke tests:
```bash
uv run pytest -m smoke
```

## Writing New Tests

### Example Test Function

```python
def test_custom_generation(openai_client: OpenAI):
    """Test custom generation scenario."""
    response = openai_client.completions.create(
        model="model",
        prompt="Your custom prompt",
        max_tokens=100,
        temperature=0.7,
    )

    assert response is not None
    assert len(response.choices[0].text) > 0
    assert response.usage.completion_tokens <= 100
```

### Adding Test Markers

```python
@pytest.mark.slow
@pytest.mark.performance
def test_long_generation(openai_client: OpenAI):
    """Test long text generation."""
    # Your test code
```

## Best Practices

1. **Always start the vLLM server before running tests**
2. **Run quick smoke tests first** to validate basic functionality
3. **Run performance tests separately** as they take longer
4. **Monitor GPU usage** during concurrency tests with `nvidia-smi`
5. **Adjust server configuration** based on your hardware
6. **Use markers** to organize and selectively run tests
7. **Check test output** for performance metrics and warnings

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
