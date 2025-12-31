# Qwen 1.5B Fine-tuned Model API

FastAPI deployment for serving a fine-tuned Qwen 1.5B model with comprehensive safety features and modern package management.

## Overview

This project provides a production-ready REST API for fine-tuned Qwen models with built-in safety features including generation limits, timeout protection, and concurrent request control. It includes a model conversion pipeline for transforming RL checkpoints to HuggingFace format.

## Features

- ✅ **FastAPI REST API** with automatic documentation
- ✅ **Safety Features**: Token limits (512 max), 60s timeout, concurrent request control
- ✅ **Model Conversion Pipeline**: Convert RL checkpoints to HuggingFace format
- ✅ **Modern Package Management**: UV support with pyproject.toml
- ✅ **Progress Visualization**: Detailed startup and generation progress
- ✅ **Smart Device Selection**: Automatic CUDA > MPS > CPU detection
- ✅ **Health Checks**: Health and model info endpoints
- ✅ **Production-Ready**: Proper logging, error handling, and safety limits

## Quick Start

### Local Development

#### Option 1: UV (Recommended - Fast)
```bash
# Install UV if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Start the server
uv run python app.py

# Server will start at http://localhost:8000
```

#### Option 2: pip (Traditional)
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

### Configuration

1. **Create `.env` file:**
```env
HF_TOKEN=your_huggingface_token_here
HF_TOKEN_READ=your_huggingface_token_here  # Alternative name
PORT=8000  # Optional, defaults to 8000
```

2. **Configure `config.yml`:**
```yaml
HUGGINGFACE_MODEL:
  NAME: "exported_hf"  # Local model path or HF repo
  BASE: "Qwen/Qwen2.5-1.5B-Instruct"  # Base architecture

GENERATION:
  MAX_NEW_TOKENS: 512        # Hard limit
  DEFAULT_MAX_NEW_TOKENS: 64 # API default
  TIMEOUT_SECONDS: 60        # Request timeout
  OCCUPANCY_SEMAPHORE: 1     # Concurrent requests limit
```

### Model Conversion (For RL Checkpoints)

If you have a custom RL checkpoint that needs to be converted to HuggingFace format:

```bash
# 1. Download the checkpoint from HuggingFace
uv run python model/download_hf_model.py

# 2. Convert to HuggingFace format
uv run python model/export_ckpt_hf.py

# 3. The converted model will be in exported_hf/
# Update config.yml to use it:
# HUGGINGFACE_MODEL:
#   NAME: "exported_hf"
```

### Testing the API

```bash
# Run usage examples
uv run python example_usage.py

# Test with curl
./test_curl.sh

# Or access the interactive docs:
# http://localhost:8000/docs
```

## API Endpoints

### GET `/`
Root endpoint with API information and version.

**Response:**
```json
{
  "message": "Qwen Model API",
  "version": "1.0.0",
  "endpoints": {
    "generate": "/generate",
    "health": "/health",
    "model_info": "/model-info"
  }
}
```

### GET `/health`
Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### GET `/model-info`
Get model metadata and configuration.

**Response:**
```json
{
  "model_name": "exported_hf",
  "device": "cpu",
  "model_type": "qwen2",
  "vocab_size": 151643
}
```

### POST `/generate`
Generate text from the model with safety limits.

**Request body:**
```json
{
  "prompt": "What is artificial intelligence?",
  "max_new_tokens": 64,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "num_return_sequences": 1
}
```

**Parameters:**
- `prompt` (string, required): Input text prompt
- `max_new_tokens` (int, 1-512): Maximum new tokens to generate (default: 64, hard limit: 512)
- `temperature` (float, 0.1-2.0): Sampling temperature (default: 0.7)
- `top_p` (float, 0.0-1.0): Nucleus sampling (default: 0.9)
- `top_k` (int, 0-100): Top-k sampling (default: 50)
- `num_return_sequences` (int, 1-5): Number of sequences (default: 1)

**Response:**
```json
{
  "generated_texts": [
    "Artificial intelligence is the simulation of human intelligence..."
  ],
  "model_name": "exported_hf",
  "device": "cpu"
}
```

**Safety Features:**
- ✅ Maximum 512 new tokens per request (hard limit)
- ✅ 60-second timeout protection
- ✅ Early stopping when EOS token is generated
- ✅ Semaphore-based concurrent request limiting
- ✅ Clear error messages for timeouts and limits

## Project Structure

```
RL_model/
├── app.py                      # FastAPI application entry point
├── config.yml                  # Model and generation configuration
├── pyproject.toml              # UV package management (modern)
├── requirements.txt            # pip dependencies (legacy)
├── render.yaml                 # Render deployment config
├── .env                        # Environment variables (not in git)
├── .gitignore                  # Git ignore rules
│
├── model/                      # Model conversion pipeline
│   ├── download_hf_model.py   # Download from HuggingFace
│   └── export_ckpt_hf.py      # Convert RL checkpoint to HF format
│
├── src/                        # Core source code
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── model.py               # Model loading and inference
│   └── analysis/              # Data analysis scripts
│       ├── __init__.py
│       └── example_analyzer.py
│
├── tests/                      # Test suite
│   └── __init__.py
│
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md        # Comprehensive architecture docs
│   ├── project_structure.txt  # Directory tree and patterns
│   └── RECORD_Change.md       # Change history
│
├── exported_hf/                # Converted local model (generated)
├── hf_repo/                    # Downloaded HF cache (generated)
│
└── [utilities]
    ├── example_usage.py       # Usage examples
    ├── test_curl.sh          # API testing with curl
    └── validate_setup.py     # Pre-deployment validation
```

## Configuration Details

### Generation Safety Limits

All configured in `config.yml`:

```yaml
GENERATION:
  MAX_NEW_TOKENS: 512        # Hard limit, prevents runaway generation
  DEFAULT_MAX_NEW_TOKENS: 64 # Default for API requests
  TIMEOUT_SECONDS: 60        # Request timeout in seconds
  OCCUPANCY_SEMAPHORE: 1     # Max concurrent generation requests
```

**Protection Layers:**
1. **API Level**: Timeout (60s), semaphore (1 concurrent), validation
2. **Model Level**: Token limit (512), early stopping, input truncation
3. **Config Level**: Centralized limits, environment-based settings

### Environment Variables

Required in `.env`:
- `HF_TOKEN`: HuggingFace API token (for downloading/converting models)
- `HF_TOKEN_READ`: Alternative token name (backward compatibility)

Optional:
- `PORT`: Server port (default: 8000, auto-set by Render)

## Development

### Package Management

**UV (Modern - Recommended):**
```bash
uv sync                              # Install dependencies
uv run python app.py                 # Run server
uv run python model/download_hf_model.py  # Download model
uv run python model/export_ckpt_hf.py     # Convert checkpoint
```

**pip (Legacy):**
```bash
pip install -r requirements.txt
python app.py
```

### Progress Visualization

The server shows detailed progress during startup:

```
============================================================
🌟 Application Startup - Initializing...
============================================================
📋 Loading configuration...
   Model: exported_hf
   Port: 8000
============================================================
🚀 Starting model loading: exported_hf
============================================================
✓ Device selected: cpu (0.00s)
📝 Loading tokenizer...
✓ Tokenizer loaded (vocab_size=151643, 0.34s)
🧠 Loading model weights (this may take 15-30s)...
✓ Model weights loaded (21.08s)
⚙️  Setting model to evaluation mode...
✓ Model ready (0.00s)
============================================================
✅ Model loading complete! Total time: 21.44s
============================================================
🎉 Application ready to accept requests!
```

### Killing the Server

If the server gets stuck:

```bash
# Find and kill by port
kill -9 $(lsof -ti:8000)

# Or find by process name
pkill -9 -f "python.*app.py"
```

## Deploy to Render

Quick steps:
1. Push this repository to GitHub
2. Create a new Web Service on [Render](https://dashboard.render.com/)
3. Connect your repository (Render will auto-detect `render.yaml`)
4. Add `HF_TOKEN` environment variable in Render dashboard
5. Deploy!

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for deployment considerations.

## Model Information

- **Base Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Current Model**: Local converted model from RL checkpoint
- **Model Type**: Causal Language Model
- **Framework**: Transformers + PyTorch
- **Format**: SafeTensors

## Requirements

- **Python**: 3.9+
- **RAM**: 2GB+ (CPU inference), 4GB+ recommended
- **GPU**: Optional but recommended for faster inference
  - CUDA support (NVIDIA GPUs)
  - MPS support (Apple Silicon M1/M2)
- **Disk**: ~3GB for model weights

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - Comprehensive system design
- [Project Structure](docs/project_structure.txt) - Directory tree and patterns
- [Change History](docs/RECORD_Change.md) - Track all changes
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when running)

## Safety & Best Practices

### Generation Safety
✅ **Token Limits**: Hard cap of 512 new tokens per request
✅ **Timeouts**: 60-second timeout on all generation requests
✅ **Early Stopping**: Automatic stop when EOS token is generated
✅ **Concurrent Control**: Semaphore limits simultaneous requests

### Error Handling
✅ **Clear Messages**: Descriptive error responses with HTTP status codes
✅ **Health Monitoring**: `/health` endpoint for uptime checks
✅ **Validation**: Pydantic models validate all requests
✅ **Logging**: Structured logging for debugging

### Performance
✅ **Singleton Pattern**: Single model instance (memory efficient)
✅ **Device Optimization**: Smart CUDA > MPS > CPU selection
✅ **Progress Tracking**: Visibility into long-running operations
✅ **Async Support**: Non-blocking request handling

## Troubleshooting

### Server Won't Start
- Check if port 8000 is in use: `lsof -i:8000`
- Verify HF_TOKEN in `.env` file
- Check model exists at path in `config.yml`

### Generation Timeout
- Reduce `max_new_tokens` in request (default 64, max 512)
- Simplify prompt
- Check `TIMEOUT_SECONDS` in `config.yml` (default 60s)

### Out of Memory
- Reduce `OCCUPANCY_SEMAPHORE` to 1 (only 1 concurrent request)
- Use CPU instead of GPU if GPU memory is limited
- Reduce batch size (`num_return_sequences`)

## License

See LICENSE file for details.

## Support

For issues and questions, please open an issue in the repository.

---

**Built with:**
- FastAPI - Modern web framework
- Transformers - HuggingFace library
- UV - Fast Python package manager
- PyTorch - Deep learning framework
- Pydantic - Data validation
