# Architecture Documentation

**Index:** 2025-12-30 - Updated architecture with model conversion pipeline, generation safety features, and modern package management.

## Overview

This project follows a modular architecture pattern separating API deployment, model management, model conversion pipeline, and data analysis capabilities. The system uses `max_new_tokens` for precise generation control and includes comprehensive safety features to prevent runaway generation.

## Directory Structure

```
RL_model/
├── app.py                      # API entry point (deployment)
├── config.yml                  # Configuration file
├── pyproject.toml              # UV package management config
├── requirements.txt            # Dependencies (legacy)
├── render.yaml                 # Render deployment config
├── .env                        # Environment variables (HF tokens)
│
├── model/                      # Model conversion pipeline
│   ├── download_hf_model.py   # Download models from HuggingFace
│   └── export_ckpt_hf.py      # Convert RL checkpoints to HF format
│
├── src/                        # Core source code
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── model.py               # Model handler (loading, inference)
│   └── analysis/              # Data analysis modules
│       ├── __init__.py
│       └── example_analyzer.py # Template for analysis scripts
│
├── tests/                      # Test suite
│   └── __init__.py
│
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md        # This file
│   ├── project_structure.txt  # Directory tree
│   └── RECORD_Change.md       # Change history
│
├── exported_hf/                # Converted local model (generated)
├── hf_repo/                    # Downloaded HF model cache (generated)
│
└── [test/utility scripts]
    ├── test_curl.sh           # API testing with curl
    ├── example_usage.py       # Usage examples
    └── validate_setup.py      # Pre-deployment validation
```

## Component Architecture

### 1. API Layer (`app.py`)
**Purpose**: Deployment entry point for the REST API.

**Responsibilities**:
- Define FastAPI application and routes
- Handle HTTP requests/responses with Pydantic models
- Route business logic to appropriate modules
- Manage application lifecycle (startup/shutdown)
- Enforce generation limits and timeouts
- Control concurrent requests via semaphore

**Key Features**:
- **Semaphore-based concurrency control**: Limits simultaneous generation requests (configurable via `OCCUPANCY_SEMAPHORE`)
- **Timeout protection**: 60-second timeout on all generation requests
- **Request validation**: Pydantic models with clear parameter limits
- **Progress visualization**: Detailed logging during startup

**Dependencies**:
- `src.config` for configuration
- `src.model` for model operations

**API Endpoints**:
- `GET /` - Root endpoint with API info
- `POST /generate` - Text generation with safety limits
- `GET /health` - Health check
- `GET /model-info` - Model metadata

### 2. Model Management (`src/model.py`)
**Purpose**: Encapsulate all model-related operations.

**Key Components**:
- `QwenModelHandler`: Main class for model operations
  - `load_model()`: Load model from Hugging Face with progress tracking
  - `generate()`: Text generation with `max_new_tokens` (512 token hard limit)
  - `get_model_info()`: Model metadata
  - `cleanup()`: Resource cleanup
  - `_select_device()`: Smart device selection (CUDA > MPS > CPU)

**Generation Safety Features**:
- **Hard limit**: Maximum 512 new tokens per request
- **Early stopping**: Stops when EOS token is generated
- **Input truncation**: Prompts limited to 2048 tokens
- **Debug logging**: Shows input length and tokens to generate

**Singleton Pattern**: Uses `get_model_handler()` for global instance.

### 3. Configuration (`src/config.py`)
**Purpose**: Centralized configuration management.

**Key Components**:
- `Config`: Configuration class
  - Reads `config.yml`
  - Manages environment variables (HF_TOKEN, PORT)
  - Provides typed access to settings

**Configuration Properties**:
- `model_name`: Model path (local or HF)
- `hf_token`: HuggingFace API token
- `port`: Server port (default: 8000)
- `max_new_tokens`: Hard generation limit (512)
- `default_max_new_tokens`: API default (64)
- `generation_timeout`: Request timeout (60s)
- `occupancy_semaphore`: Concurrent request limit (1)

**Singleton Pattern**: Uses `get_config()` for global instance.

### 4. Model Conversion Pipeline (`model/`)
**Purpose**: Convert custom RL checkpoints to HuggingFace format.

**Scripts**:

#### `download_hf_model.py`
- Downloads model snapshots from HuggingFace
- Supports authentication via HF_TOKEN
- Configurable output directory
- Resume support for interrupted downloads

#### `export_ckpt_hf.py`
- Converts RL actor checkpoints (.pt files) to HF format
- Loads base model architecture (e.g., Qwen2.5-1.5B-Instruct)
- Applies checkpoint weights with prefix stripping
- Exports to safetensors format with proper config
- Validates export by loading the model

**Usage Flow**:
```bash
# 1. Download raw checkpoint
uv run python model/download_hf_model.py

# 2. Convert to HF format
uv run python model/export_ckpt_hf.py

# 3. Update config.yml to use exported_hf/
```

### 5. Analysis Framework (`src/analysis/`)
**Purpose**: Data analysis and model management scripts.

**Design**:
- Base class `DataAnalyzer` for common functionality
- Specialized analyzers inherit from base class
- Can reuse model handler from `src.model`

**Example Use Cases**:
- Performance monitoring
- Model output analysis
- Data processing pipelines
- Batch inference

### 6. Testing (`tests/`)
**Purpose**: Future unit and integration tests.

## Data Flow

### API Request Flow
```
User Request (max_new_tokens=N)
    ↓
app.py (FastAPI endpoint)
    ↓
[Semaphore Acquisition] (limit concurrent requests)
    ↓
[Validation] min(request.max_new_tokens, config.max_new_tokens)
    ↓
get_model_handler() ← src/model.py
    ↓
QwenModelHandler.generate(max_new_tokens=N)
    ↓
[Safety Check] min(N, 512) with early_stopping=True
    ↓
Hugging Face Model (inference with timeout)
    ↓
Response back through layers
    ↓
[Semaphore Release]
```

### Model Conversion Flow
```
HuggingFace Checkpoint (actor/*.pt)
    ↓
download_hf_model.py
    ↓
Local Cache (hf_repo/)
    ↓
export_ckpt_hf.py
    ↓
Load Base Model (Qwen2.5-1.5B-Instruct)
    ↓
Apply Checkpoint Weights
    ↓
Strip Prefixes (module., actor., etc.)
    ↓
Export to SafeTensors (exported_hf/)
    ↓
Validation (load test)
    ↓
Ready for Use
```

### Analysis Script Flow
```
Analysis Script
    ↓
Import src.model, src.config
    ↓
get_model_handler()
    ↓
Load model if needed
    ↓
Perform analysis
    ↓
Generate results/reports
```

## Design Principles

### 1. Separation of Concerns
- **API**: Only handles HTTP, routing, and request limiting
- **Model**: Only handles model operations and inference
- **Config**: Only handles configuration
- **Analysis**: Independent data processing
- **Conversion**: Separate pipeline for model preparation

### 2. Safety First
- **Multiple protection layers**:
  1. API-level timeout (60s)
  2. Model-level token limit (512)
  3. Early stopping on EOS
  4. Concurrent request limiting
- **Clear error messages**: Timeout and limit violations explained
- **Progress visibility**: Users see what's happening during loading

### 3. Reusability
- Model handler can be used by both API and analysis scripts
- Configuration accessible across all modules
- Conversion scripts reusable for different models
- No code duplication

### 4. Singleton Pattern
- Global instances for model and config
- Ensures single model instance (memory efficient)
- Consistent state across application

### 5. Extensibility
- Easy to add new analysis scripts in `src/analysis/`
- New API endpoints can reuse existing model handler
- Configuration changes don't require code changes
- Model conversion pipeline adaptable to different architectures

## Module Dependencies

```
app.py
  ├─→ src.config (get_config)
  └─→ src.model (get_model_handler)

src/model.py
  └─→ [No internal dependencies, uses torch/transformers]

src/config.py
  └─→ [No internal dependencies, uses yaml/os]

src/analysis/*.py
  ├─→ src.config (optional)
  └─→ src.model (optional)

model/download_hf_model.py
  ├─→ config.yml (reads HUGGINGFACE_MODEL.NAME)
  └─→ .env (reads HF_TOKEN)

model/export_ckpt_hf.py
  ├─→ config.yml (reads HUGGINGFACE_MODEL.BASE)
  └─→ .env (reads HF_TOKEN)
```

## Configuration Schema

### config.yml
```yaml
HUGGINGFACE_MODEL:
  NAME: "exported_hf"              # Model path (local or HF)
  BASE: "Qwen/Qwen2.5-1.5B-Instruct"  # Base architecture

GENERATION:
  MAX_NEW_TOKENS: 512              # Hard limit
  DEFAULT_MAX_NEW_TOKENS: 64       # API default
  TIMEOUT_SECONDS: 60              # Request timeout
  OCCUPANCY_SEMAPHORE: 1           # Concurrent requests limit
```

### Environment Variables (.env)
```
HF_TOKEN=hf_...           # HuggingFace API token
HF_TOKEN_READ=hf_...      # Alternative token name
PORT=8000                 # Server port (optional)
```

## Adding New Analysis Scripts

To add a new data analysis script:

1. Create file in `src/analysis/your_analyzer.py`
2. Import needed modules:
   ```python
   from src.config import get_config
   from src.model import get_model_handler
   ```
3. Implement your logic
4. Can run independently or import in other scripts

Example:
```python
from src.model import get_model_handler
from src.config import get_config

def analyze_data():
    config = get_config()
    model_handler = get_model_handler()

    # Load model if needed
    if not model_handler.is_loaded():
        model_handler.load_model(
            model_name=config.model_name,
            hf_token=config.hf_token
        )

    # Your analysis logic here
    results = model_handler.generate(
        prompt="Analyze this data...",
        max_new_tokens=100
    )

    return results
```

## Package Management

### UV (Modern - Recommended)
```bash
# Install dependencies
uv sync

# Run application
uv run python app.py

# Run scripts
uv run python model/download_hf_model.py
```

### pip (Legacy)
```bash
pip install -r requirements.txt
python app.py
```

**Note**: Project uses `pyproject.toml` with `dependency-groups` for modern Python packaging.

## Deployment Considerations

### Production Deployment (Render)
- `app.py` is the entry point
- All `src/` modules are imported at runtime
- Configuration from `config.yml` and environment variables
- Model loaded once on startup (singleton pattern)
- Concurrent requests limited by semaphore

### Local Development
- Same structure as production
- Can run analysis scripts independently
- API and analysis scripts can coexist
- Use `exported_hf/` for local converted models

### Model Preparation
1. Download checkpoint from HuggingFace
2. Convert to proper HF format using export script
3. Update `config.yml` to point to local model
4. Server loads model on startup

## Safety Features

### Generation Protection
1. **Token Limit**: Hard cap of 512 new tokens
2. **Timeout**: 60-second request timeout
3. **Early Stopping**: Stops at EOS token
4. **Concurrent Limit**: Semaphore prevents overload

### Error Handling
- Model load failures logged with details
- Generation timeouts return 504 with clear message
- Invalid requests rejected with 400/422 errors
- Health check endpoint for monitoring

### Progress Visibility
- Startup shows step-by-step progress with timing
- Generation logs show token counts
- Debug logging available for troubleshooting

## Future Enhancements

1. **Database Integration**: Add `src/database.py` for data persistence
2. **Monitoring**: Add `src/monitoring.py` for metrics collection
3. **Caching**: Add `src/cache.py` for response caching
4. **Authentication**: Add `src/auth.py` for API security
5. **Advanced Analysis**: Expand `src/analysis/` with specialized tools
6. **Multi-model Support**: Load multiple models simultaneously
7. **Streaming Responses**: Support streaming text generation
8. **GPU Optimization**: Better CUDA/MPS device utilization
