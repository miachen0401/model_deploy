# Architecture Documentation

**Index:** 2025-12-30 - Project architecture overview showing modular design with separated concerns for API, model management, and data analysis.

## Overview

This project follows a modular architecture pattern separating API deployment, model management, and data analysis capabilities.

## Directory Structure

```
RL_model/
├── app.py                      # API entry point (deployment)
├── config.yml                  # Configuration file
├── requirements.txt            # Dependencies
├── render.yaml                 # Render deployment config
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
│   ├── DEPLOYMENT.md          # Deployment guide
│   ├── ARCHITECTURE.md        # This file
│   └── RECORD_Change.md       # Change history
│
└── [test/utility scripts]
    ├── test_api.py            # API testing
    ├── example_usage.py       # Usage examples
    └── validate_setup.py      # Pre-deployment validation
```

## Component Architecture

### 1. API Layer (`app.py`)
**Purpose**: Deployment entry point for the REST API.

**Responsibilities**:
- Define FastAPI application and routes
- Handle HTTP requests/responses
- Route business logic to appropriate modules
- Manage application lifecycle (startup/shutdown)

**Dependencies**:
- `src.config` for configuration
- `src.model` for model operations

### 2. Model Management (`src/model.py`)
**Purpose**: Encapsulate all model-related operations.

**Key Components**:
- `QwenModelHandler`: Main class for model operations
  - `load_model()`: Load model from Hugging Face
  - `generate()`: Text generation with configurable parameters
  - `get_model_info()`: Model metadata
  - `cleanup()`: Resource cleanup

**Singleton Pattern**: Uses `get_model_handler()` for global instance.

### 3. Configuration (`src/config.py`)
**Purpose**: Centralized configuration management.

**Key Components**:
- `Config`: Configuration class
  - Reads `config.yml`
  - Manages environment variables
  - Provides typed access to settings

**Singleton Pattern**: Uses `get_config()` for global instance.

### 4. Analysis Framework (`src/analysis/`)
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

### 5. Testing (`tests/`)
**Purpose**: Future unit and integration tests.

## Data Flow

### API Request Flow
```
User Request
    ↓
app.py (FastAPI endpoint)
    ↓
get_model_handler() ← src/model.py
    ↓
QwenModelHandler.generate()
    ↓
Hugging Face Model (inference)
    ↓
Response back through layers
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
- **API**: Only handles HTTP and routing
- **Model**: Only handles model operations
- **Config**: Only handles configuration
- **Analysis**: Independent data processing

### 2. Reusability
- Model handler can be used by both API and analysis scripts
- Configuration accessible across all modules
- No code duplication

### 3. Singleton Pattern
- Global instances for model and config
- Ensures single model instance (memory efficient)
- Consistent state across application

### 4. Extensibility
- Easy to add new analysis scripts in `src/analysis/`
- New API endpoints can reuse existing model handler
- Configuration changes don't require code changes

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
    results = model_handler.generate(...)

    return results
```

## Deployment Considerations

### Production Deployment (Render)
- `app.py` is the entry point
- All `src/` modules are imported at runtime
- Configuration from `config.yml` and environment variables

### Local Development
- Same structure as production
- Can run analysis scripts independently
- API and analysis scripts can coexist

## Future Enhancements

1. **Database Integration**: Add `src/database.py` for data persistence
2. **Monitoring**: Add `src/monitoring.py` for metrics collection
3. **Caching**: Add `src/cache.py` for response caching
4. **Authentication**: Add `src/auth.py` for API security
5. **Advanced Analysis**: Expand `src/analysis/` with specialized tools
