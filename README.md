# Qwen 1.5B Fine-tuned Model API

FastAPI deployment for serving a fine-tuned Qwen 1.5B model from Hugging Face on Render.

## Overview

This project provides a production-ready REST API for the fine-tuned Qwen 1.5B model (`Hula0401/finetuned`) using FastAPI and can be deployed to Render with one click.

## Features

- **FastAPI REST API** with automatic documentation
- **Health checks** and model info endpoints
- **Text generation** with configurable parameters
- **GPU/CPU support** with automatic device detection
- **Production-ready** with proper logging and error handling
- **Easy deployment** to Render with included configuration

## Quick Start

### Local Development

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure environment:**
   - Copy `.env.example` to `.env` (if exists) or create `.env`:
   ```
   HF_TOKEN_READ=your_huggingface_token_here
   ```

3. **Start the server:**
```bash
python app.py
# or
uvicorn app:app --reload --port 8000
```

4. **Test the API:**
```bash
# Run test suite
python test_api.py

# Run usage examples
python example_usage.py

# Or access the interactive docs at:
# http://localhost:8000/docs
```

### Deploy to Render

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed deployment instructions.

Quick steps:
1. Push this repository to GitHub
2. Create a new Web Service on [Render](https://dashboard.render.com/)
3. Connect your repository (Render will auto-detect `render.yaml`)
4. Add `HF_TOKEN_READ` environment variable in Render dashboard
5. Deploy!

## API Endpoints

### GET `/`
Root endpoint with API information.

### GET `/health`
Health check endpoint for monitoring.

### GET `/model-info`
Get model metadata and configuration.

### POST `/generate`
Generate text from the model.

**Request body:**
```json
{
  "prompt": "Your prompt here",
  "max_length": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "num_return_sequences": 1
}
```

**Response:**
```json
{
  "generated_texts": ["Generated text here..."],
  "model_name": "Hula0401/finetuned",
  "device": "cuda"
}
```

## Project Structure

```
.
├── app.py                  # FastAPI application (deployment entry)
├── requirements.txt        # Python dependencies
├── render.yaml            # Render deployment config
├── config.yml             # Model configuration
├── test_api.py           # API test suite
├── example_usage.py      # Usage examples
├── .env                  # Environment variables (not in git)
├── .gitignore           # Git ignore rules
├── src/                  # Source code modules
│   ├── __init__.py
│   ├── config.py        # Configuration management
│   ├── model.py         # Model loading and inference
│   └── analysis/        # Data analysis scripts (for future use)
│       ├── __init__.py
│       └── example_analyzer.py
├── tests/               # Test suite
│   └── __init__.py
└── docs/
    ├── DEPLOYMENT.md    # Deployment documentation
    └── RECORD_Change.md # Change history
```

## Configuration

### `config.yml`
```yaml
HUGGINGFACE_MODEL:
  NAME: "Hula0401/finetuned"
```

### Environment Variables
- `HF_TOKEN_READ`: Hugging Face read token for model access
- `PORT`: Server port (auto-set by Render, default 8000 locally)

## Development

### Running Tests
```bash
# Full test suite
python test_api.py

# Test deployed version
python test_api.py --url https://your-app.onrender.com

# Custom prompt test
python test_api.py --prompt "Your custom prompt here"
```

### Running Examples
```bash
# Edit API_URL in example_usage.py first
python example_usage.py
```

## Model Information

- **Base Model**: Qwen 1.5B
- **Fine-tuned Model**: `Hula0401/finetuned`
- **Model Type**: Causal Language Model
- **Framework**: Transformers + PyTorch

## Requirements

- Python 3.11+
- 2GB+ RAM (CPU inference)
- GPU recommended for faster inference (requires Render Pro plan)

## Documentation

- [Quick Start](QUICK_START.md) - Get started quickly
- [Architecture](docs/ARCHITECTURE.md) - System architecture and design
- [Deployment Guide](docs/DEPLOYMENT.md) - Detailed deployment instructions
- [Summary](docs/SUMMARY.md) - Project overview and structure
- [Change History](docs/RECORD_Change.md) - Track all changes
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when running locally)

## License

See LICENSE file for details.

## Support

For issues and questions, please open an issue in the repository.
