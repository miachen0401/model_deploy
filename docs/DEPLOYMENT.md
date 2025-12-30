# Hugging Face Model Deployment on Render

**Index:** 2025-12-30 - Initial deployment setup for Qwen 1.5B fine-tuned model on Render platform.

## Overview

This document describes the deployment of the fine-tuned Qwen 1.5B model (`Hula0401/finetuned`) on Render using FastAPI.

## Components

### 1. FastAPI Application (`app.py`)
- Serves the Hugging Face model via REST API
- Endpoints: `/`, `/health`, `/model-info`, `/generate`
- Uses async lifespan management for model loading
- Supports GPU/CPU inference with automatic device detection

### 2. Dependencies (`requirements.txt`)
- FastAPI and Uvicorn for web server
- Transformers and PyTorch for model inference
- Configuration management with PyYAML and python-dotenv

### 3. Render Configuration (`render.yaml`)
- Deployment configuration for Render platform
- Uses Python 3.11, Standard plan, Oregon region
- Health check endpoint configured

### 4. Test Script (`test_api.py`)
- Comprehensive testing for all endpoints
- Can test both local and deployed instances
- Includes multiple example prompts

## Deployment Steps

### Step 1: Prepare Repository
1. Ensure all files are committed to Git
2. Push repository to GitHub/GitLab
3. Verify `config.yml` and `.env` are properly configured (`.env` should be in `.gitignore`)

### Step 2: Create Render Service
1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your Git repository
4. Render will auto-detect `render.yaml` configuration

### Step 3: Configure Environment Variables
In Render dashboard, add:
- `HF_TOKEN_READ`: Your Hugging Face read token (from `.env`)

### Step 4: Deploy
1. Render will automatically build and deploy
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Initial deployment takes 5-10 minutes (model download + build)

### Step 5: Verify Deployment
Once deployed, test using the provided URL:
```bash
python test_api.py --url https://your-app-name.onrender.com
```

## API Usage

### Health Check
```bash
curl https://your-app-name.onrender.com/health
```

### Model Information
```bash
curl https://your-app-name.onrender.com/model-info
```

### Generate Text
```bash
curl -X POST https://your-app-name.onrender.com/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain machine learning:",
    "max_length": 150,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50
  }'
```

## Local Testing

### Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
# or
uvicorn app:app --reload --port 8000
```

### Test Locally
```bash
# Run full test suite
python test_api.py

# Test with custom prompt
python test_api.py --prompt "Write a poem about AI:"
```

## Cost Considerations

- **Render Standard Plan**: ~$7/month for 512MB RAM, limited to CPU inference
- **Render Pro Plan**: Required for GPU inference (contact Render for pricing)
- **Note**: Qwen 1.5B can run on CPU but will be slower

## Performance Notes

- **Cold Start**: First request after inactivity may take 30-60 seconds (model loading)
- **Warm Requests**: Subsequent requests typically complete in 2-5 seconds
- **GPU vs CPU**: GPU inference is 10-20x faster but requires Pro plan

## Troubleshooting

### Model Loading Fails
- Check `HF_TOKEN_READ` is set correctly in Render environment variables
- Verify model name in `config.yml` matches Hugging Face repository
- Check Render logs for detailed error messages

### Out of Memory
- Reduce `max_length` in generation requests
- Consider upgrading to larger Render instance
- Model uses ~2GB RAM minimum

### Slow Response Times
- Cold starts are normal after inactivity on free/standard plans
- Consider Render's always-on feature or Pro plan
- Reduce generation parameters (max_length, num_return_sequences)

## Security Notes

- Never commit `.env` file with tokens to Git
- Use Render's environment variables for secrets
- API currently has no authentication (add if needed for production)
