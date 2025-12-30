"""
FastAPI application for serving Hugging Face fine-tuned Qwen 1.5B model.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.model import get_model_handler
from src.config import get_config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and cleanup on shutdown"""
    logger.info("Starting model loading process")

    try:
        # Get configuration and model handler
        config = get_config()
        model_handler = get_model_handler()

        # Load model
        model_handler.load_model(
            model_name=config.model_name,
            hf_token=config.hf_token
        )

        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down and cleaning up resources")
    model_handler = get_model_handler()
    model_handler.cleanup()


app = FastAPI(
    title="Qwen 1.5B Fine-tuned Model API",
    description="API for serving fine-tuned Qwen 1.5B model from Hugging Face",
    version="1.0.0",
    lifespan=lifespan
)


class GenerationRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="Input text prompt for generation")
    max_length: int = Field(100, ge=1, le=2048, description="Maximum length of generated text")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling probability")
    top_k: int = Field(50, ge=0, le=100, description="Top-k sampling parameter")
    num_return_sequences: int = Field(1, ge=1, le=5, description="Number of sequences to return")


class GenerationResponse(BaseModel):
    """Response model for text generation"""
    model_config = {"protected_namespaces": ()}

    generated_texts: list[str]
    model_name: str
    device: str


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Qwen 1.5B Fine-tuned Model API",
        "status": "online",
        "endpoints": {
            "generate": "/generate",
            "health": "/health",
            "model_info": "/model-info"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_handler = get_model_handler()

    if not model_handler.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": model_handler.device
    }


@app.get("/model-info")
async def model_info():
    """Get model information"""
    model_handler = get_model_handler()
    config = get_config()

    if not model_handler.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    info = model_handler.get_model_info()

    return {
        "model_name": config.model_name,
        "device": info["device"],
        "model_type": info["model_type"],
        "vocab_size": info["vocab_size"]
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text from the model"""
    model_handler = get_model_handler()
    config = get_config()

    if not model_handler.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        logger.info(f"Generating text for prompt: {request.prompt[:50]}...")

        # Generate using model handler
        generated_texts = model_handler.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            num_return_sequences=request.num_return_sequences
        )

        logger.info(f"Successfully generated {len(generated_texts)} sequence(s)")

        return GenerationResponse(
            generated_texts=generated_texts,
            model_name=config.model_name,
            device=model_handler.device
        )

    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(app, host="0.0.0.0", port=config.port)
