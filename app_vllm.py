"""
FastAPI application for serving Hugging Face fine-tuned Qwen model using vLLM.
vLLM provides high-performance inference with PagedAttention and continuous batching.
"""
import logging
import multiprocessing
from contextlib import asynccontextmanager
from typing import List, Optional

# IMPORTANT: Set spawn method BEFORE any CUDA/torch imports
multiprocessing.set_start_method('spawn', force=True)

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from vllm import LLM
from vllm.sampling_params import SamplingParams

from src.config import get_config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global engine instance
llm: Optional[LLM] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load vLLM engine on startup and cleanup on shutdown"""
    global llm

    logger.info("\n" + "=" * 60)
    logger.info("🌟 vLLM Application Startup - Initializing...")
    logger.info("=" * 60)

    try:
        config = get_config()
        logger.info("📋 Loading configuration...")
        logger.info(f"   Model: {config.model_name}")
        logger.info(f"   Port: {config.port}")

        logger.info("🚀 Initializing vLLM Engine...")
        logger.info("   - GPU Memory Utilization: 90%")
        logger.info("   - Max Model Length: 4096 tokens")
        logger.info("   - Data Type: bfloat16")

        # Initialize vLLM with optimized settings for RTX 4080
        llm = LLM(
            model=config.model_name,
            tokenizer=config.model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            enforce_eager=False,
            tensor_parallel_size=1,
        )

        logger.info("\n🎉 vLLM engine ready to accept requests!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to initialize vLLM engine: {str(e)}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down vLLM engine")
    # vLLM engine cleanup is handled automatically


app = FastAPI(
    title="Qwen Model API with vLLM",
    description="High-performance API for serving fine-tuned Qwen model using vLLM",
    version="2.0.0",
    lifespan=lifespan
)


class GenerationRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="Input text prompt for generation")
    max_new_tokens: int = Field(
        64,
        ge=1,
        le=2048,
        description="Maximum new tokens to generate"
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling probability")
    top_k: int = Field(50, ge=-1, le=100, description="Top-k sampling parameter")
    num_return_sequences: int = Field(1, ge=1, le=5, description="Number of sequences to return")
    stream: bool = Field(False, description="Enable streaming response")
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0, description="Repetition penalty")


class GenerationResponse(BaseModel):
    """Response model for text generation"""
    model_config = {"protected_namespaces": ()}

    generated_texts: List[str]
    model_name: str
    device: str = "cuda"
    num_tokens: Optional[List[int]] = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Qwen Model API with vLLM",
        "status": "online",
        "engine": "vLLM",
        "endpoints": {
            "generate": "/generate",
            "health": "/health",
            "model_info": "/model-info"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if llm is None:
        raise HTTPException(status_code=503, detail="vLLM engine not initialized")

    return {
        "status": "healthy",
        "engine": "vLLM",
        "model_loaded": True,
        "device": "cuda"
    }


@app.get("/model-info")
async def model_info():
    """Get model information"""
    config = get_config()

    if llm is None:
        raise HTTPException(status_code=503, detail="vLLM engine not initialized")

    return {
        "model_name": config.model_name,
        "device": "cuda",
        "engine": "vLLM",
        "max_model_len": 4096,
        "dtype": "bfloat16"
    }


@app.post("/generate")
async def generate_text(request: GenerationRequest):
    """Generate text from the model using vLLM"""
    import asyncio

    if llm is None:
        raise HTTPException(status_code=503, detail="vLLM engine not initialized")

    config = get_config()

    try:
        # Enforce max_new_tokens limit from config
        max_new_tokens = min(request.max_new_tokens, config.max_new_tokens)
        if max_new_tokens != request.max_new_tokens:
            logger.info(f"Capping max_new_tokens from {request.max_new_tokens} to {max_new_tokens}")

        logger.info(f"Generating text for prompt: {request.prompt[:50]}... (max_new_tokens={max_new_tokens})")

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=max_new_tokens,
            n=request.num_return_sequences,
            repetition_penalty=request.repetition_penalty,
        )

        # Streaming not supported in synchronous mode
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming not supported with synchronous LLM. Use vLLM's OpenAI-compatible server for streaming."
            )

        # Run synchronous generate in thread pool
        outputs = await asyncio.to_thread(
            llm.generate,
            [request.prompt],
            sampling_params
        )

        # Extract generated texts and token counts
        results = []
        num_tokens = []

        for output in outputs:
            for completion in output.outputs:
                results.append(completion.text)
                num_tokens.append(len(completion.token_ids))

        logger.info(f"Successfully generated {len(results)} sequence(s)")

        return GenerationResponse(
            generated_texts=results,
            model_name=config.model_name,
            device="cuda",
            num_tokens=num_tokens
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    config = get_config()

    logger.info("\n" + "=" * 60)
    logger.info("Starting vLLM-powered Qwen Model API")
    logger.info("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.port,
        log_level="info"
    )
