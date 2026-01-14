"""
Model loading and inference logic.
"""
import logging
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

def _select_device() -> str:
    """Select the best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class QwenModelHandler:
    """Handler for Qwen model loading and inference"""

    def __init__(self):
        """Initialize model handler"""
        self.model = None
        self.tokenizer = None
        self.device = None

    def load_model(self, model_name: str, hf_token: str):
        """
        Load model and tokenizer from Hugging Face.

        Args:
            model_name: Hugging Face model identifier
            hf_token: Hugging Face API token
        """
        import time
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"🚀 Starting model loading: {model_name}")
        logger.info("=" * 60)

        try:
            # Determine device
            step_start = time.time()
            self.device = _select_device()
            logger.info(f"✓ Device selected: {self.device} ({time.time() - step_start:.2f}s)")

            # Load tokenizer
            step_start = time.time()
            logger.info("📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True
            )
            logger.info(f"✓ Tokenizer loaded (vocab_size={self.tokenizer.vocab_size}, {time.time() - step_start:.2f}s)")

            # Load model
            step_start = time.time()
            logger.info("🧠 Loading model weights (this may take 15-30s)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True,
                torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            ).to(self.device)

            logger.info(f"✓ Model weights loaded ({time.time() - step_start:.2f}s)")

            # Set to evaluation mode
            step_start = time.time()
            logger.info("⚙️  Setting model to evaluation mode...")
            self.model.eval()
            logger.info(f"✓ Model ready ({time.time() - step_start:.2f}s)")

            total_time = time.time() - start_time
            logger.info("=" * 60)
            logger.info(f"✅ Model loading complete! Total time: {total_time:.2f}s")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate text from the model.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to return

        Returns:
            List of generated text strings
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.debug(f"Generating text for prompt: {prompt[:50]}...")

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            # Enforce hard limit on tokens to generate
            input_length = inputs['input_ids'].shape[1]
            tokens_to_generate = min(max_new_tokens, 512)  # Hard limit: 512 tokens
            tokens_to_generate = max(1, tokens_to_generate)  # At least 1 token

            logger.debug(f"Input length: {input_length}, will generate up to {tokens_to_generate} new tokens")

            # Generate with timeout protection
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=tokens_to_generate,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True  # Stop when EOS token is generated
                )

            # Decode outputs
            generated_texts = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]

            logger.debug(f"Generated {len(generated_texts)} sequence(s)")
            return generated_texts

        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise

    def get_model_info(self) -> dict:
        """
        Get model information.

        Returns:
            Dictionary with model metadata
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        return {
            "device": self.device,
            "model_type": self.model.config.model_type if hasattr(self.model, 'config') else "unknown",
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else "unknown"
        }

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.tokenizer is not None

    def cleanup(self):
        """Clean up model resources"""
        logger.info("Cleaning up model resources")
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.device == "cuda":
            torch.cuda.empty_cache()


# Global model handler instance
_model_handler = None


def get_model_handler() -> QwenModelHandler:
    """Get global model handler instance"""
    global _model_handler
    if _model_handler is None:
        _model_handler = QwenModelHandler()
    return _model_handler
