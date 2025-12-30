"""
Model loading and inference logic.
"""
import logging
from typing import Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


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
        logger.info(f"Loading model: {model_name}")

        try:
            # Determine device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True
            )
            logger.info("Tokenizer loaded successfully")

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)

            self.model.eval()
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate text from the model.

        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
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

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
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
