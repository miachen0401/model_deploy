"""
Example data analyzer template.
This is a template for future data analysis scripts.
"""
import logging
from typing import List, Dict, Any

from src.config import get_config
from src.model import get_model_handler

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """Base class for data analysis tasks"""

    def __init__(self):
        """Initialize the analyzer"""
        self.config = get_config()
        self.model_handler = None

    def load_model_if_needed(self):
        """Load model if needed for analysis"""
        if self.model_handler is None:
            logger.info("Loading model for analysis...")
            self.model_handler = get_model_handler()
            if not self.model_handler.is_loaded():
                self.model_handler.load_model(
                    model_name=self.config.model_name,
                    hf_token=self.config.hf_token
                )
            logger.info("Model loaded successfully")

    def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Perform analysis on data.

        Args:
            data: Input data to analyze

        Returns:
            Analysis results
        """
        raise NotImplementedError("Subclasses must implement analyze()")

    def cleanup(self):
        """Clean up resources"""
        if self.model_handler:
            self.model_handler.cleanup()


class ExampleTextAnalyzer(DataAnalyzer):
    """Example: Analyze text generation performance"""

    def analyze_prompts(self, prompts: List[str]) -> Dict[str, Any]:
        """
        Analyze multiple prompts and generate responses.

        Args:
            prompts: List of text prompts

        Returns:
            Analysis results with generated texts
        """
        self.load_model_if_needed()

        results = []
        for prompt in prompts:
            logger.info(f"Analyzing prompt: {prompt[:50]}...")

            generated = self.model_handler.generate(
                prompt=prompt,
                max_length=100,
                temperature=0.7
            )

            results.append({
                "prompt": prompt,
                "generated": generated[0],
                "length": len(generated[0])
            })

        return {
            "total_prompts": len(prompts),
            "results": results,
            "average_length": sum(r["length"] for r in results) / len(results)
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    analyzer = ExampleTextAnalyzer()

    test_prompts = [
        "What is machine learning?",
        "Explain neural networks:",
        "The future of AI is"
    ]

    results = analyzer.analyze_prompts(test_prompts)

    logger.info(f"Analysis complete: {results['total_prompts']} prompts processed")
    logger.info(f"Average response length: {results['average_length']:.2f} characters")

    analyzer.cleanup()
