"""
Test script for the Qwen model API deployment.
Tests both local and deployed versions of the API.
"""
import logging
import requests
import json
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelAPITester:
    """Test client for the model API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the tester.

        Args:
            base_url: Base URL of the API (e.g., 'http://localhost:8000' or 'https://your-app.onrender.com')
        """
        self.base_url = base_url.rstrip('/')
        logger.info(f"Initialized tester with base URL: {self.base_url}")

    def test_root(self) -> bool:
        """Test the root endpoint"""
        logger.info("Testing root endpoint...")
        try:
            response = requests.get(f"{self.base_url}/")
            response.raise_for_status()
            data = response.json()
            logger.info(f"Root endpoint response: {json.dumps(data, indent=2)}")
            return True
        except Exception as e:
            logger.error(f"Root endpoint test failed: {str(e)}")
            return False

    def test_health(self) -> bool:
        """Test the health check endpoint"""
        logger.info("Testing health endpoint...")
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            data = response.json()
            logger.info(f"Health check response: {json.dumps(data, indent=2)}")
            return data.get('status') == 'healthy'
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False

    def test_model_info(self) -> bool:
        """Test the model info endpoint"""
        logger.info("Testing model info endpoint...")
        try:
            response = requests.get(f"{self.base_url}/model-info")
            response.raise_for_status()
            data = response.json()
            logger.info(f"Model info response: {json.dumps(data, indent=2)}")
            return True
        except Exception as e:
            logger.error(f"Model info test failed: {str(e)}")
            return False

    def test_generation(
        self,
        prompt: str = "Hello, I am a language model",
        max_length: int = 100,
        temperature: float = 0.7
    ) -> Optional[dict]:
        """
        Test the text generation endpoint.

        Args:
            prompt: Input prompt for generation
            max_length: Maximum length of generated text
            temperature: Sampling temperature

        Returns:
            Generated response or None if failed
        """
        logger.info(f"Testing generation with prompt: '{prompt}'")

        payload = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 50,
            "num_return_sequences": 1
        }

        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            data = response.json()

            logger.info(f"Generation successful!")
            logger.info(f"Model: {data.get('model_name')}")
            logger.info(f"Device: {data.get('device')}")
            logger.info(f"Generated texts:")
            for i, text in enumerate(data.get('generated_texts', []), 1):
                logger.info(f"  [{i}] {text}")

            return data
        except Exception as e:
            logger.error(f"Generation test failed: {str(e)}")
            return None

    def run_all_tests(self):
        """Run all test cases"""
        logger.info("=" * 60)
        logger.info("Starting API Test Suite")
        logger.info("=" * 60)

        results = {
            "root": self.test_root(),
            "health": self.test_health(),
            "model_info": self.test_model_info(),
        }

        logger.info("\n" + "=" * 60)
        logger.info("Testing Text Generation")
        logger.info("=" * 60)

        # Test with different prompts
        test_prompts = [
            "Explain what machine learning is in simple terms:",
            "Write a short poem about AI:",
            "What is the capital of France?",
        ]

        for prompt in test_prompts:
            result = self.test_generation(prompt=prompt, max_length=150)
            results[f"generation_{prompt[:30]}"] = result is not None

        logger.info("\n" + "=" * 60)
        logger.info("Test Results Summary")
        logger.info("=" * 60)
        for test_name, passed in results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            logger.info(f"{test_name}: {status}")

        all_passed = all(results.values())
        logger.info("=" * 60)
        if all_passed:
            logger.info("All tests PASSED!")
        else:
            logger.info("Some tests FAILED. Please check the logs above.")
        logger.info("=" * 60)

        return all_passed


def main():
    """Main function to run tests"""
    import argparse

    parser = argparse.ArgumentParser(description="Test the Qwen model API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Custom prompt for generation test"
    )

    args = parser.parse_args()

    tester = ModelAPITester(base_url=args.url)

    if args.prompt:
        logger.info("Running single generation test with custom prompt")
        tester.test_generation(prompt=args.prompt)
    else:
        logger.info("Running full test suite")
        tester.run_all_tests()


if __name__ == "__main__":
    main()
