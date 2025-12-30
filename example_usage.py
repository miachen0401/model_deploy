"""
Example usage script for the deployed Qwen model API.
Demonstrates how to interact with the API programmatically.
"""
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_generation(api_url: str):
    """
    Example 1: Basic text generation with default parameters.

    Args:
        api_url: Base URL of the deployed API
    """
    logger.info("Example 1: Basic Text Generation")

    payload = {
        "prompt": "What is artificial intelligence?",
        "max_length": 100
    }

    response = requests.post(f"{api_url}/generate", json=payload)

    if response.status_code == 200:
        result = response.json()
        logger.info(f"Generated text: {result['generated_texts'][0]}")
    else:
        logger.error(f"Error: {response.status_code} - {response.text}")


def example_custom_parameters(api_url: str):
    """
    Example 2: Generation with custom sampling parameters.

    Args:
        api_url: Base URL of the deployed API
    """
    logger.info("\nExample 2: Custom Sampling Parameters")

    payload = {
        "prompt": "Write a creative story about a robot:",
        "max_length": 200,
        "temperature": 0.9,  # Higher temperature for more creative output
        "top_p": 0.95,
        "top_k": 50,
        "num_return_sequences": 1
    }

    response = requests.post(f"{api_url}/generate", json=payload)

    if response.status_code == 200:
        result = response.json()
        logger.info(f"Model: {result['model_name']}")
        logger.info(f"Device: {result['device']}")
        logger.info(f"Generated text:\n{result['generated_texts'][0]}")
    else:
        logger.error(f"Error: {response.status_code} - {response.text}")


def example_multiple_sequences(api_url: str):
    """
    Example 3: Generate multiple different sequences.

    Args:
        api_url: Base URL of the deployed API
    """
    logger.info("\nExample 3: Multiple Sequence Generation")

    payload = {
        "prompt": "The future of technology is",
        "max_length": 80,
        "temperature": 0.8,
        "num_return_sequences": 3  # Generate 3 different continuations
    }

    response = requests.post(f"{api_url}/generate", json=payload)

    if response.status_code == 200:
        result = response.json()
        logger.info("Generated 3 different continuations:")
        for i, text in enumerate(result['generated_texts'], 1):
            logger.info(f"\nSequence {i}:\n{text}")
    else:
        logger.error(f"Error: {response.status_code} - {response.text}")


def example_conversational(api_url: str):
    """
    Example 4: Multi-turn conversation simulation.

    Args:
        api_url: Base URL of the deployed API
    """
    logger.info("\nExample 4: Conversational Usage")

    conversation_history = []
    prompts = [
        "Hello! Can you help me understand neural networks?",
        "What are the main components of a neural network?",
        "How does backpropagation work?"
    ]

    for user_input in prompts:
        # Build context from conversation history
        if conversation_history:
            context = "\n".join(conversation_history)
            full_prompt = f"{context}\nHuman: {user_input}\nAssistant:"
        else:
            full_prompt = f"Human: {user_input}\nAssistant:"

        payload = {
            "prompt": full_prompt,
            "max_length": 150,
            "temperature": 0.7
        }

        response = requests.post(f"{api_url}/generate", json=payload)

        if response.status_code == 200:
            result = response.json()
            assistant_reply = result['generated_texts'][0]

            logger.info(f"\nHuman: {user_input}")
            logger.info(f"Assistant: {assistant_reply}")

            # Update conversation history
            conversation_history.append(f"Human: {user_input}")
            conversation_history.append(f"Assistant: {assistant_reply}")
        else:
            logger.error(f"Error: {response.status_code} - {response.text}")
            break


def example_check_health(api_url: str):
    """
    Example 5: Check API health before making requests.

    Args:
        api_url: Base URL of the deployed API
    """
    logger.info("\nExample 5: Health Check")

    # Check if API is healthy
    health_response = requests.get(f"{api_url}/health")

    if health_response.status_code == 200:
        health_data = health_response.json()
        logger.info(f"API Status: {health_data['status']}")
        logger.info(f"Model Loaded: {health_data['model_loaded']}")
        logger.info(f"Device: {health_data['device']}")

        # Get model info
        info_response = requests.get(f"{api_url}/model-info")
        if info_response.status_code == 200:
            info_data = info_response.json()
            logger.info(f"Model Name: {info_data['model_name']}")
            logger.info(f"Vocabulary Size: {info_data['vocab_size']}")

        return True
    else:
        logger.error("API is not healthy")
        return False


def main():
    """Run all examples"""
    # Change this to your deployed URL
    # For local testing: "http://localhost:8000"
    # For Render: "https://your-app-name.onrender.com"
    API_URL = "http://localhost:8000"

    logger.info("=" * 60)
    logger.info("Qwen Model API Usage Examples")
    logger.info(f"API URL: {API_URL}")
    logger.info("=" * 60)

    # First check if API is healthy
    if not example_check_health(API_URL):
        logger.error("API is not available. Please start the server first.")
        return

    # Run examples
    try:
        example_basic_generation(API_URL)
        example_custom_parameters(API_URL)
        example_multiple_sequences(API_URL)
        example_conversational(API_URL)

        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully!")
        logger.info("=" * 60)

    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to API at {API_URL}")
        logger.error("Please ensure the server is running.")
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")


if __name__ == "__main__":
    main()
