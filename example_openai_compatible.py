"""
Example usage of vLLM's OpenAI-compatible API server.
This is the recommended production deployment method.
"""
import logging
import time
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_completion(client: OpenAI):
    """Example 1: Text completion"""
    logger.info("Example 1: Text Completion")

    start_time = time.time()
    completion = client.completions.create(
        model="model",
        prompt="What is artificial intelligence?",
        max_tokens=100,
        temperature=0.7,
    )
    elapsed = time.time() - start_time

    logger.info(f"Generated text: {completion.choices[0].text}")
    logger.info(f"Time taken: {elapsed:.2f}s")
    logger.info(f"Tokens: {completion.usage.completion_tokens}")
    logger.info(f"Throughput: {completion.usage.completion_tokens/elapsed:.2f} tokens/sec")


def example_streaming_completion(client: OpenAI):
    """Example 2: Streaming completion"""
    logger.info("\nExample 2: Streaming Completion")
    logger.info("-" * 60)

    stream = client.completions.create(
        model="model",
        prompt="Write a short story about a robot learning to paint:",
        max_tokens=200,
        temperature=0.8,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].text:
            print(chunk.choices[0].text, end='', flush=True)

    print()
    logger.info("-" * 60)


def example_chat_completion(client: OpenAI):
    """Example 3: Chat completion (if model supports it)"""
    logger.info("\nExample 3: Chat Completion")

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model="model",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum computing briefly."}
            ],
            max_tokens=150,
            temperature=0.7,
        )
        elapsed = time.time() - start_time

        logger.info(f"Assistant: {response.choices[0].message.content}")
        logger.info(f"Time taken: {elapsed:.2f}s")
        logger.info(f"Tokens: {response.usage.completion_tokens}")
    except Exception as e:
        logger.warning(f"Chat completion not supported or failed: {e}")


def example_batch_completion(client: OpenAI):
    """Example 4: Batch completions"""
    logger.info("\nExample 4: Batch Completions")

    prompts = [
        "The future of AI will be",
        "Machine learning enables",
        "Neural networks are"
    ]

    start_time = time.time()
    for i, prompt in enumerate(prompts, 1):
        completion = client.completions.create(
            model="model",
            prompt=prompt,
            max_tokens=50,
            temperature=0.8,
        )
        logger.info(f"\nPrompt {i}: {prompt}")
        logger.info(f"Completion: {completion.choices[0].text}")

    elapsed = time.time() - start_time
    logger.info(f"\nTotal time for {len(prompts)} requests: {elapsed:.2f}s")
    logger.info(f"Average per request: {elapsed/len(prompts):.2f}s")


def example_models_list(client: OpenAI):
    """Example 5: List available models"""
    logger.info("\nExample 5: List Models")

    models = client.models.list()
    logger.info("Available models:")
    for model in models.data:
        logger.info(f"  - {model.id}")


def benchmark_performance(client: OpenAI):
    """Bonus: Performance benchmark"""
    logger.info("\nBonus: Performance Benchmark")
    logger.info("=" * 60)

    times = []
    throughputs = []

    for i in range(5):
        start_time = time.time()
        completion = client.completions.create(
            model="model",
            prompt="Artificial intelligence is",
            max_tokens=100,
            temperature=0.7,
        )
        elapsed = time.time() - start_time

        tokens = completion.usage.completion_tokens
        throughput = tokens / elapsed
        times.append(elapsed)
        throughputs.append(throughput)

        logger.info(f"Run {i+1}: {elapsed:.2f}s, {throughput:.2f} tokens/sec")

    avg_time = sum(times) / len(times)
    avg_throughput = sum(throughputs) / len(throughputs)
    logger.info(f"\nAverage: {avg_time:.2f}s, {avg_throughput:.2f} tokens/sec")


def main():
    """Run all examples"""
    API_URL = "http://localhost:8000/v1"

    logger.info("=" * 60)
    logger.info("vLLM OpenAI-Compatible API Examples")
    logger.info(f"API URL: {API_URL}")
    logger.info("=" * 60)

    # Initialize OpenAI client pointing to vLLM server
    client = OpenAI(
        api_key="EMPTY",  # vLLM doesn't require API key by default
        base_url=API_URL,
    )

    try:
        # Check if server is available
        example_models_list(client)

        # Run examples
        example_completion(client)
        example_streaming_completion(client)
        example_chat_completion(client)
        example_batch_completion(client)
        benchmark_performance(client)

        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully!")
        logger.info("\nKey advantages of vLLM OpenAI-compatible server:")
        logger.info("  ✓ Drop-in replacement for OpenAI API")
        logger.info("  ✓ Full streaming support")
        logger.info("  ✓ Batch processing")
        logger.info("  ✓ Production-ready with monitoring")
        logger.info("  ✓ 10-20x faster than standard inference")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.info("\nMake sure the vLLM server is running:")
        logger.info("  bash start_vllm_server.sh")


if __name__ == "__main__":
    main()
