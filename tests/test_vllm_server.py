"""
Pytest tests for vLLM OpenAI-compatible server.
Tests health, basic generation, streaming, and various parameters.
"""
import pytest
import time
from openai import OpenAI


class TestVLLMServerHealth:
    """Test server health and availability."""

    def test_server_is_running(self, openai_client: OpenAI):
        """Test that the vLLM server is running and responding."""
        models = openai_client.models.list()
        assert models is not None
        assert len(models.data) > 0

    def test_model_is_loaded(self, openai_client: OpenAI):
        """Test that the model is loaded and available."""
        models = openai_client.models.list()
        model_ids = [model.id for model in models.data]
        assert "model" in model_ids


class TestBasicGeneration:
    """Test basic text generation functionality."""

    def test_simple_completion(self, openai_client: OpenAI):
        """Test simple text completion."""
        response = openai_client.completions.create(
            model="model",
            prompt="Hello, I am",
            max_tokens=10,
            temperature=0.7,
        )

        assert response is not None
        assert len(response.choices) == 1
        assert response.choices[0].text is not None
        assert len(response.choices[0].text) > 0
        assert response.usage.completion_tokens > 0

    def test_longer_completion(self, openai_client: OpenAI):
        """Test longer text completion."""
        response = openai_client.completions.create(
            model="model",
            prompt="Explain artificial intelligence:",
            max_tokens=100,
            temperature=0.7,
        )

        assert response is not None
        assert response.choices[0].text is not None
        assert response.usage.completion_tokens <= 100
        assert response.usage.completion_tokens > 10  # Should generate some text

    def test_multiple_prompts(self, openai_client: OpenAI, sample_prompts: list[str]):
        """Test generation with multiple different prompts."""
        for prompt in sample_prompts[:3]:  # Test first 3 prompts
            response = openai_client.completions.create(
                model="model",
                prompt=prompt,
                max_tokens=50,
                temperature=0.7,
            )

            assert response is not None
            assert len(response.choices[0].text) > 0


class TestSamplingParameters:
    """Test various sampling parameters."""

    def test_temperature_variation(self, openai_client: OpenAI):
        """Test different temperature values."""
        prompt = "The future of AI is"
        temperatures = [0.1, 0.7, 1.0]

        for temp in temperatures:
            response = openai_client.completions.create(
                model="model",
                prompt=prompt,
                max_tokens=50,
                temperature=temp,
            )

            assert response is not None
            assert len(response.choices[0].text) > 0

    def test_max_tokens_limits(self, openai_client: OpenAI):
        """Test different max_tokens values."""
        prompt = "Write a story:"
        max_tokens_list = [10, 50, 100, 200]

        for max_tokens in max_tokens_list:
            response = openai_client.completions.create(
                model="model",
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7,
            )

            assert response is not None
            assert response.usage.completion_tokens <= max_tokens

    def test_top_p_parameter(self, openai_client: OpenAI):
        """Test top_p (nucleus sampling) parameter."""
        response = openai_client.completions.create(
            model="model",
            prompt="Machine learning enables",
            max_tokens=50,
            temperature=0.8,
            top_p=0.95,
        )

        assert response is not None
        assert len(response.choices[0].text) > 0


class TestStreamingGeneration:
    """Test streaming text generation."""

    def test_streaming_completion(self, openai_client: OpenAI):
        """Test streaming text completion."""
        stream = openai_client.completions.create(
            model="model",
            prompt="Artificial intelligence is",
            max_tokens=50,
            temperature=0.7,
            stream=True,
        )

        chunks = list(stream)
        assert len(chunks) > 0

        # Concatenate streamed text
        full_text = "".join(
            chunk.choices[0].text
            for chunk in chunks
            if chunk.choices and chunk.choices[0].text
        )
        assert len(full_text) > 0

    def test_streaming_with_different_prompts(
        self, openai_client: OpenAI, sample_prompts: list[str]
    ):
        """Test streaming with various prompts."""
        for prompt in sample_prompts[:2]:  # Test first 2 prompts
            stream = openai_client.completions.create(
                model="model",
                prompt=prompt,
                max_tokens=30,
                temperature=0.7,
                stream=True,
            )

            chunks = list(stream)
            assert len(chunks) > 0


class TestPerformance:
    """Test performance and throughput."""

    def test_response_time(self, openai_client: OpenAI):
        """Test that response time is reasonable."""
        start_time = time.time()

        response = openai_client.completions.create(
            model="model",
            prompt="What is machine learning?",
            max_tokens=100,
            temperature=0.7,
        )

        elapsed = time.time() - start_time

        assert response is not None
        # Should complete within 5 seconds for 100 tokens on RTX 4080
        assert elapsed < 5.0, f"Response took {elapsed:.2f}s, expected < 5s"

    def test_throughput(
        self, openai_client: OpenAI, performance_test_config: dict
    ):
        """Test throughput (tokens per second)."""
        # Warmup
        for _ in range(performance_test_config["warmup_requests"]):
            openai_client.completions.create(
                model="model",
                prompt="Test warmup",
                max_tokens=10,
                temperature=0.7,
            )

        # Measure throughput
        total_tokens = 0
        start_time = time.time()

        for _ in range(performance_test_config["test_requests"]):
            response = openai_client.completions.create(
                model="model",
                prompt="Artificial intelligence is",
                max_tokens=performance_test_config["max_tokens"],
                temperature=performance_test_config["temperature"],
            )
            total_tokens += response.usage.completion_tokens

        elapsed = time.time() - start_time
        throughput = total_tokens / elapsed

        # vLLM should achieve > 50 tokens/sec on RTX 4080
        assert throughput > 50, f"Throughput {throughput:.2f} tokens/sec is too low"

        print(f"\nThroughput: {throughput:.2f} tokens/sec")
        print(f"Total tokens: {total_tokens}")
        print(f"Total time: {elapsed:.2f}s")


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_model(self, openai_client: OpenAI):
        """Test requesting an invalid model."""
        with pytest.raises(Exception):
            openai_client.completions.create(
                model="nonexistent-model",
                prompt="Test",
                max_tokens=10,
            )

    def test_empty_prompt(self, openai_client: OpenAI):
        """Test generation with empty prompt - should be rejected."""
        # vLLM correctly rejects empty prompts
        with pytest.raises(Exception) as exc_info:
            openai_client.completions.create(
                model="model",
                prompt="",
                max_tokens=10,
                temperature=0.7,
            )

        # Verify it's a bad request error
        assert "400" in str(exc_info.value) or "BadRequest" in str(exc_info.value)

    def test_very_long_prompt(self, openai_client: OpenAI):
        """Test generation with very long prompt (within limits)."""
        # Create a long prompt that's challenging but within limits
        # Each repetition is ~4 tokens, so 200 repetitions = ~800 tokens
        # Plus 50 max_tokens for generation = ~850 total (well within 4096)
        long_prompt = "This is a test sentence. " * 200

        response = openai_client.completions.create(
            model="model",
            prompt=long_prompt,
            max_tokens=50,
            temperature=0.7,
        )

        assert response is not None
        assert response.usage.completion_tokens > 0
        # Prompt should be substantial
        assert response.usage.prompt_tokens > 500

    def test_exceeding_context_window(self, openai_client: OpenAI):
        """Test that requests exceeding context window are handled properly."""
        # Create a prompt that will definitely exceed the 4096 token context
        # Each repetition is ~4 tokens, 1200 repetitions = ~4800 tokens (exceeds 4096)
        extremely_long_prompt = "This is a very long test sentence that will help us exceed the context window. " * 1200

        # This should either truncate or return an error
        try:
            response = openai_client.completions.create(
                model="model",
                prompt=extremely_long_prompt,
                max_tokens=50,
                temperature=0.7,
            )
            # If it succeeds, the prompt should have been truncated
            assert response is not None
            # Prompt tokens should be less than max_model_len (4096)
            assert response.usage.prompt_tokens < 4096
        except Exception as e:
            # Or it might return an error, which is also acceptable
            assert "400" in str(e) or "context" in str(e).lower() or "length" in str(e).lower()


class TestUsageMetrics:
    """Test usage metrics and token counting."""

    def test_token_counting(self, openai_client: OpenAI):
        """Test that token counts are accurate."""
        response = openai_client.completions.create(
            model="model",
            prompt="Count tokens:",
            max_tokens=50,
            temperature=0.7,
        )

        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )

    def test_usage_tracking(self, openai_client: OpenAI):
        """Test that usage is tracked across multiple requests."""
        total_tokens = 0

        for _ in range(5):
            response = openai_client.completions.create(
                model="model",
                prompt="Test",
                max_tokens=10,
                temperature=0.7,
            )
            total_tokens += response.usage.total_tokens

        assert total_tokens > 0
        print(f"\nTotal tokens across 5 requests: {total_tokens}")
