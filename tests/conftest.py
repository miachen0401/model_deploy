"""
Pytest configuration and fixtures for vLLM server tests.
"""
import pytest
import time
import requests
from typing import Generator
from openai import OpenAI


@pytest.fixture(scope="session")
def vllm_base_url() -> str:
    """Base URL for vLLM OpenAI-compatible server."""
    return "http://localhost:8000/v1"


@pytest.fixture(scope="session")
def custom_base_url() -> str:
    """Base URL for custom FastAPI server."""
    return "http://localhost:8000"


@pytest.fixture(scope="session")
def openai_client(vllm_base_url: str) -> Generator[OpenAI, None, None]:
    """OpenAI client configured for vLLM server."""
    client = OpenAI(
        api_key="EMPTY",
        base_url=vllm_base_url,
    )

    # Wait for server to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            client.models.list()
            break
        except Exception:
            if i == max_retries - 1:
                pytest.skip("vLLM server not available")
            time.sleep(1)

    yield client


@pytest.fixture(scope="session")
def custom_api_available(custom_base_url: str) -> bool:
    """Check if custom FastAPI server is available."""
    try:
        response = requests.get(f"{custom_base_url}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


@pytest.fixture
def sample_prompts() -> list[str]:
    """Sample prompts for testing."""
    return [
        "What is artificial intelligence?",
        "Explain machine learning in simple terms:",
        "Write a haiku about technology:",
        "The future of computing is",
        "List three benefits of renewable energy:",
    ]


@pytest.fixture
def performance_test_config() -> dict:
    """Configuration for performance tests."""
    return {
        "warmup_requests": 3,
        "test_requests": 10,
        "max_tokens": 100,
        "temperature": 0.7,
    }


@pytest.fixture
def concurrency_test_config() -> dict:
    """Configuration for concurrency tests."""
    return {
        "concurrent_requests": [1, 5, 10, 20, 50],
        "max_tokens": 50,
        "temperature": 0.7,
        "timeout": 60,
    }
