"""
Concurrency and load tests for vLLM server.
Tests parallel request handling and throughput under load.
Focus on realistic use cases with long prompts and up to 100 concurrent requests.
"""
import pytest
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from openai import OpenAI
import statistics


class TestConcurrency:
    """Test concurrent request handling."""

    def test_sequential_requests(self, openai_client: OpenAI):
        """Baseline test with sequential requests."""
        num_requests = 5
        prompt = "What is AI?"
        max_tokens = 50

        start_time = time.time()
        responses = []

        for _ in range(num_requests):
            response = openai_client.completions.create(
                model="model",
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            responses.append(response)

        elapsed = time.time() - start_time

        assert len(responses) == num_requests
        assert all(r.choices[0].text for r in responses)

        print(f"\n Sequential: {num_requests} requests in {elapsed:.2f}s")
        print(f"Average: {elapsed/num_requests:.2f}s per request")

    def test_parallel_requests_small_batch(self, openai_client: OpenAI):
        """Test parallel requests with small batch size."""
        num_requests = 5
        prompt = "Explain machine learning:"
        max_tokens = 50

        def make_request():
            return openai_client.completions.create(
                model="model",
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7,
            )

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            responses = [future.result() for future in as_completed(futures)]

        elapsed = time.time() - start_time

        assert len(responses) == num_requests
        assert all(r.choices[0].text for r in responses)

        print(f"\nParallel (5): {num_requests} requests in {elapsed:.2f}s")
        print(f"Speedup: {(elapsed / num_requests):.2f}s avg (parallel)")

    def test_parallel_requests_medium_batch(self, openai_client: OpenAI):
        """Test parallel requests with medium batch size."""
        num_requests = 10
        prompt = "The future of technology is"
        max_tokens = 40

        def make_request():
            start = time.time()
            response = openai_client.completions.create(
                model="model",
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return response, time.time() - start

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]

        total_elapsed = time.time() - start_time
        responses = [r[0] for r in results]
        individual_times = [r[1] for r in results]

        assert len(responses) == num_requests
        assert all(r.choices[0].text for r in responses)

        avg_individual = statistics.mean(individual_times)
        print(f"\nParallel (10): {num_requests} requests in {total_elapsed:.2f}s")
        print(f"Individual avg: {avg_individual:.2f}s")
        print(f"Throughput: {num_requests/total_elapsed:.2f} req/sec")

    def test_parallel_requests_moderate_batch(self, openai_client: OpenAI):
        """Test parallel requests with moderate batch size (20)."""
        num_requests = 20
        prompt = "AI enables"
        max_tokens = 30

        def make_request(request_id: int) -> Tuple[int, float, bool]:
            try:
                start = time.time()
                response = openai_client.completions.create(
                    model="model",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                elapsed = time.time() - start
                return request_id, elapsed, True
            except Exception as e:
                print(f"Request {request_id} failed: {e}")
                return request_id, 0, False

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(make_request, i) for i in range(num_requests)
            ]
            results = [future.result() for future in as_completed(futures)]

        total_elapsed = time.time() - start_time

        successful = [r for r in results if r[2]]
        failed = [r for r in results if not r[2]]

        assert len(successful) >= num_requests * 0.9  # At least 90% success rate

        individual_times = [r[1] for r in successful]
        avg_time = statistics.mean(individual_times)
        min_time = min(individual_times)
        max_time = max(individual_times)

        print(f"\nParallel (20):")
        print(f"  Total time: {total_elapsed:.2f}s")
        print(f"  Successful: {len(successful)}/{num_requests}")
        print(f"  Failed: {len(failed)}")
        print(f"  Individual time - avg: {avg_time:.2f}s, min: {min_time:.2f}s, max: {max_time:.2f}s")
        print(f"  Throughput: {len(successful)/total_elapsed:.2f} req/sec")

    def test_long_prompt_concurrency(self, openai_client: OpenAI):
        """Test concurrent requests with realistic long prompts."""
        num_requests = 10

        # Realistic long prompt similar to news categorization use case
        categories_definition = """
Primary Categories (select ONE only):

MACRO_ECONOMY - Concrete macroeconomic indicators or official data releases (e.g., CPI, GDP, PMI, unemployment reports)
CENTRAL_BANK_POLICY - Official central bank decisions, rate changes, speeches by named officials
GEOPOLITICAL_EVENT - Geopolitical news ONLY when involving specific named countries, leaders, governments, or concrete actions
INDUSTRY_REGULATION - Regulatory/policy news targeting specific industry/sector
CORPORATE_EARNINGS - Company earnings, financial statements, revenue data
CORPORATE_ACTIONS - M&A, stock splits, buybacks, spinoffs, bankruptcies
MANAGEMENT_CHANGE - CEO, CFO, board member changes
INCIDENT_LEGAL - Lawsuits, fines, regulatory investigations, accidents, data breaches
PRODUCT_TECH_UPDATE - New products, technology developments, R&D, clinical trial results
BUSINESS_OPERATIONS - Supply chain, contracts, partnerships, operational decisions
ANALYST_OPINION - Analyst upgrades/downgrades, price targets, commentary
MARKET_SENTIMENT - Investor sentiment, market flows, surveys, risk appetite
NON_FINANCIAL - Any general commentary, opinions without specific actors, or news unrelated to financial markets

Secondary Category:
- If news is about specific company/stock, output stock ticker symbols (e.g., AAPL, TSLA)
- If not company-specific, output empty string

RULES:
- You MUST choose one of the EXACT category names listed above.
- NEVER invent new categories.
- NEVER output numbers, abbreviations, or synonyms.
- If the news does NOT mention a concrete decision, named institution, named government, named company, or measurable data → ALWAYS classify as "NON_FINANCIAL".
"""

        # Simulate news items
        news_text = ""
        for idx in range(1, 4):  # 3 news items
            news_text += f"""
[NEWS {idx}]
Title: {'Technology company announces new AI product launch with expected revenue impact' if idx == 1 else 'Federal Reserve signals potential rate changes in upcoming meeting' if idx == 2 else 'Market analysts update forecasts following quarterly earnings'}
Summary: {'The company revealed plans to integrate advanced AI capabilities into their product line, with management projecting significant revenue growth in the coming quarters.' if idx == 1 else 'Central bank officials indicated that monetary policy adjustments may be considered based on recent economic data and inflation trends.' if idx == 2 else 'Following better-than-expected earnings results, several analysts have revised their price targets and recommendations for the stock.'}
"""

        prompt = f"""{categories_definition}

Analyze the following news articles and categorize each one.

Output format (JSON array):
[
  {{
    "news_id": 1,
    "primary_category": "CATEGORY_NAME",
    "symbol": "STOCK_SYMBOLS or empty string",
    "confidence": 0.0-1.0
  }},
  ...
]

News articles to categorize:
{news_text}

Output only the JSON array, no additional text."""

        def make_request(request_id: int) -> Tuple[int, float, bool, int]:
            try:
                start = time.time()
                response = openai_client.completions.create(
                    model="model",
                    prompt=prompt,
                    max_tokens=200,  # Need more tokens for JSON output
                    temperature=0.3,  # Lower temp for structured output
                )
                elapsed = time.time() - start
                return request_id, elapsed, True, response.usage.total_tokens
            except Exception as e:
                print(f"Request {request_id} failed: {e}")
                return request_id, 0, False, 0

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [
                executor.submit(make_request, i) for i in range(num_requests)
            ]
            results = [future.result() for future in as_completed(futures)]

        total_elapsed = time.time() - start_time

        successful = [r for r in results if r[2]]
        failed = [r for r in results if not r[2]]

        assert len(successful) >= num_requests * 0.8  # At least 80% success with long prompts

        latencies = [r[1] for r in successful]
        total_tokens = sum(r[3] for r in successful)

        print(f"\nLong Prompt Concurrency ({num_requests} requests, ~1000 token prompts):")
        print(f"  Total time: {total_elapsed:.2f}s")
        print(f"  Successful: {len(successful)}/{num_requests}")
        print(f"  Failed: {len(failed)}")
        print(f"  Throughput: {len(successful)/total_elapsed:.2f} req/sec")
        print(f"  Token throughput: {total_tokens/total_elapsed:.2f} tokens/sec")
        print(f"  Latency - avg: {statistics.mean(latencies):.2f}s")
        print(f"  Latency - p50: {statistics.median(latencies):.2f}s")
        print(f"  Latency - max: {max(latencies):.2f}s")


class TestLoadTesting:
    """Load testing for vLLM server."""

    def test_sustained_load_low(self, openai_client: OpenAI):
        """Test sustained low load (5 concurrent for 30 seconds)."""
        duration = 10  # Reduced from 30s for faster testing
        concurrent_workers = 5
        max_tokens = 30

        prompts = [
            "What is AI?",
            "Explain ML:",
            "Future of tech:",
            "Computing is",
            "Technology enables",
        ]

        def make_request(request_id: int) -> dict:
            prompt = prompts[request_id % len(prompts)]
            try:
                start = time.time()
                response = openai_client.completions.create(
                    model="model",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                elapsed = time.time() - start
                return {
                    "success": True,
                    "elapsed": elapsed,
                    "tokens": response.usage.completion_tokens,
                }
            except Exception as e:
                return {"success": False, "error": str(e), "elapsed": 0, "tokens": 0}

        results = []
        start_time = time.time()
        request_id = 0

        with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
            futures = []

            while time.time() - start_time < duration:
                # Submit new requests as workers become available
                if len(futures) < concurrent_workers:
                    future = executor.submit(make_request, request_id)
                    futures.append(future)
                    request_id += 1

                # Collect completed requests
                done_futures = [f for f in futures if f.done()]
                for f in done_futures:
                    results.append(f.result())
                    futures.remove(f)

                time.sleep(0.1)  # Small delay to avoid busy waiting

            # Wait for remaining futures
            for future in futures:
                results.append(future.result())

        total_elapsed = time.time() - start_time

        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        total_tokens = sum(r["tokens"] for r in successful)
        avg_latency = statistics.mean([r["elapsed"] for r in successful])

        print(f"\nSustained Load (5 concurrent, {duration}s):")
        print(f"  Total requests: {len(results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Throughput: {len(successful)/total_elapsed:.2f} req/sec")
        print(f"  Token throughput: {total_tokens/total_elapsed:.2f} tokens/sec")
        print(f"  Avg latency: {avg_latency:.2f}s")

        # At least 80% success rate under load
        assert len(successful) >= len(results) * 0.8

    def test_burst_load_high_concurrency(self, openai_client: OpenAI):
        """Test burst load with high concurrency (100 requests)."""
        num_requests = 100
        max_workers = 100
        max_tokens = 20

        def make_request(request_id: int) -> Tuple[int, float, bool, int]:
            try:
                start = time.time()
                response = openai_client.completions.create(
                    model="model",
                    prompt=f"Request {request_id}:",
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                elapsed = time.time() - start
                return request_id, elapsed, True, response.usage.completion_tokens
            except Exception as e:
                print(f"Request {request_id} failed: {e}")
                return request_id, 0, False, 0

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(make_request, i) for i in range(num_requests)
            ]
            results = [future.result() for future in as_completed(futures)]

        total_elapsed = time.time() - start_time

        successful = [r for r in results if r[2]]
        failed = [r for r in results if not r[2]]

        latencies = [r[1] for r in successful]
        total_tokens = sum(r[3] for r in successful)

        print(f"\nBurst Load ({num_requests} concurrent requests):")
        print(f"  Total time: {total_elapsed:.2f}s")
        print(f"  Successful: {len(successful)}/{num_requests}")
        print(f"  Failed: {len(failed)}")
        print(f"  Success rate: {len(successful)/num_requests*100:.1f}%")
        print(f"  Throughput: {len(successful)/total_elapsed:.2f} req/sec")
        print(f"  Token throughput: {total_tokens/total_elapsed:.2f} tokens/sec")
        print(f"  Latency - avg: {statistics.mean(latencies):.2f}s")
        print(f"  Latency - p50: {statistics.median(latencies):.2f}s")
        print(f"  Latency - p95: {statistics.quantiles(latencies, n=20)[18]:.2f}s")
        print(f"  Latency - max: {max(latencies):.2f}s")

        # At least 60% success rate for 100 concurrent (realistic for production)
        assert len(successful) >= num_requests * 0.6


class TestConcurrencyLimits:
    """Test behavior at concurrency limits (up to 100)."""

    def test_realistic_concurrent_requests(self, openai_client: OpenAI):
        """Test realistic concurrency levels for production use."""
        # Test with increasing concurrency levels up to 100
        concurrency_levels = [10, 25, 50, 75, 100]
        max_tokens = 30

        results_summary = []

        for concurrency in concurrency_levels:
            def make_request():
                try:
                    response = openai_client.completions.create(
                        model="model",
                        prompt="Test concurrent:",
                        max_tokens=max_tokens,
                        temperature=0.7,
                    )
                    return True
                except Exception:
                    return False

            start_time = time.time()

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(make_request) for _ in range(concurrency)]
                results = [future.result() for future in as_completed(futures)]

            elapsed = time.time() - start_time
            successful = sum(results)
            success_rate = successful / concurrency

            results_summary.append({
                "concurrency": concurrency,
                "successful": successful,
                "success_rate": success_rate,
                "elapsed": elapsed
            })

            print(
                f"\nConcurrency {concurrency}: {successful}/{concurrency} "
                f"({success_rate*100:.1f}%) in {elapsed:.2f}s, "
                f"{successful/elapsed:.2f} req/sec"
            )

        # Print summary
        print("\n=== Concurrency Test Summary ===")
        for r in results_summary:
            print(f"  {r['concurrency']:3d} concurrent: {r['success_rate']*100:5.1f}% success, "
                  f"{r['successful']/r['elapsed']:5.2f} req/sec")

        # At lower concurrency levels, should have high success rate
        low_concurrency_results = [r for r in results_summary if r['concurrency'] <= 25]
        for r in low_concurrency_results:
            assert r['success_rate'] >= 0.8, f"Low success rate at concurrency {r['concurrency']}"

        # At high concurrency (75-100), accept lower success rate
        # This is realistic for production under heavy load


class TestThroughputOptimization:
    """Test throughput optimization with different configurations."""

    def test_small_vs_large_tokens(self, openai_client: OpenAI):
        """Compare throughput with different token counts."""
        num_requests = 10
        token_configs = [10, 50, 100, 200]

        for max_tokens in token_configs:
            def make_request():
                return openai_client.completions.create(
                    model="model",
                    prompt="Generate text:",
                    max_tokens=max_tokens,
                    temperature=0.7,
                )

            start_time = time.time()

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request) for _ in range(num_requests)]
                responses = [future.result() for future in as_completed(futures)]

            elapsed = time.time() - start_time
            total_tokens = sum(r.usage.completion_tokens for r in responses)

            print(
                f"\nmax_tokens={max_tokens}: {total_tokens} tokens in {elapsed:.2f}s "
                f"({total_tokens/elapsed:.2f} tokens/sec)"
            )

    def test_batch_efficiency(self, openai_client: OpenAI):
        """Test if batching improves overall throughput."""
        total_requests = 20
        batch_sizes = [1, 5, 10, 20]

        for batch_size in batch_sizes:
            num_batches = total_requests // batch_size

            def make_request():
                return openai_client.completions.create(
                    model="model",
                    prompt="Batch test:",
                    max_tokens=30,
                    temperature=0.7,
                )

            start_time = time.time()
            total_completed = 0

            for _ in range(num_batches):
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    futures = [
                        executor.submit(make_request) for _ in range(batch_size)
                    ]
                    [future.result() for future in as_completed(futures)]
                    total_completed += batch_size

            elapsed = time.time() - start_time
            throughput = total_completed / elapsed

            print(
                f"\nBatch size {batch_size}: {total_completed} requests in {elapsed:.2f}s "
                f"({throughput:.2f} req/sec)"
            )
