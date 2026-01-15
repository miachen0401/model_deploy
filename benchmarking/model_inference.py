"""
Model inference module for classifying news using deployed vLLM model.
"""
import logging
import json
import re
from typing import List, Dict, Optional, Tuple
import pandas as pd
from openai import OpenAI
from pathlib import Path

logger = logging.getLogger(__name__)


class NewsClassifier:
    """Classifier for news categorization using deployed vLLM model."""

    # Load classification rules
    RULES_FILE = Path(__file__).parent / "claasification_rules.txt"

    # All valid categories
    VALID_CATEGORIES = [
        "MACRO_ECONOMY",
        "CENTRAL_BANK_POLICY",
        "GEOPOLITICAL_EVENT",
        "INDUSTRY_REGULATION",
        "CORPORATE_EARNINGS",
        "CORPORATE_ACTIONS",
        "MANAGEMENT_CHANGE",
        "INCIDENT_LEGAL",
        "PRODUCT_TECH_UPDATE",
        "BUSINESS_OPERATIONS",
        "ANALYST_OPINION",
        "MARKET_SENTIMENT",
        "NON_FINANCIAL",
    ]

    def __init__(
        self,
        api_base_url: str = "http://localhost:8000/v1",
        model_name: str = "model",
        api_key: str = "EMPTY",
    ):
        """
        Initialize news classifier.

        Args:
            api_base_url: Base URL for vLLM OpenAI-compatible API
            model_name: Model name
            api_key: API key (default "EMPTY" for local vLLM)
        """
        self.client = OpenAI(api_key=api_key, base_url=api_base_url)
        self.model_name = model_name

        # Load classification rules
        if self.RULES_FILE.exists():
            with open(self.RULES_FILE, "r") as f:
                self.categories_definition = f.read()
        else:
            raise FileNotFoundError(f"Classification rules file not found: {self.RULES_FILE}")

        logger.info(f"NewsClassifier initialized (API: {api_base_url}, Model: {model_name})")

    def _build_prompt(self, news_items: List[Dict[str, str]]) -> str:
        """
        Build classification prompt for news items.

        Args:
            news_items: List of dictionaries with 'title' and 'summary' keys

        Returns:
            Formatted prompt string
        """
        news_text = ""
        for idx, item in enumerate(news_items, 1):
            title = item.get("title", "No title")
            summary = item.get("summary", "No summary")

            # Use full content without truncation
            if not title:
                title = "No title"
            if not summary:
                summary = "No summary"

            news_text += f"\n[NEWS {idx}]\nTitle: {title}\nSummary: {summary}\n"

        prompt = f"""{self.categories_definition}

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

        return prompt

    def _parse_response(self, response_text: str, num_items: int) -> List[Dict]:
        """
        Parse model response and extract classifications.

        Args:
            response_text: Raw model response
            num_items: Expected number of news items

        Returns:
            List of classification dictionaries (always exactly num_items)
        """
        valid_classifications = []

        try:
            # Extract all JSON arrays from response (model may generate multiple)
            json_arrays = re.findall(r'\[[\s\S]*?\]', response_text)

            for json_str in json_arrays:
                try:
                    classifications = json.loads(json_str)

                    # Handle both list and dict responses
                    if isinstance(classifications, dict):
                        classifications = [classifications]

                    if isinstance(classifications, list):
                        for item in classifications:
                            if isinstance(item, dict):
                                # Ensure category is valid
                                category = item.get("primary_category", "NON_FINANCIAL")
                                if category not in self.VALID_CATEGORIES:
                                    logger.warning(f"Invalid category '{category}', defaulting to NON_FINANCIAL")
                                    category = "NON_FINANCIAL"

                                valid_classifications.append({
                                    "news_id": item.get("news_id", len(valid_classifications) + 1),
                                    "primary_category": category,
                                    "symbol": item.get("symbol", ""),
                                    "confidence": float(item.get("confidence", 0.5)),
                                })
                except json.JSONDecodeError:
                    continue  # Try next JSON array

            if not valid_classifications:
                logger.error("No valid JSON found in response")
                logger.debug(f"Response text: {response_text[:500]}")

        except Exception as e:
            logger.error(f"Parsing error: {e}")
            logger.debug(f"Response text: {response_text[:500]}")

        # Ensure we always return exactly num_items results
        # Pad with defaults if we got fewer
        while len(valid_classifications) < num_items:
            valid_classifications.append({
                "news_id": len(valid_classifications) + 1,
                "primary_category": "NON_FINANCIAL",
                "symbol": "",
                "confidence": 0.0,
            })

        # Truncate if we got more (shouldn't happen, but be safe)
        if len(valid_classifications) > num_items:
            logger.warning(f"Got {len(valid_classifications)} classifications but expected {num_items}, truncating")
            valid_classifications = valid_classifications[:num_items]

        return valid_classifications

    def classify_batch(
        self,
        news_items: List[Dict[str, str]],
        max_tokens: int = 400,
        temperature: float = 0.3,
    ) -> List[Dict]:
        """
        Classify a batch of news items.

        Args:
            news_items: List of dictionaries with 'title' and 'summary'
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature

        Returns:
            List of classification results
        """
        if not news_items:
            return []

        prompt = self._build_prompt(news_items)

        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            response_text = response.choices[0].text
            classifications = self._parse_response(response_text, len(news_items))

            logger.debug(f"Classified {len(news_items)} items")
            return classifications

        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            # Return default classifications on error
            return [
                {
                    "news_id": i,
                    "primary_category": "NON_FINANCIAL",
                    "symbol": "",
                    "confidence": 0.0,
                }
                for i in range(1, len(news_items) + 1)
            ]

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        batch_size: int = 3,
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        Classify news items from a DataFrame.

        Args:
            df: DataFrame with 'title' and 'summary' columns
            batch_size: Number of items to classify per batch
            progress_callback: Optional callback function(current, total)

        Returns:
            DataFrame with added prediction columns
        """
        logger.info(f"Classifying {len(df)} news items in batches of {batch_size}")

        predictions = []
        total_batches = (len(df) + batch_size - 1) // batch_size

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            batch_items = [
                {"title": row.get("title", ""), "summary": row.get("summary", "")}
                for _, row in batch.iterrows()
            ]

            batch_predictions = self.classify_batch(batch_items)
            predictions.extend(batch_predictions)

            if progress_callback:
                progress_callback(i + len(batch), len(df))

            logger.info(f"Completed batch {i//batch_size + 1}/{total_batches}")

        # Add predictions to DataFrame
        df_copy = df.copy()
        df_copy["predicted_category"] = [p["primary_category"] for p in predictions]
        df_copy["predicted_symbol"] = [p["symbol"] for p in predictions]
        df_copy["prediction_confidence"] = [p["confidence"] for p in predictions]

        return df_copy


def main():
    """Example usage of NewsClassifier."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example news items
    news_items = [
        {
            "title": "Apple Reports Record Q4 Earnings",
            "summary": "Apple Inc. announced record fourth-quarter earnings today, beating analyst expectations with revenue of $89.5 billion.",
        },
        {
            "title": "Federal Reserve Signals Rate Cuts",
            "summary": "The Federal Reserve indicated potential interest rate cuts in the coming months based on recent economic indicators.",
        },
    ]

    try:
        # Initialize classifier
        classifier = NewsClassifier()

        # Classify news
        results = classifier.classify_batch(news_items)

        logger.info("\nClassification Results:")
        for i, result in enumerate(results, 1):
            logger.info(f"\nNews {i}:")
            logger.info(f"  Category: {result['primary_category']}")
            logger.info(f"  Symbol: {result['symbol']}")
            logger.info(f"  Confidence: {result['confidence']:.2f}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
