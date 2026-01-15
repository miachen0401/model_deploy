"""
Supabase client for fetching stock news data.
"""
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class SupabaseNewsClient:
    """Client for fetching news data from Supabase."""

    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        """
        Initialize Supabase client.

        Args:
            supabase_url: Supabase project URL (or from env SUPABASE_URL)
            supabase_key: Supabase API key (or from env SUPABASE_KEY)
        """
        try:
            from supabase import create_client, Client
        except ImportError:
            raise ImportError(
                "supabase package not installed. "
                "Install with: pip install supabase"
            )

        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Supabase URL and Key must be provided either as arguments "
                "or through SUPABASE_URL and SUPABASE_KEY environment variables"
            )

        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        logger.info("Supabase client initialized successfully")

    def fetch_news_data(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        filters: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Fetch news data from stock_news table.

        Args:
            limit: Maximum number of records to fetch (None for all)
            offset: Number of records to skip
            filters: Dictionary of column filters (e.g., {'category': 'CORPORATE_EARNINGS'})

        Returns:
            DataFrame with news data including ground truth labels
        """
        logger.info(f"Fetching news data (limit={limit}, offset={offset})")

        try:
            # Build query
            query = self.client.table("stock_news").select("*")

            # Apply filters if provided
            if filters:
                for column, value in filters.items():
                    query = query.eq(column, value)

            # Apply pagination
            if offset:
                query = query.range(offset, offset + (limit - 1) if limit else 999999)
            elif limit:
                query = query.limit(limit)

            # Execute query
            response = query.execute()

            # Convert to DataFrame
            if response.data:
                df = pd.DataFrame(response.data)
                logger.info(f"Fetched {len(df)} news records")
                return df
            else:
                logger.warning("No data returned from Supabase")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching news data: {str(e)}")
            raise

    def fetch_labeled_data(
        self,
        limit: Optional[int] = None,
        require_category: bool = True,
        require_symbol: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch news data that has ground truth labels for benchmarking.

        Args:
            limit: Maximum number of records to fetch
            require_category: Only fetch records with category labels
            require_symbol: Only fetch records with symbol labels

        Returns:
            DataFrame with labeled news data
        """
        logger.info("Fetching labeled news data for benchmarking")

        try:
            query = self.client.table("stock_news").select("*")

            # Filter for labeled data
            if require_category:
                query = query.not_.is_("category", "null")

            if require_symbol:
                query = query.not_.is_("symbol", "null")

            if limit:
                query = query.limit(limit)

            response = query.execute()

            if response.data:
                df = pd.DataFrame(response.data)
                logger.info(f"Fetched {len(df)} labeled records")

                # Ensure required columns exist
                required_cols = ["title", "summary", "category"]
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logger.warning(f"Missing columns in data: {missing_cols}")

                return df
            else:
                logger.warning("No labeled data found")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching labeled data: {str(e)}")
            raise

    def get_category_distribution(self) -> Dict[str, int]:
        """
        Get distribution of categories in the database.

        Returns:
            Dictionary mapping category names to counts
        """
        logger.info("Fetching category distribution")

        try:
            response = (
                self.client.table("stock_news")
                .select("category")
                .not_.is_("category", "null")
                .execute()
            )

            if response.data:
                df = pd.DataFrame(response.data)
                distribution = df["category"].value_counts().to_dict()
                logger.info(f"Category distribution: {distribution}")
                return distribution
            else:
                return {}

        except Exception as e:
            logger.error(f"Error fetching category distribution: {str(e)}")
            raise

    def save_to_csv(self, df: pd.DataFrame, filename: str) -> str:
        """
        Save DataFrame to CSV file.

        Args:
            df: DataFrame to save
            filename: Output filename

        Returns:
            Full path to saved file
        """
        output_path = os.path.join("benchmarking/results", filename)
        os.makedirs("benchmarking/results", exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
        return output_path


def main():
    """Example usage of SupabaseNewsClient."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Initialize client
        client = SupabaseNewsClient()

        # Fetch labeled data
        df = client.fetch_labeled_data(limit=100)

        if not df.empty:
            logger.info(f"\nFetched {len(df)} records")
            logger.info(f"Columns: {df.columns.tolist()}")
            logger.info(f"\nFirst few records:")
            logger.info(df.head())

            # Get category distribution
            distribution = client.get_category_distribution()
            logger.info(f"\nCategory distribution:")
            for category, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {category}: {count}")

            # Save to CSV
            output_file = client.save_to_csv(
                df, f"labeled_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            logger.info(f"\nData saved to: {output_file}")

        else:
            logger.warning("No data fetched")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
