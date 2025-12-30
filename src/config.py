"""
Configuration management for the application.
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Config:
    """Application configuration manager"""

    def __init__(self, config_path: str = "config.yml"):
        """
        Initialize configuration.

        Args:
            config_path: Path to config.yml file
        """
        self.config_path = config_path
        self._config = None
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise

    @property
    def model_name(self) -> str:
        """Get Hugging Face model name"""
        return self._config['HUGGINGFACE_MODEL']['NAME']

    @property
    def hf_token(self) -> Optional[str]:
        """Get Hugging Face token from environment"""
        token = os.getenv('HF_TOKEN_READ')
        if not token:
            logger.warning("HF_TOKEN_READ not found in environment variables")
        return token

    @property
    def port(self) -> int:
        """Get server port"""
        return int(os.getenv("PORT", 8000))

    def get(self, key: str, default=None):
        """Get configuration value by key"""
        return self._config.get(key, default)


# Global config instance
_config_instance = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
