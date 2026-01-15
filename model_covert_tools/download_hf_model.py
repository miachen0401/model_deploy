#!/usr/bin/env python3
"""Download the Fine-tuned model from Hugging Face to a local directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from huggingface_hub import snapshot_download


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config file into a dict."""
    if not config_path.exists():
         raise FileNotFoundError(f"Config file not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_repo_id(config: Dict[str, Any]) -> str:
    """Read HUGGINGFACE MODEL repo_id from config.yml."""
    # Access keys safely
    fine_tune_section = config.get("HUGGINGFACE_MODEL")
    if not fine_tune_section:
        raise ValueError("Missing 'HuggingFace Model' section in config.yml")
    
    repo_id = fine_tune_section.get("NAME")
    if not repo_id or not isinstance(repo_id, str):
        raise ValueError("Missing 'HuggingFace Model.NAME' in config.yml")
        
    return repo_id


def get_hf_token() -> Optional[str]:
    """Get HF token from environment."""
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )


def download_repo(repo_id: str, out_dir: Path, token: Optional[str]) -> Path:
    """Download HF repo snapshot to out_dir and return the resolved path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} to {out_dir}...")
    local_dir = snapshot_download(
        repo_id=repo_id,
        token=token,
        local_dir=str(out_dir),
    )
    return Path(local_dir).resolve()


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Download HUGGINGFACE MODEL.")
    parser.add_argument("--config", default="config.yml", help="Path to config.yml")
    parser.add_argument(
        "--out_dir",
        default="model",
        help="Local directory to store the downloaded model",
    )
    parser.add_argument(
        "--env_file",
        default=".env",
        help="Path to .env file (contains HF_TOKEN)",
    )
    args = parser.parse_args()

    # Load env vars
    load_dotenv(args.env_file)

    config_path = Path(args.config)
    
    try:
        config = load_config(config_path)
        repo_id = get_repo_id(config)
        token = get_hf_token()

        out_dir = Path(args.out_dir)
        out_dir = Path(config.get("HUGGINGFACE_MODEL").get("OUTPUT"))
        local_path = download_repo(repo_id=repo_id, out_dir=out_dir, token=token)
        
        print(f"✅ Successfully downloaded model: {repo_id}")
        print(f"📂 Saved to: {local_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
