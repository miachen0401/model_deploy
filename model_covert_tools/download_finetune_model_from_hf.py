#!/usr/bin/env python3
"""Download a Hugging Face repo snapshot to a local directory."""

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
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_repo_id(config: Dict[str, Any]) -> str:
    """Read repo_id from config.yml."""
    repo_id = (
        config.get("HUGGINGFACE_MODEL", {})
        .get("NAME")
    )
    if not repo_id or not isinstance(repo_id, str):
        raise ValueError('Missing HUGGINGFACE_MODEL.NAME in config.yml')
    return repo_id


def get_hf_token() -> Optional[str]:
    """Get HF token from environment (.env is supported)."""
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )


def download_repo(repo_id: str, out_dir: Path, token: Optional[str]) -> Path:
    """Download HF repo snapshot to out_dir and return the resolved path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    local_dir = snapshot_download(
        repo_id=repo_id,
        token=token,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return Path(local_dir).resolve()


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml", help="Path to config.yml")
    parser.add_argument(
        "--out_dir",
        default="hf_repo",
        help="Local directory to store the downloaded repo snapshot",
    )
    parser.add_argument(
        "--env_file",
        default=".env",
        help="Path to .env file (contains HF_TOKEN)",
    )
    args = parser.parse_args()

    load_dotenv(args.env_file)

    config_path = Path(args.config)
    config = load_config(config_path)
    repo_id = get_repo_id(config)
    token = get_hf_token()

    out_dir = Path(args.out_dir)
    local_path = download_repo(repo_id=repo_id, out_dir=out_dir, token=token)
    print(f"Downloaded repo: {repo_id}")
    print(f"Local path: {local_path}")


if __name__ == "__main__":
    main()
