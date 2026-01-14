#!/usr/bin/env python3
"""Export an RL/FSDP actor checkpoint (.pt) to Hugging Face HF format (safetensors)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config file into a dict."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_repo_id(config: Dict[str, Any]) -> str:
    """Read repo_id from config.yml."""
    repo_id = config.get("HUGGINGFACE_MODEL", {}).get("NAME")
    if not repo_id or not isinstance(repo_id, str):
        raise ValueError('Missing HUGGINGFACE_MODEL.NAME in config.yml')
    return repo_id


def get_base_model_id(config: Dict[str, Any], cli_value: Optional[str]) -> str:
    """Get base model id (architecture) used to rebuild the model for export.

    Priority:
    1) CLI --base_model_id
    2) config.yml: HUGGINGFACE_MODEL.BASE
    3) default (you should change this to your actual base model)
    """
    if cli_value:
        return cli_value

    base = config.get("HUGGINGFACE_MODEL", {}).get("BASE")
    if isinstance(base, str) and base.strip():
        return base.strip()

    # Best-effort default; change to the exact base you trained from.
    return "Qwen/Qwen2.5-1.5B-Instruct"


def get_hf_token() -> Optional[str]:
    """Get HF token from environment (.env is supported)."""
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )


def find_actor_ckpt(repo_dir: Path) -> Path:
    """Find the most likely actor checkpoint file in a downloaded repo.

    Heuristic:
    - prefer files named like model_world_size*_rank_*.pt
    - otherwise fall back to the largest *.pt under the repo
    """
    patterns = [
        "**/model_world_size*_rank_*.pt",
        "**/*model*rank*.pt",
    ]

    candidates = []
    for pat in patterns:
        candidates.extend(repo_dir.glob(pat))

    candidates = [p for p in candidates if p.is_file()]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_size).resolve()

    pt_files = [p for p in repo_dir.glob("**/*.pt") if p.is_file()]
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found under: {repo_dir}")

    return max(pt_files, key=lambda p: p.stat().st_size).resolve()


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Extract a model state_dict from common checkpoint formats."""
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    for key in ("model", "state_dict", "module"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]

    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt

    raise ValueError(f"Cannot find state_dict in checkpoint keys: {list(ckpt.keys())[:50]}")


def strip_prefixes(
    state_dict: Dict[str, torch.Tensor],
    prefixes: Tuple[str, ...] = (
        "module.",
        "model.",
        "_fsdp_wrapped_module.",
        "actor.",
        "policy.",
    ),
) -> Dict[str, torch.Tensor]:
    """Strip common wrapper prefixes from checkpoint parameter names."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_k = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if new_k.startswith(p):
                    new_k = new_k[len(p):]
                    changed = True
        out[new_k] = v
    return out


def export_checkpoint_to_hf(
    base_model_id: str,
    ckpt_path: Path,
    out_dir: Path,
    dtype: str,
) -> None:
    """Load base model, apply actor checkpoint, and export as HF safetensors."""
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="cpu",
    )

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = strip_prefixes(extract_state_dict(ckpt))

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing examples:", missing[:20])
    if unexpected:
        print("  unexpected examples:", unexpected[:20])

    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(out_dir)
    print(f"Exported HF model to: {out_dir}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml", help="Path to config.yml")
    parser.add_argument("--repo_dir", default="hf_repo", help="Local repo directory")
    parser.add_argument("--out_dir", default="exported_hf", help="Output HF model dir")
    parser.add_argument("--env_file", default=".env", help="Path to .env file")
    parser.add_argument(
        "--base_model_id",
        default=None,
        help="Base model id (if not set, uses config.yml HUGGINGFACE_MODEL.BASE)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Export dtype used when rebuilding the base model on CPU",
    )
    parser.add_argument(
        "--ckpt_path",
        default=None,
        help="Optional explicit checkpoint path; otherwise auto-detects the largest actor-ish .pt",
    )
    args = parser.parse_args()

    load_dotenv(args.env_file)

    config = load_config(Path(args.config))
    _ = get_repo_id(config)  # validate presence
    base_model_id = get_base_model_id(config, args.base_model_id)

    repo_dir = Path(args.repo_dir).resolve()
    if args.ckpt_path:
        ckpt_path = Path(args.ckpt_path).resolve()
    else:
        ckpt_path = find_actor_ckpt(repo_dir)

    print(f"Using base_model_id: {base_model_id}")
    print(f"Using ckpt_path:     {ckpt_path}")

    export_checkpoint_to_hf(
        base_model_id=base_model_id,
        ckpt_path=ckpt_path,
        out_dir=Path(args.out_dir).resolve(),
        dtype=args.dtype,
    )

    # Quick load test
    print("Quick load test...")
    _ = AutoTokenizer.from_pretrained(str(Path(args.out_dir).resolve()), trust_remote_code=True)
    _ = AutoModelForCausalLM.from_pretrained(
        str(Path(args.out_dir).resolve()),
        trust_remote_code=True,
        device_map="cpu",
    )
    print("OK: exported model can be loaded by transformers.")


if __name__ == "__main__":
    main()
