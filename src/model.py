"""Model configuration, hardware-aware siting and checkpoint persistence."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from transformer import GPT


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 192
    dropout: float = 0.1
    tie_weights: bool = True
    arch: str = "decoder_only_rmsnorm_rope_swiglu"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ModelConfig":
        fields = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        return cls(**fields)

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def estimated_parameters(self) -> int:
        """Closed-form parameter estimate used before the model is built."""
        attn = 4 * self.n_embd * self.n_embd            # qkv + proj
        hidden = 32 * ((int(2 * 4.0 * self.n_embd / 3) + 31) // 32)
        mlp = 3 * self.n_embd * hidden                  # w1, w3, w2
        per_block = attn + mlp + 2 * self.n_embd        # + 2 RMSNorm gains
        embeddings = self.vocab_size * self.n_embd
        head = 0 if self.tie_weights else self.vocab_size * self.n_embd
        return per_block * self.n_layer + embeddings + head + self.n_embd

    def estimate_memory_mb(self, batch_size: int, bytes_per_param: int = 4,
                           seq_len: Optional[int] = None) -> Dict[str, float]:
        """Rough training-memory budget (params, gradients, Adam states, activations)."""
        seq_len = seq_len or self.block_size
        params = self.estimated_parameters()
        model_mb = params * bytes_per_param / 1024 ** 2
        grad_mb = params * bytes_per_param / 1024 ** 2
        adam_mb = params * 8 / 1024 ** 2  # two float32 moments
        # attention + activations: extremely rough but useful as a lower bound
        activations = batch_size * seq_len * self.n_embd * self.n_layer * 4 * bytes_per_param
        activation_mb = activations / 1024 ** 2
        return {
            "parameters": params,
            "model_mb": round(model_mb, 2),
            "gradients_mb": round(grad_mb, 2),
            "optimizer_mb": round(adam_mb, 2),
            "activations_mb": round(activation_mb, 2),
            "total_train_mb": round(model_mb + grad_mb + adam_mb + activation_mb, 2),
        }


# Training presets chosen to fit the detected hardware.
PRESETS: Dict[str, Dict[str, Any]] = {
    "cpu_tiny": dict(block_size=128, n_layer=4, n_head=4, n_embd=192, dropout=0.1,
                     batch_size=16, grad_accum=1, epochs=4, lr=1.5e-3, max_steps=900),
    "cpu_small": dict(block_size=192, n_layer=5, n_head=6, n_embd=240, dropout=0.1,
                      batch_size=12, grad_accum=2, epochs=4, lr=1.2e-3, max_steps=1400),
    "mps_small": dict(block_size=256, n_layer=6, n_head=8, n_embd=320, dropout=0.1,
                      batch_size=24, grad_accum=1, epochs=6, lr=1e-3, max_steps=3000),
    "gpu_small": dict(block_size=256, n_layer=8, n_head=8, n_embd=384, dropout=0.1,
                      batch_size=48, grad_accum=1, epochs=8, lr=8e-4, max_steps=6000),
    "smoke": dict(block_size=64, n_layer=2, n_head=2, n_embd=64, dropout=0.0,
                  batch_size=8, grad_accum=1, epochs=1, lr=3e-3, max_steps=40),
}


def select_preset(hardware: Dict[str, Any], force: Optional[str] = None) -> str:
    """Pick a model/training preset from the real detected hardware."""
    if force:
        if force not in PRESETS:
            raise ValueError(f"Unknown preset '{force}'. Options: {sorted(PRESETS)}")
        return force
    device = hardware.get("selected_device", "cpu")
    ram = hardware.get("total_ram_gb") or 8
    if device == "cuda":
        vram = hardware.get("gpu_total_vram_gb") or 0
        return "gpu_small" if vram >= 6 else "mps_small"
    if device == "mps":
        return "mps_small"
    return "cpu_small" if ram >= 8 else "cpu_tiny"


def build_config(vocab_size: int, preset: str = "cpu_small", overrides: Optional[Dict[str, Any]] = None) -> ModelConfig:
    spec = dict(PRESETS[preset])
    cfg = ModelConfig(
        vocab_size=vocab_size,
        block_size=spec["block_size"],
        n_layer=spec["n_layer"],
        n_head=spec["n_head"],
        n_embd=spec["n_embd"],
        dropout=spec["dropout"],
    )
    for key, value in (overrides or {}).items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    if cfg.n_embd % cfg.n_head != 0:
        # keep head_dim integral and even (RoPE requirement)
        while cfg.n_head > 1 and (cfg.n_embd % cfg.n_head != 0 or (cfg.n_embd // cfg.n_head) % 2):
            cfg.n_head -= 1
    return cfg


def build_model(config: ModelConfig) -> GPT:
    return GPT(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        tie_weights=config.tie_weights,
    )


def save_config(config: ModelConfig, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config.to_dict(), fh, indent=2)
        fh.write("\n")


def load_config(path: str) -> ModelConfig:
    with open(path, "r", encoding="utf-8") as fh:
        return ModelConfig.from_dict(json.load(fh))


def save_checkpoint(
    path: str,
    model: GPT,
    optimizer: Optional[torch.optim.Optimizer],
    config: ModelConfig,
    step: int,
    epoch: int,
    best_val: Optional[float],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "config": config.to_dict(),
        "step": step,
        "epoch": epoch,
        "best_val": best_val,
        "torch_version": torch.__version__,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # older torch signature
        return torch.load(path, map_location=map_location)


def load_model_from_checkpoint(path: str, device: str = "cpu") -> Tuple[GPT, ModelConfig, Dict[str, Any]]:
    payload = load_checkpoint(path, map_location=device)
    config = ModelConfig.from_dict(payload["config"])
    model = build_model(config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, config, payload


def load_model_for_inference(models_dir: str, prefer: str = "best") -> Tuple[GPT, Any, ModelConfig, Dict[str, Any]]:
    """Load tokenizer + weights from a models/air_quality_slm style directory."""
    from tokenizer import ByteLevelBPETokenizer

    device = "cpu"
    try:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        device = "cpu"

    candidates = []
    if prefer == "final":
        candidates = ["final_model.pt", "best_model.pt"]
    else:
        candidates = ["best_model.pt", "final_model.pt"]
    weights_path = None
    for name in candidates:
        p = os.path.join(models_dir, name)
        if os.path.exists(p):
            weights_path = p
            break
    if weights_path is None:
        raise FileNotFoundError(
            f"No model weights in {models_dir}. Train first: python src/train_slm.py"
        )

    model, config, payload = load_model_from_checkpoint(weights_path, device=device)
    tokenizer_dir = os.path.join(models_dir, "tokenizer")
    tokenizer = ByteLevelBPETokenizer.load(tokenizer_dir)
    meta = {
        "weights_path": weights_path,
        "device": device,
        "step": payload.get("step"),
        "best_val": payload.get("best_val"),
        "extra": payload.get("extra", {}),
    }
    return model, tokenizer, config, meta
