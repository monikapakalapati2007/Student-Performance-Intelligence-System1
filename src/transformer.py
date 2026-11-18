"""Decoder-only causal Transformer building blocks (RoPE + RMSNorm + SwiGLU).

Written from scratch in PyTorch so the whole model is inspectable and runs on
CPU, Apple MPS or CUDA without external model libraries.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * norm)


def build_rotary_cache(head_dim: int, max_seq_len: int, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for rotary position embeddings."""
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)          # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)           # (T, head_dim)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate half-split Q/K tensors: x shape (B, H, T, D)."""
    T = x.size(-2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with rotary position embeddings."""

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.dropout = dropout
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # explicit boolean causal mask: True = allowed to attend
        if attn_mask is None:
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            attn_mask = causal.view(1, 1, T, T)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class SwiGLU(nn.Module):
    """Position-wise feed-forward network with a gated (SwiGLU) hidden layer."""

    def __init__(self, n_embd: int, hidden_mult: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(hidden_mult * n_embd)
        # keep the parameter count comparable to a classic 4x MLP
        hidden = int(2 * hidden / 3)
        hidden = 32 * ((hidden + 31) // 32)
        self.w1 = nn.Linear(n_embd, hidden, bias=False)
        self.w3 = nn.Linear(n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    """Pre-norm Transformer block: attention then feed-forward, both residual."""

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = SwiGLU(n_embd, 4.0, dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    """Compact decoder-only causal language model."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float = 0.0,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.norm_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        cos, sin = build_rotary_cache(n_embd // n_head, block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaled init on residual projections
        for name, param in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embedding.weight.numel()
        return n

    def causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.ones(T, T, dtype=torch.bool, device=device).tril().view(1, 1, T, T)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.block_size}")

        x = self.embed_dropout(self.token_embedding(idx))
        mask = self.causal_mask(T, idx.device)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin, mask)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        eos_id: Optional[int] = None,
        stop_strings: Optional[Tuple[str, ...]] = None,
        decode_fn=None,
        prompt_length: int = 0,
    ) -> torch.Tensor:
        """Autoregressive sampling with top-k filtering and EOS stopping.

        ``prompt_length`` marks how many leading tokens belong to the prompt so
        that ``stop_strings`` is only tested against newly generated text (testing
        the whole sequence would match markers that are part of the prompt).
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    values, _ = torch.topk(logits, k)
                    threshold = values[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            if eos_id is not None and bool((next_id == eos_id).all()):
                break
            if stop_strings and decode_fn is not None:
                text = decode_fn(idx[0, prompt_length:].tolist())
                if any(s in text for s in stop_strings):
                    break
        return idx
