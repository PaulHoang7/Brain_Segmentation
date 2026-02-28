"""
LoRA (Low-Rank Adaptation) for SAM ViT-B.

Injects trainable low-rank matrices into attention layers (qkv, proj)
and mask decoder, keeping all other parameters frozen.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with low-rank adaptation."""

    def __init__(self, original: nn.Linear,
                 rank: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_f  = original.in_features
        out_f = original.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Kaiming init for A, zero init for B → LoRA starts as identity
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze original weights
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling


def inject_lora_sam(model: nn.Module,
                    rank: int = 16,
                    alpha: int = 32,
                    dropout: float = 0.05,
                    target_modules: List[str] | None = None
                    ) -> nn.Module:
    """
    Inject LoRA into SAM model attention layers and mask decoder.

    target_modules: list of substrings to match. Default targets
    SAM ViT attention qkv/proj and mask decoder MLP layers.

    Returns the modified model (in-place).
    """
    if target_modules is None:
        target_modules = ["attn.qkv", "attn.proj", "mask_decoder"]

    replaced = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue

        # Navigate to parent and replace
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr_name = parts[-1]

        lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr_name, lora_layer)
        replaced += 1

    print(f"[LoRA] Injected into {replaced} layers "
          f"(rank={rank}, alpha={alpha}, dropout={dropout})")
    return model


def lora_state_dict(model: nn.Module) -> dict:
    """Extract only LoRA parameters (for saving small checkpoints)."""
    return {k: v for k, v in model.state_dict().items()
            if "lora_" in k}


def load_lora_weights(model: nn.Module, path: str) -> nn.Module:
    """Load LoRA-only weights into an already-injected model."""
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model


def count_params(model: nn.Module) -> dict[str, int]:
    """Count total / trainable / frozen parameters."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
