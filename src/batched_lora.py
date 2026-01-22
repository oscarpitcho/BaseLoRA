"""
Batched LoRA implementation for multi-adapter inference.

Stores multiple adapters and applies them per-sample within a batch,
avoiding CPU/GPU sync overhead from adapter switching.
"""

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum
from jaxtyping import Float, Int


class BatchedLoRALinear(nn.Module):
    """
    LoRA layer supporting multiple adapters with per-sample selection.
    
    Stores adapters as stacked tensors:
        A_all: (n_adapters, rank, in_features)
        B_all: (n_adapters, out_features, rank)
    
    Single adapter (n_adapters=1): uses broadcasting, zero indexing overhead.
    Multiple adapters: indexes per sample via adapter_ids.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 8):
        """
        Args:
            base: Linear layer to wrap (will be frozen)
            r: LoRA rank
            alpha: Scaling factor numerator (scaling = alpha / r)
        """
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r

        in_feat, out_feat = base.in_features, base.out_features

        self.A_all = nn.Parameter(torch.zeros(1, r, in_feat))
        self.B_all = nn.Parameter(torch.zeros(1, out_feat, r))
        self.adapter_ids: Optional[Tensor] = None

        nn.init.kaiming_uniform_(self.A_all, a=math.sqrt(5))
        nn.init.zeros_(self.B_all)

        for p in self.base.parameters():
            p.requires_grad = False

    def set_adapter(
        self,
        A: Float[Tensor, "rank in_feat"],
        B: Float[Tensor, "out_feat rank"],
    ):
        """
        Set single adapter weights. Uses broadcasting in forward.
        
        Args:
            A: (rank, in_features)
            B: (out_features, rank)
        """
        device = self.A_all.device
        self.A_all = nn.Parameter(A.unsqueeze(0).to(device))
        self.B_all = nn.Parameter(B.unsqueeze(0).to(device))
        self.adapter_ids = None

    def set_adapters(self, adapters: List[Tuple[Tensor, Tensor]]):
        """
        Set multiple adapters. Stacks them internally.
        
        Must call set_adapter_ids() before forward to specify which adapter per sample.
        
        Args:
            adapters: List of (A, B) tuples where A: (rank, in_feat), B: (out_feat, rank)
        """
        device = self.A_all.device
        A_list, B_list = zip(*adapters)
        self.A_all = nn.Parameter(torch.stack(A_list, dim=0).to(device))
        self.B_all = nn.Parameter(torch.stack(B_list, dim=0).to(device))

    def set_adapter_ids(self, adapter_ids: Int[Tensor, "batch"]):
        """
        Set which adapter to use for each sample in the batch.
        
        Args:
            adapter_ids: (batch,) tensor of indices into A_all/B_all
        """
        self.adapter_ids = adapter_ids

    def get_adapters(self) -> List[Tuple[Tensor, Tensor]]:
        """Return list of (A, B) tuples for all loaded adapters."""
        return [(self.A_all[i], self.B_all[i]) for i in range(self.A_all.shape[0])]

    def create_fresh_adapter(self) -> Tuple[Tensor, Tensor]:
        """Return newly initialized (A, B) tensors on CPU matching this layer's shape."""
        A = torch.empty(self.r, self.base.in_features)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        B = torch.zeros(self.base.out_features, self.r)
        return A, B

    def forward(
        self, x: Float[Tensor, "batch seq in_feat"]
    ) -> Float[Tensor, "batch seq out_feat"]:
        base_out = self.base(x)
        n_adapt = self.A_all.shape[0]

        if n_adapt == 1:
            # Single adapter: broadcast, no per-sample indexing
            A = self.A_all[0]
            B = self.B_all[0]

            intermediate = einsum(
                A, x,
                "rank in_feat, batch seq in_feat -> batch seq rank"
            )
            lora_out = einsum(
                B, intermediate,
                "out_feat rank, batch seq rank -> batch seq out_feat"
            )
        else:
            # Multiple adapters: index per sample
            A_batch = self.A_all[self.adapter_ids]
            B_batch = self.B_all[self.adapter_ids]

            intermediate = einsum(
                A_batch, x,
                "batch rank in_feat, batch seq in_feat -> batch seq rank"
            )
            lora_out = einsum(
                B_batch, intermediate,
                "batch out_feat rank, batch seq rank -> batch seq out_feat"
            )

        return base_out + self.scaling * lora_out


def inject_batched_lora(model: nn.Module, r: int = 8, alpha: int = 8) -> List[str]:
    """
    Replace all nn.Linear layers in model with BatchedLoRALinear layers.
    
    Freezes all non-LoRA parameters.
    
    Returns:
        List of module names that were converted to LoRA layers.
    """
    lora_names = []

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            lora_layer = BatchedLoRALinear(module, r=r, alpha=alpha)
            model.set_submodule(name, lora_layer)
            lora_names.append(name)
        else:
            for p in module.parameters(recurse=False):
                p.requires_grad = False

    return lora_names
