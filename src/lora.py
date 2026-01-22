"""
Single-adapter LoRA implementation.

Provides LoRALinear layer and injection utility for adding LoRA adapters to nn.Linear layers.
"""

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float


class LoRALinear(nn.Module):
    """
    LoRA-adapted linear layer.
    
    Wraps a frozen base nn.Linear and adds trainable low-rank decomposition:
        output = base(x) + scaling * B(A(x))
    
    where A: (in_features -> r) and B: (r -> out_features).
    """
    
    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: int = 8,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            base: Linear layer to wrap (will be frozen)
            r: LoRA rank
            alpha: Scaling factor numerator (scaling = alpha / r)
            device: Target device for LoRA parameters
        """
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r

        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: Float[Tensor, "batch seq hidden"]) -> Float[Tensor, "batch seq hidden"]:
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(x))

    def set_adapter(
        self,
        A: Float[Tensor, "r in_features"],
        B: Float[Tensor, "out_features r"],
    ):
        """Load adapter weights from tensors."""
        device = self.lora_A.weight.device
        self.lora_A.weight.data = A.to(device)
        self.lora_B.weight.data = B.to(device)

    def get_adapter(self) -> Tuple[Float[Tensor, "r in_features"], Float[Tensor, "out_features r"]]:
        """Return current adapter weights (A, B)."""
        return self.lora_A.weight.data, self.lora_B.weight.data


def inject_lora(model: nn.Module, r: int = 8, alpha: int = 8) -> List[str]:
    """
    Replace all nn.Linear layers in model with LoRALinear layers.
    
    Freezes all non-LoRA parameters.
    
    Returns:
        List of module names that were converted to LoRA layers.
    """
    lora_names = []

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            lora_layer = LoRALinear(module, r=r, alpha=alpha)
            model.set_submodule(name, lora_layer)
            lora_names.append(name)
        else:
            for p in module.parameters(recurse=False):
                p.requires_grad = False

    return lora_names
