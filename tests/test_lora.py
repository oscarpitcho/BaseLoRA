import torch
import torch.nn as nn
import sys
sys.path.insert(0, "src")

from lora import LoRALinear


def test_lora_forward_numerical():
    """Test with 2x2 base, rank-1 LoRA for easy manual verification."""
    base = nn.Linear(2, 2, bias=False)
    base.weight.data = torch.tensor([[1., 0.], [0., 1.]])  # identity
    
    lora = LoRALinear(base, r=1, alpha=1)
    lora.lora_A.weight.data = torch.tensor([[1., 1.]])  # (1, 2)
    lora.lora_B.weight.data = torch.tensor([[1.], [0.]])  # (2, 1)
    
    x = torch.tensor([[1., 2.]])
    out = lora(x)
    
    # base: [1, 2] @ I = [1, 2]
    # lora_A: [1, 2] @ [1, 1]^T = 3
    # lora_B: 3 @ [1, 0]^T = [3, 0]
    # total: [1, 2] + [3, 0] = [4, 2]
    expected = torch.tensor([[4., 2.]])
    assert torch.allclose(out, expected)


def test_lora_scaling():
    """Test that alpha/r scaling is applied correctly."""
    base = nn.Linear(2, 2, bias=False)
    base.weight.data = torch.zeros(2, 2)
    
    lora = LoRALinear(base, r=1, alpha=2)  # scaling = 2
    lora.lora_A.weight.data = torch.tensor([[1., 0.]])
    lora.lora_B.weight.data = torch.tensor([[1.], [1.]])
    
    x = torch.tensor([[3., 0.]])
    out = lora(x)
    
    # lora_A: 3, lora_B: [3, 3], scaled: [6, 6]
    expected = torch.tensor([[6., 6.]])
    assert torch.allclose(out, expected)


def test_base_frozen_lora_trainable():
    base = nn.Linear(2, 2)
    lora = LoRALinear(base, r=1, alpha=1)
    
    for p in lora.base.parameters():
        assert not p.requires_grad
    assert lora.lora_A.weight.requires_grad
    assert lora.lora_B.weight.requires_grad
