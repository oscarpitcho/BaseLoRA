import torch
import torch.nn as nn
import sys
sys.path.insert(0, "src")

from batched_lora import BatchedLoRALinear
from lora import LoRALinear


# =============================================================================
# Numerical tests (manual verification possible)
# =============================================================================

def test_batched_lora_single_adapter_numerical():
    """2x2 base, rank-1 LoRA - same setup as test_lora_forward_numerical"""
    base = nn.Linear(2, 2, bias=False)
    base.weight.data = torch.tensor([[1., 0.], [0., 1.]])  # identity
    
    batched_lora = BatchedLoRALinear(base, r=1, alpha=1)
    A = torch.tensor([[1., 1.]])  # (rank=1, in_feat=2)
    B = torch.tensor([[1.], [0.]])  # (out_feat=2, rank=1)
    batched_lora.set_adapter(A, B)
    
    x = torch.tensor([[[1., 2.]]])  # (batch=1, seq=1, in_feat=2)
    out = batched_lora(x)
    
    # base: [1, 2] @ I = [1, 2]
    # lora_A: [1, 2] @ [1, 1]^T = 3
    # lora_B: 3 @ [1, 0]^T = [3, 0]
    # total: [1, 2] + [3, 0] = [4, 2]
    expected = torch.tensor([[[4., 2.]]])
    assert torch.allclose(out, expected)


def test_batched_lora_multi_adapter_numerical():
    """Two adapters, two samples, each using different adapter"""
    base = nn.Linear(2, 2, bias=False)
    base.weight.data = torch.zeros(2, 2)  # zero base for easy verification
    
    batched_lora = BatchedLoRALinear(base, r=1, alpha=1)
    
    # Adapter 0: A0 @ B0 = [[1,1]] @ [[1],[0]] -> adds [sum(x), 0]
    # Adapter 1: A1 @ B1 = [[1,0]] @ [[0],[1]] -> adds [0, x[0]]
    A0, B0 = torch.tensor([[1., 1.]]), torch.tensor([[1.], [0.]])
    A1, B1 = torch.tensor([[1., 0.]]), torch.tensor([[0.], [1.]])
    
    batched_lora.set_adapters([(A0, B0), (A1, B1)])
    batched_lora.set_adapter_ids(torch.tensor([0, 1]))
    
    x = torch.tensor([[[1., 2.]], [[3., 4.]]])  # (batch=2, seq=1, in=2)
    out = batched_lora(x)
    
    # Sample 0 (adapter 0): [1+2, 0] = [3, 0]
    # Sample 1 (adapter 1): [0, 3] = [0, 3]
    expected = torch.tensor([[[3., 0.]], [[0., 3.]]])
    assert torch.allclose(out, expected)


def test_batched_scaling():
    """Test that alpha/r scaling is applied correctly."""
    base = nn.Linear(2, 2, bias=False)
    base.weight.data = torch.zeros(2, 2)
    
    batched_lora = BatchedLoRALinear(base, r=1, alpha=2)  # scaling = 2
    A = torch.tensor([[1., 0.]])  # (rank=1, in_feat=2)
    B = torch.tensor([[1.], [1.]])  # (out_feat=2, rank=1)
    batched_lora.set_adapter(A, B)
    
    x = torch.tensor([[[3., 0.]]])  # (batch=1, seq=1, in_feat=2)
    out = batched_lora(x)
    
    # lora_A: 3, lora_B: [3, 3], scaled by 2: [6, 6]
    expected = torch.tensor([[[6., 6.]]])
    assert torch.allclose(out, expected)


# =============================================================================
# Equivalency tests (critical correctness validation)
# =============================================================================

def test_equivalency_single_adapter():
    """BatchedLoRALinear with single adapter == LoRALinear with same weights"""
    torch.manual_seed(42)
    base = nn.Linear(64, 32, bias=True)
    
    lora = LoRALinear(base, r=8, alpha=16)
    batched = BatchedLoRALinear(base, r=8, alpha=16)
    
    # Copy weights from LoRALinear to BatchedLoRALinear
    A = lora.lora_A.weight.data  # (r, in_feat)
    B = lora.lora_B.weight.data  # (out_feat, r)
    batched.set_adapter(A, B)
    
    x = torch.randn(4, 10, 64)  # (batch, seq, in_feat)
    
    lora_out = lora(x)
    batched_out = batched(x)
    
    assert torch.allclose(lora_out, batched_out, atol=1e-6)


def test_equivalency_multi_adapter_same_as_individual():
    """Each sample processed with correct adapter matches individual LoRA"""
    torch.manual_seed(42)
    base = nn.Linear(32, 16, bias=False)
    
    # Create two separate LoRALinear instances
    lora_0 = LoRALinear(base, r=4, alpha=8)
    lora_1 = LoRALinear(base, r=4, alpha=8)
    # Randomize weights differently
    nn.init.normal_(lora_1.lora_A.weight)
    nn.init.normal_(lora_1.lora_B.weight)
    
    # Create batched with both adapters
    batched = BatchedLoRALinear(base, r=4, alpha=8)
    batched.set_adapters([
        (lora_0.lora_A.weight.data, lora_0.lora_B.weight.data),
        (lora_1.lora_A.weight.data, lora_1.lora_B.weight.data),
    ])
    batched.set_adapter_ids(torch.tensor([0, 1, 0, 1]))
    
    x = torch.randn(4, 5, 32)
    batched_out = batched(x)
    
    # Verify each sample individually
    assert torch.allclose(batched_out[0], lora_0(x[0:1]).squeeze(0), atol=1e-6)
    assert torch.allclose(batched_out[1], lora_1(x[1:2]).squeeze(0), atol=1e-6)
    assert torch.allclose(batched_out[2], lora_0(x[2:3]).squeeze(0), atol=1e-6)
    assert torch.allclose(batched_out[3], lora_1(x[3:4]).squeeze(0), atol=1e-6)


# =============================================================================
# Property tests
# =============================================================================

def test_base_frozen_adapters_trainable():
    """Base layer params should be frozen, adapter params should require grad."""
    base = nn.Linear(64, 32)
    batched = BatchedLoRALinear(base, r=8, alpha=8)
    
    for p in batched.base.parameters():
        assert not p.requires_grad
    assert batched.A_all.requires_grad
    assert batched.B_all.requires_grad


def test_gradient_flow_single_adapter():
    """Gradients should flow through A_all, B_all in single-adapter path."""
    base = nn.Linear(16, 8, bias=False)
    batched = BatchedLoRALinear(base, r=2, alpha=2)
    
    # Set non-zero adapter weights (default B=zeros blocks gradient to A)
    A = torch.randn(2, 16)
    B = torch.randn(8, 2)
    batched.set_adapter(A, B)
    
    x = torch.randn(2, 4, 16)
    out = batched(x)
    loss = out.sum()
    loss.backward()
    
    assert batched.A_all.grad is not None
    assert batched.B_all.grad is not None
    assert batched.A_all.grad.abs().sum() > 0
    assert batched.B_all.grad.abs().sum() > 0


def test_gradient_flow_multi_adapter():
    """Gradients should flow correctly through indexing path."""
    base = nn.Linear(16, 8, bias=False)
    batched = BatchedLoRALinear(base, r=2, alpha=2)
    
    A0 = torch.randn(2, 16)
    B0 = torch.randn(8, 2)
    A1 = torch.randn(2, 16)
    B1 = torch.randn(8, 2)
    
    batched.set_adapters([(A0, B0), (A1, B1)])
    batched.set_adapter_ids(torch.tensor([0, 1, 0]))
    
    x = torch.randn(3, 4, 16)
    out = batched(x)
    loss = out.sum()
    loss.backward()
    
    assert batched.A_all.grad is not None
    assert batched.B_all.grad is not None
    # Both adapters should have gradients (adapter 0 used by samples 0,2; adapter 1 by sample 1)
    assert batched.A_all.grad[0].abs().sum() > 0
    assert batched.A_all.grad[1].abs().sum() > 0


def test_initialization_zeros_lora_contribution():
    """With default init (B=zeros), LoRA contribution should be zero."""
    base = nn.Linear(32, 16, bias=False)
    base.weight.data = torch.ones(16, 32)
    
    batched = BatchedLoRALinear(base, r=4, alpha=4)
    # Don't call set_adapter - use default initialization
    
    x = torch.randn(2, 5, 32)
    out = batched(x)
    
    # Output should equal base output since B is initialized to zeros
    base_out = base(x)
    assert torch.allclose(out, base_out)


# =============================================================================
# Edge cases
# =============================================================================

def test_all_samples_same_adapter_multi_mode():
    """When all adapter_ids are the same in multi-adapter setup."""
    base = nn.Linear(8, 4, bias=False)
    base.weight.data = torch.zeros(4, 8)
    
    batched = BatchedLoRALinear(base, r=1, alpha=1)
    
    A0 = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.]])
    B0 = torch.tensor([[1.], [0.], [0.], [0.]])
    A1 = torch.tensor([[0., 1., 0., 0., 0., 0., 0., 0.]])
    B1 = torch.tensor([[0.], [1.], [0.], [0.]])
    
    batched.set_adapters([(A0, B0), (A1, B1)])
    # All samples use adapter 0
    batched.set_adapter_ids(torch.tensor([0, 0, 0]))
    
    x = torch.tensor([
        [[1., 2., 0., 0., 0., 0., 0., 0.]],
        [[3., 4., 0., 0., 0., 0., 0., 0.]],
        [[5., 6., 0., 0., 0., 0., 0., 0.]],
    ])
    out = batched(x)
    
    # Adapter 0 extracts first element: [x[0], 0, 0, 0]
    expected = torch.tensor([
        [[1., 0., 0., 0.]],
        [[3., 0., 0., 0.]],
        [[5., 0., 0., 0.]],
    ])
    assert torch.allclose(out, expected)


def test_large_batch_sequence():
    """Test with larger batch and sequence sizes."""
    torch.manual_seed(123)
    base = nn.Linear(256, 128, bias=True)
    
    batched = BatchedLoRALinear(base, r=16, alpha=32)
    
    # Create multiple adapters
    adapters = [(torch.randn(16, 256), torch.randn(128, 16)) for _ in range(4)]
    batched.set_adapters(adapters)
    batched.set_adapter_ids(torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))
    
    x = torch.randn(8, 64, 256)  # batch=8, seq=64
    out = batched(x)
    
    assert out.shape == (8, 64, 128)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_set_adapter_overwrites():
    """Calling set_adapter should replace previous adapter."""
    base = nn.Linear(4, 2, bias=False)
    base.weight.data = torch.zeros(2, 4)
    
    batched = BatchedLoRALinear(base, r=1, alpha=1)
    
    # Set first adapter
    A1 = torch.tensor([[1., 0., 0., 0.]])
    B1 = torch.tensor([[1.], [0.]])
    batched.set_adapter(A1, B1)
    
    x = torch.tensor([[[1., 2., 3., 4.]]])
    out1 = batched(x).clone()
    
    # Set second adapter (different behavior)
    A2 = torch.tensor([[0., 1., 0., 0.]])
    B2 = torch.tensor([[0.], [1.]])
    batched.set_adapter(A2, B2)
    
    out2 = batched(x)
    
    # Outputs should be different
    assert not torch.allclose(out1, out2)
    # First adapter: [1, 0], second adapter: [0, 2]
    assert torch.allclose(out1, torch.tensor([[[1., 0.]]]))
    assert torch.allclose(out2, torch.tensor([[[0., 2.]]]))
