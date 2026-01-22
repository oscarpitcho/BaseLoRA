import torch
import torch.nn as nn
import sys
import pytest
sys.path.insert(0, "src")

from batched_lora import inject_batched_lora, BatchedLoRALinear
from adapter_manager import AdapterCache, AdapterManager


@pytest.fixture
def simple_model_with_adapters(tmp_path):
    """Create a simple model with BatchedLoRALinear layers and two registered adapters."""
    # Create model: embedding -> linear -> output
    model = nn.Sequential(
        nn.Embedding(100, 64),
        nn.Linear(64, 32),
    )
    inject_batched_lora(model, r=4, alpha=4)
    
    # Create two synthetic adapters with different weights
    torch.manual_seed(42)
    adapter_a = {"1": (torch.randn(4, 64), torch.randn(32, 4))}
    torch.manual_seed(123)
    adapter_b = {"1": (torch.randn(4, 64), torch.randn(32, 4))}
    
    # Save to files
    AdapterManager._save_adapter_dict(adapter_a, str(tmp_path / "adapter_a.safetensors"))
    AdapterManager._save_adapter_dict(adapter_b, str(tmp_path / "adapter_b.safetensors"))
    
    manager = AdapterManager(model, max_cache_entries=100)
    manager.register_adapter("adapter_a", str(tmp_path / "adapter_a.safetensors"))
    manager.register_adapter("adapter_b", str(tmp_path / "adapter_b.safetensors"))
    
    return manager, model, adapter_a, adapter_b


def test_inject_replaces_linear():
    model = nn.Sequential(nn.Linear(64, 32))
    lora_names = inject_batched_lora(model)
    
    assert isinstance(model[0], BatchedLoRALinear)
    assert set(lora_names) == {"0"}


def test_inject_recursive():
    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
        )
    )
    lora_names = inject_batched_lora(model)
    
    assert isinstance(model[0], BatchedLoRALinear)
    assert isinstance(model[1][0], BatchedLoRALinear)
    assert isinstance(model[1][1], nn.ReLU)
    assert set(lora_names) == {"0", "1.0"}


def test_inject_preserves_r_alpha():
    model = nn.Sequential(nn.Linear(64, 32))
    lora_names = inject_batched_lora(model, r=4, alpha=16)
    
    layer = model.get_submodule(lora_names[0])
    assert layer.r == 4
    assert layer.scaling == 16 / 4


# AdapterCache tests

def test_cache_hit():
    cache = AdapterCache(max_entries=10)
    input_ids = torch.tensor([1, 2, 3, 4, 5])
    embedding = torch.randn(256)
    
    cache.put(input_ids, "adapter_a", embedding)
    
    result = cache.get(input_ids, "adapter_a")
    assert result is not None
    assert torch.equal(result, embedding)


def test_cache_miss_different_adapter():
    cache = AdapterCache(max_entries=10)
    input_ids = torch.tensor([1, 2, 3, 4, 5])
    embedding = torch.randn(256)
    
    cache.put(input_ids, "adapter_a", embedding)
    
    # Same input_ids but different adapter should miss
    result = cache.get(input_ids, "adapter_b")
    assert result is None


def test_cache_miss_different_input():
    cache = AdapterCache(max_entries=10)
    input_ids_a = torch.tensor([1, 2, 3, 4, 5])
    input_ids_b = torch.tensor([1, 2, 3, 4, 6])
    embedding = torch.randn(256)
    
    cache.put(input_ids_a, "adapter_a", embedding)
    
    # Different input_ids should miss
    result = cache.get(input_ids_b, "adapter_a")
    assert result is None


def test_cache_lru_eviction():
    cache = AdapterCache(max_entries=3)
    
    # Fill cache to capacity
    for i in range(3):
        input_ids = torch.tensor([i])
        cache.put(input_ids, "adapter", torch.tensor([float(i)]))
    
    # All three should be present
    for i in range(3):
        assert cache.get(torch.tensor([i]), "adapter") is not None
    
    # Add a fourth entry - should evict the LRU (i=0, since we just accessed all in order 0,1,2)
    # But actually, the get() calls above moved them to end, so order is now 0,1,2
    # Adding new entry should evict 0
    cache.put(torch.tensor([99]), "adapter", torch.tensor([99.0]))
    
    # Entry 0 should be evicted
    assert cache.get(torch.tensor([0]), "adapter") is None
    # Entries 1, 2, and 99 should still be present
    assert cache.get(torch.tensor([1]), "adapter") is not None
    assert cache.get(torch.tensor([2]), "adapter") is not None
    assert cache.get(torch.tensor([99]), "adapter") is not None


def test_cache_lru_access_updates_order():
    cache = AdapterCache(max_entries=3)
    
    # Fill cache: 0, 1, 2 (in that order)
    for i in range(3):
        cache.put(torch.tensor([i]), "adapter", torch.tensor([float(i)]))
    
    # Access entry 0 to move it to end
    cache.get(torch.tensor([0]), "adapter")
    
    # Now order is: 1, 2, 0 (oldest to newest)
    # Adding new entry should evict 1
    cache.put(torch.tensor([99]), "adapter", torch.tensor([99.0]))
    
    # Entry 1 should be evicted (it was LRU)
    assert cache.get(torch.tensor([1]), "adapter") is None
    # Entries 0, 2, and 99 should still be present
    assert cache.get(torch.tensor([0]), "adapter") is not None
    assert cache.get(torch.tensor([2]), "adapter") is not None
    assert cache.get(torch.tensor([99]), "adapter") is not None


def test_cache_clear():
    cache = AdapterCache(max_entries=10)
    
    for i in range(5):
        cache.put(torch.tensor([i]), "adapter", torch.tensor([float(i)]))
    
    cache.clear()
    
    # All entries should be gone
    for i in range(5):
        assert cache.get(torch.tensor([i]), "adapter") is None


# AdapterManager.forward() tests

def test_forward_single_adapter(simple_model_with_adapters):
    """Basic forward pass returns correct output shape."""
    manager, model, _, _ = simple_model_with_adapters
    x = torch.randint(0, 100, (4, 8))  # batch=4, seq=8
    
    output = manager.forward(x, adapter_name="adapter_a")
    
    assert output.shape == (4, 8, 32)  # batch, seq, out_features


def test_forward_sets_adapter_when_name_provided(simple_model_with_adapters):
    """Verify set_adapters is called when adapter_name is provided."""
    manager, model, _, _ = simple_model_with_adapters
    x = torch.randint(0, 100, (2, 4))
    
    # Initially no adapters set
    assert manager.current_adapters is None
    
    manager.forward(x, adapter_name="adapter_a")
    
    # Now adapter_a should be set
    assert manager.current_adapters == frozenset(["adapter_a"])


def test_forward_uses_current_adapter_when_name_none(simple_model_with_adapters):
    """Forward without adapter_name uses existing adapter state."""
    manager, model, _, _ = simple_model_with_adapters
    x = torch.randint(0, 100, (2, 4))
    
    # First set an adapter
    manager.set_adapters("adapter_a")
    
    # Forward without specifying adapter_name
    output1 = manager.forward(x, adapter_name=None)
    
    # Should still use adapter_a
    assert manager.current_adapters == frozenset(["adapter_a"])
    
    # Output should match explicit call
    output2 = manager.forward(x, adapter_name="adapter_a")
    assert torch.allclose(output1, output2)


# AdapterManager.forward_multi() tests

def test_forward_multi_different_adapters_per_sample(simple_model_with_adapters):
    """Batch with samples using different adapters produces correct outputs."""
    manager, model, _, _ = simple_model_with_adapters
    x = torch.randint(0, 100, (4, 8))
    adapter_ids = ["adapter_a", "adapter_b", "adapter_a", "adapter_b"]
    
    output = manager.forward_multi(x, adapter_ids)
    
    assert output.shape == (4, 8, 32)


def test_forward_multi_all_same_adapter(simple_model_with_adapters):
    """All samples use same adapter - verify output matches forward()."""
    manager, model, _, _ = simple_model_with_adapters
    torch.manual_seed(999)
    x = torch.randint(0, 100, (4, 8))
    adapter_ids = ["adapter_a", "adapter_a", "adapter_a", "adapter_a"]
    
    # Clear cache to ensure fresh computation
    manager.cache.clear()
    
    output_multi = manager.forward_multi(x, adapter_ids)
    output_single = manager.forward(x, adapter_name="adapter_a")
    
    assert torch.allclose(output_multi, output_single)


def test_forward_multi_populates_cache(simple_model_with_adapters):
    """After forward_multi, cache contains the computed embeddings."""
    manager, model, _, _ = simple_model_with_adapters
    torch.manual_seed(888)
    x = torch.randint(0, 100, (2, 4))
    adapter_ids = ["adapter_a", "adapter_b"]
    
    # Clear cache
    manager.cache.clear()
    assert len(manager.cache.cache) == 0
    
    manager.forward_multi(x, adapter_ids)
    
    # Cache should now have 2 entries
    assert len(manager.cache.cache) == 2
    
    # Verify each sample is cached
    assert manager.cache.get(x[0], "adapter_a") is not None
    assert manager.cache.get(x[1], "adapter_b") is not None


def test_forward_multi_uses_cached_results(simple_model_with_adapters):
    """Repeated call with same inputs returns cached embeddings."""
    manager, model, _, _ = simple_model_with_adapters
    torch.manual_seed(777)
    x = torch.randint(0, 100, (2, 4))
    adapter_ids = ["adapter_a", "adapter_b"]
    
    manager.cache.clear()
    
    output1 = manager.forward_multi(x, adapter_ids)
    output2 = manager.forward_multi(x, adapter_ids)
    
    # Outputs should be identical (from cache)
    assert torch.equal(output1, output2)


def test_forward_multi_partial_cache_hit(simple_model_with_adapters):
    """Mix of cached and uncached samples - verify only uncached samples run inference."""
    manager, model, _, _ = simple_model_with_adapters
    torch.manual_seed(666)
    x = torch.randint(0, 100, (4, 4))
    
    manager.cache.clear()
    
    # Pre-populate cache with first two samples
    adapter_ids_partial = ["adapter_a", "adapter_b"]
    manager.forward_multi(x[:2], adapter_ids_partial)
    
    # Now run with all 4 samples
    adapter_ids_full = ["adapter_a", "adapter_b", "adapter_a", "adapter_b"]
    output = manager.forward_multi(x, adapter_ids_full)
    
    assert output.shape == (4, 4, 32)
    
    # Cache should now have 4 entries (2 old + 2 new)
    assert len(manager.cache.cache) == 4

# AdapterManager.register_adapter() / save_adapter() roundtrip tests

def test_save_and_register_roundtrip(tmp_path):
    """Save an adapter, register it, verify weights match."""
    # Create original adapter weights
    torch.manual_seed(555)
    original_A = torch.randn(4, 64)
    original_B = torch.randn(32, 4)
    adapter = {"layer1": (original_A, original_B)}
    
    # Save to file
    path = str(tmp_path / "test_adapter.safetensors")
    AdapterManager._save_adapter_dict(adapter, path)
    
    # Create model and manager
    model = nn.Sequential(
        nn.Embedding(100, 64),
        nn.Linear(64, 32),
    )
    inject_batched_lora(model, r=4, alpha=4)
    
    # Rename layer to match saved adapter
    # The injected layer is named "1", but we saved as "layer1"
    # So let's save with the correct name
    adapter_correct = {"1": (original_A, original_B)}
    path_correct = str(tmp_path / "test_adapter_correct.safetensors")
    AdapterManager._save_adapter_dict(adapter_correct, path_correct)
    
    manager = AdapterManager(model, max_cache_entries=0)
    manager.register_adapter("test_adapter", path_correct)
    
    # Verify weights are loaded correctly
    loaded_A, loaded_B = manager.registered_adapters["test_adapter"]["1"]
    
    assert torch.allclose(loaded_A, original_A)
    assert torch.allclose(loaded_B, original_B)


# Integration / Correctness tests

def test_lora_modifies_output(simple_model_with_adapters):
    """Output with LoRA differs from base model output."""
    manager, model, _, _ = simple_model_with_adapters
    torch.manual_seed(444)
    x = torch.randint(0, 100, (2, 4))
    
    # Get output with LoRA adapter
    lora_output = manager.forward(x, adapter_name="adapter_a")
    
    # Get base output (by setting LoRA weights to zero)
    layer = model.get_submodule("1")
    original_A = layer.A_all.clone()
    original_B = layer.B_all.clone()
    
    # Zero out LoRA weights
    layer.A_all = nn.Parameter(torch.zeros_like(layer.A_all))
    layer.B_all = nn.Parameter(torch.zeros_like(layer.B_all))
    
    base_output = model(x)
    
    # Restore original weights
    layer.A_all = nn.Parameter(original_A)
    layer.B_all = nn.Parameter(original_B)
    
    # LoRA output should differ from base output
    assert not torch.allclose(lora_output, base_output)


def test_different_adapters_produce_different_outputs(simple_model_with_adapters):
    """Two adapters yield different embeddings for same input."""
    manager, model, _, _ = simple_model_with_adapters
    torch.manual_seed(333)
    x = torch.randint(0, 100, (2, 4))
    
    output_a = manager.forward(x, adapter_name="adapter_a")
    output_b = manager.forward(x, adapter_name="adapter_b")
    
    # Outputs should be different since adapters have different weights
    assert not torch.allclose(output_a, output_b)


def test_batched_vs_sequential_equivalence(simple_model_with_adapters):
    """forward_multi with mixed adapters produces same results as running forward sequentially per adapter."""
    manager, model, _, _ = simple_model_with_adapters
    torch.manual_seed(222)
    x = torch.randint(0, 100, (4, 8))
    adapter_ids = ["adapter_a", "adapter_b", "adapter_a", "adapter_b"]
    
    # Clear cache to ensure fresh computation
    manager.cache.clear()
    
    # Batched result
    batched_out = manager.forward_multi(x, adapter_ids)
    
    # Sequential result
    sequential_out = []
    for i, adapter in enumerate(adapter_ids):
        manager.set_adapters(adapter)
        out = manager.forward(x[i:i+1], adapter)
        sequential_out.append(out.squeeze(0))
    sequential_out = torch.stack(sequential_out)
    
    assert torch.allclose(batched_out, sequential_out, atol=1e-5)


def test_forward_multi_numerical():
    """Test batched multi-adapter inference with 2x2 base, rank-1 LoRA for manual verification."""
    base = nn.Linear(2, 2, bias=False)
    base.weight.data = torch.tensor([[1., 0.], [0., 1.]])  # identity
    
    lora = BatchedLoRALinear(base, r=1, alpha=1)
    
    # Adapter A: A=[1,1], B=[1,0]^T  (same as test_lora_forward_numerical)
    # Adapter B: A=[1,0], B=[0,1]^T  (different adapter)
    A_a = torch.tensor([[1., 1.]])   # (1, 2)
    B_a = torch.tensor([[1.], [0.]]) # (2, 1)
    A_b = torch.tensor([[1., 0.]])   # (1, 2)
    B_b = torch.tensor([[0.], [1.]]) # (2, 1)
    
    lora.set_adapters([(A_a, B_a), (A_b, B_b)])
    lora.set_adapter_ids(torch.tensor([0, 1]))  # sample 0 -> adapter A, sample 1 -> adapter B
    
    x = torch.tensor([
        [[1., 2.]],  # sample 0, seq_len=1
        [[3., 4.]],  # sample 1, seq_len=1
    ])
    out = lora(x)
    
    # Sample 0 with adapter A:
    #   base: [1,2] @ I = [1,2]
    #   A: [1,2] @ [1,1]^T = 3
    #   B: 3 @ [1,0]^T = [3,0]
    #   total: [1,2] + [3,0] = [4,2]
    #
    # Sample 1 with adapter B:
    #   base: [3,4] @ I = [3,4]
    #   A: [3,4] @ [1,0]^T = 3
    #   B: 3 @ [0,1]^T = [0,3]
    #   total: [3,4] + [0,3] = [3,7]
    
    expected = torch.tensor([
        [[4., 2.]],
        [[3., 7.]],
    ])
    assert torch.allclose(out, expected)
