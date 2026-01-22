"""
Adapter management for batched multi-adapter LoRA finetuning and inference.

Handles adapter registration, loading from safetensors, caching, and batch dispatch.
"""

import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from safetensors.torch import load_file, save_file


class AdapterManager:
    """
    Manages multiple LoRA adapters for a model with BatchedLoRALinear layers.
    
    Handles:
        - Loading adapters from safetensors files to CPU
        - Setting active adapters on model layers
        - Optional LRU caching of inference results
    """

    def __init__(self, model: nn.Module, r: int = 8, alpha: int = 8, max_cache_entries: int = 1000):
        """
        Args:
            model: Base model (will be modified in-place with BatchedLoRALinear layers).
                   Other layers are frozen
            r: LoRA rank
            alpha: LoRA scaling factor
            max_cache_entries: Max cached embeddings (0 disables caching)
        """
        from batched_lora import inject_batched_lora
        
        self.lora_names = inject_batched_lora(model, r=r, alpha=alpha)
        self.model = model
        self.registered_adapters: Dict[str, Dict[str, Tuple[Tensor, Tensor]]] = {}
        self.current_adapters: Optional[frozenset[str]] = None
        self._adapter_name_to_idx: Dict[str, int] = {}
        self.cache = AdapterCache(max_entries=max_cache_entries) if max_cache_entries > 0 else None


    def register_adapter(self, name: str, path: str):
        """
        Load adapter weights from safetensors file into CPU memory.
        
        Expected format: {layer_name}.lora_A and {layer_name}.lora_B tensors
        """
        tensors = load_file(path)
        layer_names = [k[:-7] for k in tensors.keys() if k.endswith(".lora_A")]

        self.registered_adapters[name] = {
            layer: (tensors[f"{layer}.lora_A"], tensors[f"{layer}.lora_B"])
            for layer in layer_names
        }

    def register_new_adapter(self, name: str):
        """
        Create a fresh adapter with default initialization (on CPU).
        
        A initialized with Kaiming uniform, B initialized to zeros.
        Use this when starting finetuning for a new disease area without pre-existing weights.
        """
        from batched_lora import BatchedLoRALinear
        
        adapter_weights = {}
        for module_name, module in self.model.named_modules():
            if isinstance(module, BatchedLoRALinear):
                adapter_weights[module_name] = module.create_fresh_adapter()
        
        self.registered_adapters[name] = adapter_weights

    @staticmethod
    def _save_adapter_dict(adapter: Dict[str, Tuple[Tensor, Tensor]], path: str):
        """Save adapter dict to safetensors format. Low-level utility."""
        tensors = {}
        for layer_name, (A, B) in adapter.items():
            tensors[f"{layer_name}.lora_A"] = A
            tensors[f"{layer_name}.lora_B"] = B
        save_file(tensors, path)

    def save_adapter(self, name: str, path: str):
        """Save adapter weights to safetensors format. Syncs from GPU if needed."""
        # Sync from GPU if this adapter is currently active
        if name in self._adapter_name_to_idx:
            self._sync_adapters_to_cpu([name])
        
        self._save_adapter_dict(self.registered_adapters[name], path)

    def _sync_adapters_to_cpu(self, names: List[str]):
        """Pull current GPU adapter weights back to registered_adapters (CPU)."""
        if not self.current_adapters:
            return
        
        # For each layer, extract weights for requested adapters
        for layer_name in next(iter(self.registered_adapters.values())).keys():
            layer = self.model.get_submodule(layer_name)
            layer_adapters = layer.get_adapters()
            
            for name in names:
                if name in self._adapter_name_to_idx:
                    idx = self._adapter_name_to_idx[name]
                    A, B = layer_adapters[idx]
                    self.registered_adapters[name][layer_name] = (A.cpu(), B.cpu())

    @contextmanager
    def training_mode(self, adapter_names: List[str]):
        """
        Context manager for finetuning adapters.
        
        Disables cache during training. On exit, syncs GPU weights to CPU
        and invalidates cache entries for trained adapters.

        Args:
            names: Existing adapters
        """
        original_cache = self.cache
        self.cache = None
        
        try:
            yield self
        finally:
            self._sync_adapters_to_cpu(adapter_names)
            self.cache = original_cache
            if self.cache is not None:
                for name in adapter_names:
                    self.cache.invalidate_adapter(name)

    def set_adapters(self, names: str | List[str]) -> Tuple[float, Dict[str, int]]:
        """
        Set active adapter(s) on all BatchedLoRALinear layers.
        
        Args:
            names: Single adapter name or list of adapter names
        
        Returns:
            (elapsed_time, adapter_name_to_idx mapping)
        """
        start_time = time.time()

        if isinstance(names, str):
            names = [names]

        names_set = frozenset(names)

        # Skip if adapters unchanged
        if names_set == self.current_adapters:
            return time.time() - start_time, self._adapter_name_to_idx

        ordered_names = list(names_set)
        adapter_name_to_idx = {name: i for i, name in enumerate(ordered_names)}

        for layer_name in next(iter(self.registered_adapters.values())).keys():
            layer = self.model.get_submodule(layer_name)
            adapters_for_layer = [
                self.registered_adapters[name][layer_name] for name in ordered_names
            ]
            layer.set_adapters(adapters_for_layer)

        self.current_adapters = names_set
        self._adapter_name_to_idx = adapter_name_to_idx
        return time.time() - start_time, adapter_name_to_idx

    def forward(
        self,
        x: Int[Tensor, "batch seq"],
        adapter_name: Optional[str] = None,
    ) -> Float[Tensor, "batch seq hidden"]:
        """Single-adapter forward pass."""
        if adapter_name is not None:
            self.set_adapters(adapter_name)
        output = self.model(x)
        if hasattr(output, 'last_hidden_state'):
            return output.last_hidden_state
        return output

    def _run_batched_inference(
        self,
        x: Int[Tensor, "batch seq"],
        adapter_ids: List[str],
    ) -> Float[Tensor, "batch seq hidden"]:
        """Run batched LoRA inference without caching."""
        unique_adapters = list(set(adapter_ids))
        _, adapter_name_to_idx = self.set_adapters(unique_adapters)

        adapter_idx_tensor = torch.tensor(
            [adapter_name_to_idx[name] for name in adapter_ids],
            device=x.device,
        )

        for layer_name in next(iter(self.registered_adapters.values())).keys():
            layer = self.model.get_submodule(layer_name)
            layer.set_adapter_ids(adapter_idx_tensor)

        output = self.model(x)
        # Extract tensor from ModelOutput (works for BertModel, etc.)
        if hasattr(output, 'last_hidden_state'):
            return output.last_hidden_state
        return output

    def forward_multi(
        self,
        x: Int[Tensor, "batch seq"],
        adapter_ids: List[str],
    ) -> Float[Tensor, "batch seq hidden"]:
        """
        Multi-adapter forward pass with per-sample adapter selection.
        
        Args:
            x: (batch, seq) token IDs
            adapter_ids: List of adapter names, one per sample. Must have been registered beforehand
        
        Returns:
            (batch, seq, hidden) hidden states
        """
        if self.cache is None:
            return self._run_batched_inference(x, adapter_ids)

        # Check cache for each sample
        results: List[Optional[Tensor]] = [None] * len(adapter_ids)
        uncached_indices = []

        for i, (token_ids, adapter) in enumerate(zip(x, adapter_ids)):
            cached = self.cache.get(token_ids, adapter)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)

        # Run inference only on cache misses
        if uncached_indices:
            uncached_indices_tensor = torch.tensor(uncached_indices, device=x.device)
            uncached_x = x[uncached_indices_tensor]
            uncached_adapters = [adapter_ids[i] for i in uncached_indices]

            uncached_outputs = self._run_batched_inference(uncached_x, uncached_adapters)

            for idx, orig_i in enumerate(uncached_indices):
                self.cache.put(x[orig_i], adapter_ids[orig_i], uncached_outputs[idx])
                results[orig_i] = uncached_outputs[idx]

        return torch.stack(results)


class AdapterCache:
    """LRU cache for adapter inference results, keyed by (input_hash, adapter_name).
    Input hashes are computed based on token_id sequences for performance reasons"""

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self.cache: OrderedDict[Tuple[int, str], Float[Tensor, "seq hidden"]] = OrderedDict()

    def _make_key(self, input_ids: Int[Tensor, "seq"], adapter: str) -> Tuple[int, str]:
        return (hash(tuple(input_ids.tolist())), adapter)

    def get(
        self, input_ids: Int[Tensor, "seq"], adapter: str
    ) -> Optional[Float[Tensor, "seq hidden"]]:
        key = self._make_key(input_ids, adapter)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(
        self,
        input_ids: Int[Tensor, "seq"],
        adapter: str,
        embedding: Float[Tensor, "seq hidden"],
    ):
        key = self._make_key(input_ids, adapter)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_entries:
                self.cache.popitem(last=False)  # Evict LRU
            self.cache[key] = embedding

    def clear(self):
        self.cache.clear()

    def invalidate_adapter(self, adapter: str):
        """Remove all cache entries for a specific adapter."""
        keys_to_remove = [k for k in self.cache if k[1] == adapter]
        for key in keys_to_remove:
            del self.cache[key]
