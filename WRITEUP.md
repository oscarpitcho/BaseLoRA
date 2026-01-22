# LoRA for Geneformer

Model: `gf-20L-95M-i4096` (BertForMaskedLM, 20 transformer layers, hidden_dim=512, intermediate=2048)

## Linear Layer Count

| Location | Layers per Block | Blocks | Total |
|----------|------------------|--------|-------|
| Q, K, V projections | 3 | 20 | 60 |
| Attention output | 1 | 20 | 20 |
| Intermediate (FFN) | 1 | 20 | 20 |
| Output (FFN) | 1 | 20 | 20 |
| MLM head | 2 | 1 | 2 |
| **Total** | | | **122** |

## Standard LoRA

Applies one adapter at a time. For batches with multiple adapters, we split it into subsets with a single adapter and run separate forward passes. 

**Problem:** A batch with 50 different adapters requires 50 sync points per batch — significant overhead as switching adapters creates a CPU / GPU sync. We implement batched LoRA as a solution to this.

## Batched LoRA

Multiple adapters are pre-loaded onto GPU as stacked tensors (`A_all`, `B_all`). At forward time, only indices are provided (e.g., `[0, 0, 0, 1, 1, 1, 0, 0]`) — the layer indexes into the stacked adapters per-sample. Not all pre-loaded adapters need to be used in every forward pass.

**Benefit:** Pre-store adapters from upcoming batches on GPU. No CPU/GPU sync when switching between them — just change the index tensor.

**Tradeoff:** Requires setting adapter indices before each forward call (per-forward state). This isn't clean, but we can't easily modify BERT's forward interface, so it's a necessary measure.

### Memory per Adapter

Each LoRA layer adds `A: (in, r)` and `B: (r, out)` matrices.  
Parameters per layer: `r × (in + out)`

**All 122 layers:** `~211,000 × r parameters`
- Attention (80 layers): 80 × r(512+512) = 81,920r
- FFN (40 layers): 40 × r(512+2048) = 102,400r
- MLM head: r(512+512) + r(512+25426) = 26,962r

Rank `r = 8`: ~1.7M params (~6.8 MB float32)

Rank `r = 16`: ~3.4M params (~13.5 MB float32)

**Attention only (Q, K, V, O):** 81,920 × r parameters
- Rank `r = 8`: ~655K params (~2.6 MB float32)
- Rank `r = 16`: ~1.3M params (~5.2 MB float32)

Adapters are ~1-2% of base model size (~380 MB).

## Embedding Cache

For repeated (input, adapter) pairs, we cache the output embeddings to skip redundant computation. Uses LRU eviction.

**Memory per entry:** ~2 KB (embedding tensor: `hidden_dim × 4` = 512 × 4 = 2048 bytes). 100K entries ≈ 200 MB — about half the model size. Marginal amount to store in RAM.


# TODO: 
- Targetted injection