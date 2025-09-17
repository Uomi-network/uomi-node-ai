import torch
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

"""
Paged / Block KV scaffolding.

Goal: Enable true rolling admission without full recompute by allocating fixed-size token blocks
per sequence and gathering views for attention. This initial version only provides data structures;
no integration with attention yet.
"""

@dataclass
class PagedKVConfig:
    block_size: int = 16
    max_blocks: int = 8192  # total blocks per layer
    dtype: torch.dtype = torch.float16

@dataclass
class SequenceKV:
    seq_id: str
    blocks: List[int] = field(default_factory=list)  # block indices
    lengths: List[int] = field(default_factory=list)  # filled tokens per block (last block partially filled)
    total_tokens: int = 0

    def append_token(self):
        self.total_tokens += 1
        if not self.blocks:
            return
        # increment last block fill count
        last_idx = len(self.lengths) - 1
        self.lengths[last_idx] += 1

class BlockAllocator:
    def __init__(self, n_layers: int, n_heads: int, head_dim: int, device: str, config: Optional[PagedKVConfig] = None):
        self.cfg = config or PagedKVConfig()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        # Allocate K/V storage per layer: [total_blocks, n_heads, block_size, head_dim]
        shape = (self.cfg.max_blocks, n_heads, self.cfg.block_size, head_dim)
        self.keys = [torch.empty(shape, dtype=self.cfg.dtype, device=self.device) for _ in range(n_layers)]
        self.values = [torch.empty(shape, dtype=self.cfg.dtype, device=self.device) for _ in range(n_layers)]
        self.free_list = list(range(self.cfg.max_blocks))
        self.block_refcount = [0] * self.cfg.max_blocks

    def allocate_block(self) -> int:
        if not self.free_list:
            raise RuntimeError("Out of KV blocks")
        idx = self.free_list.pop()
        self.block_refcount[idx] = 1
        return idx

    def free_blocks(self, indices: List[int]):
        for i in indices:
            self.block_refcount[i] = 0
            self.free_list.append(i)

    def store_prefill(self, layer: int, block_idx: int, k_slice: torch.Tensor, v_slice: torch.Tensor):
        # k_slice/v_slice shapes: [n_heads, slice_len, head_dim]; slice_len <= block_size
        bs = k_slice.shape[1]
        self.keys[layer][block_idx, :, :bs, :] = k_slice
        self.values[layer][block_idx, :, :bs, :] = v_slice

    def store_token(self, layer: int, block_idx: int, offset: int, k_tok: torch.Tensor, v_tok: torch.Tensor):
        # k_tok/v_tok shapes: [n_heads, head_dim]
        self.keys[layer][block_idx, :, offset, :] = k_tok
        self.values[layer][block_idx, :, offset, :] = v_tok

    def gather(self, layer: int, seq: SequenceKV) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return concatenated K/V for a sequence (warning: O(num_blocks * block_size))
        ks = []
        vs = []
        for bi, fill in zip(seq.blocks, seq.lengths):
            ks.append(self.keys[layer][bi, :, :fill, :])
            vs.append(self.values[layer][bi, :, :fill, :])
        k = torch.cat(ks, dim=1) if ks else torch.empty((self.n_heads,0,self.head_dim), device=self.device, dtype=self.cfg.dtype)
        v = torch.cat(vs, dim=1) if vs else torch.empty((self.n_heads,0,self.head_dim), device=self.device, dtype=self.cfg.dtype)
        return k, v

    def stats(self):
        used = self.cfg.max_blocks - len(self.free_list)
        return {"used_blocks": used, "free_blocks": len(self.free_list), "capacity_tokens": self.cfg.max_blocks * self.cfg.block_size}
