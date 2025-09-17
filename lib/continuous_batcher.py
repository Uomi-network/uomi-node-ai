import threading
import time
import uuid
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple, Any, cast
import torch
import os

from lib.config import BATCH_MAX_SIZE, TRANSFORMERS_INFERENCE_TEMPERATURE, TRANSFORMERS_INFERENCE_MAX_TOKENS
from lib.paged_kv import BlockAllocator, PagedKVConfig, SequenceKV


class SequenceState:
    __slots__ = [
        'id', 'prompt_input_ids', 'generated_ids', 'past_key_values', 'finished', 'max_new_tokens', 'tokens_generated',
        'sampling_cfg', 'enable_thinking', 'prob_tokens', 'last_emit_index', 'arrival_time', 'on_token', 'on_complete', 'is_check', 'forced_tokens', 'check_failed', 'stream_tokens_text'
    ]

    def __init__(self, sid, prompt_input_ids, max_new_tokens, sampling_cfg, enable_thinking, on_token, on_complete, is_check=False, forced_tokens=None):
        self.id = sid
        self.prompt_input_ids = prompt_input_ids  # tensor [prompt_len]
        self.generated_ids: List[int] = []
        self.past_key_values = None  # per-sequence past when using paged/grouped mode (tuple of layer tuples)
        self.finished = False
        self.max_new_tokens = max_new_tokens
        self.tokens_generated = 0
        self.sampling_cfg = sampling_cfg
        self.enable_thinking = enable_thinking
        self.prob_tokens: List[Dict] = []
        self.last_emit_index = 0
        self.arrival_time = time.time()
        self.on_token = on_token
        self.on_complete = on_complete
        self.is_check = is_check
        self.forced_tokens = forced_tokens or []  # list of token ids for verification path
        self.check_failed = False
        self.stream_tokens_text: List[str] = []  # raw per-token decoded (skip_special_tokens=True) for fallback


class ContinuousBatcher:
    def __init__(self, model, tokenizer, device, max_active: int = BATCH_MAX_SIZE):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_active = max_active
        self.pending: Deque[SequenceState] = deque()
        self.active: List[SequenceState] = []
        self.lock = threading.Lock()
        self.kill = False
        # Batched KV cache (disabled in rolling mode)
        self._batched_past = None
        self._decoding_started = False
        self.rolling_mode = os.getenv("CONTINUOUS_ROLLING", "0") == "1"
        self.use_paged_kv = os.getenv("USE_PAGED_KV", "1") == "1"
        self._paged_allocator = None
        self._seq_kv_map = {}  # seq_id -> SequenceKV
        if self.use_paged_kv:
            # Infer model dims (supports MHA + MQA/GQA: use key/value head count if present)
            try:
                n_layers = len(model.model.layers)
                attn = model.model.layers[0].self_attn

                def _probe(module, names):
                    for name in names:
                        if hasattr(module, name):
                            val = getattr(module, name)
                            if isinstance(val, (int, float)):
                                return int(val)
                    return None

                n_attn_heads = _probe(attn, ['num_heads','n_heads','n_head','num_attention_heads'])
                if n_attn_heads is None and hasattr(model.config, 'num_attention_heads'):
                    n_attn_heads = int(model.config.num_attention_heads)

                n_kv_heads = _probe(attn, ['num_key_value_heads','n_kv_heads','num_kv_heads'])
                if n_kv_heads is None and hasattr(model.config, 'num_key_value_heads'):
                    n_kv_heads = int(getattr(model.config, 'num_key_value_heads'))
                if n_kv_heads is None:
                    n_kv_heads = n_attn_heads  # fallback (pure MHA)

                head_dim = _probe(attn, ['head_dim','dim_head','attention_head_size'])
                if head_dim is None and hasattr(model.config, 'head_dim'):
                    head_dim = int(model.config.head_dim)
                if head_dim is None and hasattr(model.config, 'hidden_size') and n_attn_heads:
                    head_dim = int(model.config.hidden_size) // int(n_attn_heads)

                if (n_kv_heads is None) or (head_dim is None):
                    raise ValueError("Unable to infer KV head dims for paged KV")
                cfg = PagedKVConfig(
                    block_size=int(os.getenv("PAGED_BLOCK_SIZE", "16")),
                    max_blocks=int(os.getenv("PAGED_MAX_BLOCKS", "4096")),
                    dtype=torch.float16 if device == 'cuda' else torch.float32
                )
                self._paged_allocator = BlockAllocator(n_layers, n_kv_heads, head_dim, device, cfg)
                print(f"[paged-kv] enabled block_size={cfg.block_size} max_blocks={cfg.max_blocks} kv_heads={n_kv_heads} head_dim={head_dim}")
            except Exception as e:
                print(f"[paged-kv] failed to init: {e}; falling back to dense path")
                self.use_paged_kv = False
        # Staged rolling admission (consolidation) parameters
        self._needs_consolidation = False
        self._steps_since_last_admission = 0
        try:
            self._consolidation_interval = max(1, int(os.getenv("ROLLING_INTERVAL_STEPS", "8")))
        except ValueError:
            self._consolidation_interval = 8
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def status(self):
        with self.lock:
            return {
                'pending': len(self.pending),
                'active': len(self.active),
                'decoding_started': self._decoding_started,
                'needs_consolidation': self._needs_consolidation,
                'steps_since_last_admission': self._steps_since_last_admission,
                'rolling_mode': self.rolling_mode,
                'use_paged_kv': self.use_paged_kv
            }

    def stop(self):
        self.kill = True
        self.thread.join()

    def submit(self, messages, enable_thinking: bool, sampling_cfg: Dict, max_new_tokens: Optional[int], on_token: Callable, on_complete: Callable, is_check: bool=False, forced_tokens: Optional[List[int]]=None):
        sid = uuid.uuid4().hex
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        prompt_ids = self.tokenizer(prompt_text, return_tensors='pt').input_ids[0].to(self.device)
        seq = SequenceState(sid, prompt_ids, max_new_tokens or TRANSFORMERS_INFERENCE_MAX_TOKENS, sampling_cfg, enable_thinking, on_token, on_complete, is_check=is_check, forced_tokens=forced_tokens)
        with self.lock:
            self.pending.append(seq)
        return sid

    # Internal methods
    def _admit_newcomers(self):
        if not self.pending:
            return []
        admitted = []
        # In incremental KV mode we queue newcomers until consolidation point
        if self._decoding_started and not self.rolling_mode and not self.use_paged_kv:
            self._needs_consolidation = True
            return []
        while self.pending and len(self.active) + len(admitted) < self.max_active:
            admitted.append(self.pending.popleft())
        if not admitted:
            return []

        # True rolling admission path: per-sequence independent caches
        if self.use_paged_kv:
            for seq in admitted:
                with torch.no_grad():
                    out = self.model(input_ids=seq.prompt_input_ids.unsqueeze(0), use_cache=True)
                logits = out.logits[0]
                seq.past_key_values = out.past_key_values
                # Block allocator integration (prefill copy)
                if self._paged_allocator is not None:
                    # Allocate required blocks for this prefill length
                    prefill_len = logits.shape[0]  # number of tokens including prompt; actually need prompt len
                    prompt_len = seq.prompt_input_ids.size(0)
                    block_size = self._paged_allocator.cfg.block_size
                    n_blocks = (prompt_len + block_size - 1) // block_size
                    seq_kv = SequenceKV(seq.id)
                    for b in range(n_blocks):
                        blk = self._paged_allocator.allocate_block()
                        seq_kv.blocks.append(blk)
                        # Determine slice range inside this block
                        start = b * block_size
                        end = min(start + block_size, prompt_len)
                        slice_len = end - start
                        seq_kv.lengths.append(slice_len)
                    seq_kv.total_tokens = prompt_len
                    # Copy layer K/V into blocks
                    for layer_idx, (k, v) in enumerate(out.past_key_values):
                        # Normalize shapes. Common cases:
                        # 1. [1, n_heads, seq_len, head_dim]
                        # 2. [n_heads, seq_len, head_dim]
                        # 3. Merged heads: [seq_len, hidden] (rare in past cache)
                        # 4. Gemma style may already be tuple of tensors with required shape
                        def _norm(t: torch.Tensor):
                            if self._paged_allocator is None:
                                return t  # should not happen
                            if t.dim() == 4:  # [1, n_h, L, D]
                                return t[0]
                            if t.dim() == 3:  # [n_h, L, D]
                                return t
                            if t.dim() == 2:  # [L, hidden] -> attempt split
                                nh = self._paged_allocator.n_heads
                                L = t.shape[0]
                                hidden = t.shape[1]
                                if hidden % nh != 0:
                                    raise ValueError("Cannot split merged head dimension")
                                hd = hidden // nh
                                return t.view(L, nh, hd).permute(1,0,2)  # [n_h, L, D]
                            raise ValueError("Unexpected past key/value tensor rank")
                        k_layer = _norm(k)
                        v_layer = _norm(v)
                        if k_layer.shape[0] != self._paged_allocator.n_heads:
                            # If GQA (kv heads < attn heads), we only store kv heads subset; slice
                            k_layer = k_layer[:self._paged_allocator.n_heads]
                            v_layer = v_layer[:self._paged_allocator.n_heads]
                        seq_len_layer = k_layer.shape[1]
                        if seq_len_layer < prompt_len:  # safety
                            prompt_len = seq_len_layer
                        cursor = 0
                        for bi, fill in zip(seq_kv.blocks, seq_kv.lengths):
                            if fill == 0:
                                continue
                            sl = k_layer[:, cursor:cursor+fill, :]
                            vl = v_layer[:, cursor:cursor+fill, :]
                            self._paged_allocator.store_prefill(layer_idx, bi, sl, vl)
                            cursor += fill
                    self._seq_kv_map[seq.id] = seq_kv
                # First token sample
                self._advance_sequence(seq, logits[-1, :])
                # Append generated token to cache (only if not finished)
                if not seq.finished:
                    with torch.no_grad():
                        next_input = torch.tensor([[seq.generated_ids[-1]]], dtype=torch.long, device=self.device)
                        out2 = self.model(input_ids=next_input, past_key_values=seq.past_key_values, use_cache=True)
                    seq.past_key_values = out2.past_key_values
            self.active.extend(admitted)
            self._steps_since_last_admission = 0
            return admitted

        # Batch prefill (capture KV cache unless rolling mode recompute path) for dense path
        if admitted:
            batch_ids = [s.prompt_input_ids for s in admitted]
            max_len = max(t.size(0) for t in batch_ids)
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            padded = []
            attn = []
            for t in batch_ids:
                pad_len = max_len - t.size(0)
                if pad_len:
                    pad_tensor = torch.full((pad_len,), pad_id, dtype=t.dtype, device=self.device)
                    if self.tokenizer.padding_side == 'left':
                        padded.append(torch.cat([pad_tensor, t]))
                        attn.append(torch.cat([torch.zeros(pad_len, dtype=torch.long, device=self.device), torch.ones_like(t)]))
                    else:
                        padded.append(torch.cat([t, pad_tensor]))
                        attn.append(torch.cat([torch.ones_like(t), torch.zeros(pad_len, dtype=torch.long, device=self.device)]))
                else:
                    padded.append(t)
                    attn.append(torch.ones_like(t))
            input_ids = torch.stack(padded)
            attention_mask = torch.stack(attn)
            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.rolling_mode or self.use_paged_kv)
            logits = out.logits  # [B, L, V]
            if not self.rolling_mode and not self.use_paged_kv:
                self._batched_past = out.past_key_values
            if self.use_paged_kv:
                # Store per-sequence prefill into block allocator (future step: avoid dense concat on decode)
                # For now just register empty SequenceKV placeholders
                for i, seq in enumerate(admitted):
                    self._seq_kv_map[seq.id] = SequenceKV(seq.id)
            # First token sampling for each sequence
            for i, seq in enumerate(admitted):
                last_pos = input_ids.size(1) - 1
                last_logits = logits[i, last_pos, :]
                self._advance_sequence(seq, last_logits)
            self.active.extend(admitted)
            self._steps_since_last_admission = 0
        return admitted

    def _consolidate_admissions(self):
        if not self.pending or len(self.active) >= self.max_active:
            return False
        # Build unified context for existing unfinished + pending newcomers
        newcomers = []
        while self.pending and len(self.active) + len(newcomers) < self.max_active:
            newcomers.append(self.pending.popleft())
        if not newcomers:
            return False
        # Collect contexts (prompt+generated for active unfinished; prompt only for newcomers)
        ctx_tensors = []
        active_indices = []
        for seq in self.active:
            if seq.finished:
                continue
            gen = torch.tensor(seq.generated_ids, dtype=torch.long, device=self.device) if seq.generated_ids else torch.empty(0, dtype=torch.long, device=self.device)
            ctx_tensors.append(torch.cat([seq.prompt_input_ids, gen]))
            active_indices.append(seq.id)
        newcomer_indices = []
        for n in newcomers:
            ctx_tensors.append(n.prompt_input_ids)
            newcomer_indices.append(n.id)
        if not ctx_tensors:
            return False
        max_len = max(t.size(0) for t in ctx_tensors)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        padded = []
        attn = []
        for t in ctx_tensors:
            pad_len = max_len - t.size(0)
            if pad_len:
                pad_tensor = torch.full((pad_len,), pad_id, dtype=t.dtype, device=self.device)
                if self.tokenizer.padding_side == 'left':
                    padded.append(torch.cat([pad_tensor, t]))
                    attn.append(torch.cat([torch.zeros(pad_len, dtype=torch.long, device=self.device), torch.ones_like(t)]))
                else:
                    padded.append(torch.cat([t, pad_tensor]))
                    attn.append(torch.cat([torch.ones_like(t), torch.zeros(pad_len, dtype=torch.long, device=self.device)]))
            else:
                padded.append(t)
                attn.append(torch.ones_like(t))
        input_ids = torch.stack(padded)
        attention_mask = torch.stack(attn)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        logits = out.logits
        self._batched_past = out.past_key_values
        # Sample first token for newcomers only
        last_pos = input_ids.size(1) - 1
        # Active sequences keep their existing generated_ids; we do not advance them here
        offset_newcomers = len(active_indices)
        for i, seq in enumerate(newcomers):
            self._advance_sequence(seq, logits[offset_newcomers + i, last_pos, :])
        self.active.extend(newcomers)
        self._needs_consolidation = False
        self._steps_since_last_admission = 0
        return True

    def _advance_sequence(self, seq: SequenceState, logits):
        if seq.finished:
            return
        max_seconds = float(os.getenv("CONTINUOUS_MAX_SECONDS", "3600"))
        if (time.time() - seq.arrival_time) > max_seconds:
            seq.finished = True
            if seq.on_complete:
                print(f"[continuous] timeout seq={seq.id} after {max_seconds} seconds")
                response = self.tokenizer.decode(seq.generated_ids, skip_special_tokens=True)
                seq.on_complete(seq.id, response, {'tokens': seq.prob_tokens, 'full_sequence_length': len(seq.prompt_input_ids) + len(seq.generated_ids), 'timeout': True})
            return
        # Sampling / forced token
        if seq.is_check and seq.tokens_generated < len(seq.forced_tokens):
            next_token_id = seq.forced_tokens[seq.tokens_generated]
            probs = torch.softmax(logits, dim=-1)
            token_prob = probs[next_token_id].item() if next_token_id < probs.size(0) else 0.0
            top_probs, top_idx = probs.topk(10)
            rank = -1
            for r, idx in enumerate(top_idx):
                if idx.item() == int(next_token_id):
                    rank = r
                    break
            if rank == -1:
                seq.check_failed = True
        else:
            probs = torch.softmax(logits, dim=-1)
            if seq.sampling_cfg.get('deterministic'):
                next_token_id = int(torch.argmax(probs).item())
                token_prob = probs[next_token_id].item()
                rank = 0
            else:
                top_k = seq.sampling_cfg.get('top_k', 5)
                temperature = seq.sampling_cfg.get('temperature', TRANSFORMERS_INFERENCE_TEMPERATURE)
                working_logits = logits / max(temperature, 1e-6)
                if top_k:
                    kth = torch.topk(working_logits, top_k).values[-1]
                    working_logits = torch.where(working_logits >= kth, working_logits, torch.full_like(working_logits, -float('inf')))
                probs = torch.softmax(working_logits, dim=-1)
                next_token_id = int(torch.multinomial(probs, num_samples=1).item())
                token_prob = probs[next_token_id].item()
                # Rank among top_k (approx)
                top_probs, top_idx = probs.topk(min(10, probs.size(0)))
                rank = -1
                for r, idx in enumerate(top_idx):
                    if idx.item() == next_token_id:
                        rank = r
                        break
        seq.generated_ids.append(next_token_id)
        seq.prob_tokens.append({'id': next_token_id, 'prob': token_prob, 'index': rank})
        seq.tokens_generated += 1
        # Emit token (stream)
        if seq.on_token:
            try:
                txt = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
                seq.stream_tokens_text.append(txt)
                seq.on_token(seq.id, txt, {'id': next_token_id, 'prob': token_prob, 'rank': rank})
            except Exception:
                seq.stream_tokens_text.append("")
        # Termination
        repetitive_guard = False
        # Improved repetitive guard: only trigger if sufficiently long and no alphabetical output recently
        if not seq.finished and seq.tokens_generated >= 24:
            window = seq.stream_tokens_text[-12:]
            has_alpha = any(any(ch.isalpha() for ch in tok) for tok in window if tok)
            if not has_alpha:
                repetitive_guard = True
        if next_token_id == self.tokenizer.eos_token_id or seq.tokens_generated >= seq.max_new_tokens or repetitive_guard or (seq.is_check and (seq.check_failed or seq.tokens_generated >= len(seq.forced_tokens))):
            seq.finished = True
            if seq.on_complete:
                if seq.is_check and seq.check_failed:
                    seq.on_complete(seq.id, "", {'tokens': seq.prob_tokens, 'full_sequence_length': len(seq.prompt_input_ids) + len(seq.generated_ids), 'verified': False})
                else:
                    full_gen = self.tokenizer.decode(seq.generated_ids, skip_special_tokens=True)
                    if (not full_gen.strip()) and seq.stream_tokens_text:
                        # Fallback extraction: remove obvious structural tokens
                        structural_tokens = {"<","|","｜","User","Assistant","think","</think>","<think>",">","<|","|>"}
                        fallback = ''.join([t for t in seq.stream_tokens_text if t and t.strip() not in structural_tokens])
                        # Trim after last 'Assistant' occurrence if still structural prefix
                        if 'Assistant' in fallback:
                            idx = fallback.rfind('Assistant')
                            tail = fallback[idx+len('Assistant'):]
                            fallback = tail.strip() or fallback
                        full_gen = fallback.strip()
                    meta = {'tokens': seq.prob_tokens, 'full_sequence_length': len(seq.prompt_input_ids) + len(seq.generated_ids), 'verified': (not seq.is_check) or (seq.is_check and not seq.check_failed)}
                    if repetitive_guard:
                        meta['repetitive_guard'] = True
                    if not full_gen:
                        meta['empty_after_fallback'] = True
                    seq.on_complete(seq.id, full_gen, meta)

    def _decode_step(self):
        if not self.active:
            return
        if self.rolling_mode:
            # Recompute full context for all unfinished sequences (simple but allows rolling admissions)
            batch_ctx = []
            for seq in self.active:
                if seq.finished:
                    batch_ctx.append(torch.tensor([self.tokenizer.eos_token_id], device=self.device))
                else:
                    gen = torch.tensor(seq.generated_ids, dtype=torch.long, device=self.device) if seq.generated_ids else torch.empty(0, dtype=torch.long, device=self.device)
                    ctx = torch.cat([seq.prompt_input_ids, gen])
                    batch_ctx.append(ctx)
            max_len = max(t.size(0) for t in batch_ctx)
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            padded = []
            attn = []
            for t in batch_ctx:
                pad_len = max_len - t.size(0)
                if pad_len:
                    pad_tensor = torch.full((pad_len,), pad_id, dtype=t.dtype, device=self.device)
                    if self.tokenizer.padding_side == 'left':
                        padded.append(torch.cat([pad_tensor, t]))
                        attn.append(torch.cat([torch.zeros(pad_len, dtype=torch.long, device=self.device), torch.ones_like(t)]))
                    else:
                        padded.append(torch.cat([t, pad_tensor]))
                        attn.append(torch.cat([torch.ones_like(t), torch.zeros(pad_len, dtype=torch.long, device=self.device)]))
                else:
                    padded.append(t)
                    attn.append(torch.ones_like(t))
            input_ids = torch.stack(padded)
            attention_mask = torch.stack(attn)
            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = out.logits  # [B,L,V]
            last_pos = input_ids.size(1) - 1
            for i, seq in enumerate(self.active):
                if seq.finished:
                    continue
                self._advance_sequence(seq, logits[i, last_pos, :])
        else:
            # KV-cache incremental path (dense) or paged grouped decode
            if self.use_paged_kv:
                # Group sequences by current past length (seq_len distinct groups)
                groups: Dict[int, List[SequenceState]] = {}
                for seq in self.active:
                    if seq.finished or seq.past_key_values is None:
                        continue
                    seq_len = seq.past_key_values[0][0].shape[2]
                    groups.setdefault(seq_len, []).append(seq)
                for seq_len, seqs in groups.items():
                    # Skip groups where any sequence lacks initialized past (should not happen post-prefill)
                    if any(s.past_key_values is None for s in seqs):
                        continue
                    # If any past object exposes get_seq_length (e.g., Gemma3 Cache), fall back to per-sequence decode
                    if any(hasattr(s.past_key_values, 'get_seq_length') for s in seqs):
                        for s in seqs:
                            if s.finished or s.past_key_values is None:
                                continue
                            last_token = s.generated_ids[-1] if s.generated_ids else self.tokenizer.eos_token_id
                            input_ids = torch.tensor([[last_token]], dtype=torch.long, device=self.device)
                            with torch.no_grad():
                                out = self.model(input_ids=input_ids, past_key_values=s.past_key_values, use_cache=True)
                            s.past_key_values = out.past_key_values  # preserve original cache object
                            if not s.finished:
                                self._advance_sequence(s, out.logits[0, -1, :])
                        continue
                    input_ids = [s.generated_ids[-1] if s.generated_ids else self.tokenizer.eos_token_id for s in seqs]
                    input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(1)
                    # Stack per layer past along batch dim
                    stacked = []
                    n_layers = len(seqs[0].past_key_values) if seqs[0].past_key_values is not None else 0
                    for l in range(n_layers):
                        ks = []
                        vs = []
                        for s in seqs:
                            pkv = s.past_key_values
                            if pkv is None:
                                continue
                            layer_tuple = cast(Tuple[Any, Any], pkv[l])  # (k,v)
                            k, v = layer_tuple
                            ks.append(k)
                            vs.append(v)
                        stacked.append((torch.cat(ks, dim=0), torch.cat(vs, dim=0)))
                    with torch.no_grad():
                        out = self.model(input_ids=input_ids, past_key_values=tuple(stacked), use_cache=True)
                    new_past = out.past_key_values
                    logits_batch = out.logits[:, -1, :]
                    # Split back
                    for b_idx, s in enumerate(seqs):
                        per_seq = []
                        for l in range(n_layers):
                            k_full, v_full = new_past[l]
                            per_seq.append((k_full[b_idx:b_idx+1, ...], v_full[b_idx:b_idx+1, ...]))
                        s.past_key_values = tuple(per_seq)  # type: ignore
                        if not s.finished:
                            self._advance_sequence(s, logits_batch[b_idx])
                self._decoding_started = True
                self._steps_since_last_admission += 1
                return
            # Dense shared past path
            last_tokens = []
            for seq in self.active:
                if seq.generated_ids:
                    last_tokens.append(seq.generated_ids[-1])
                else:
                    last_tokens.append(self.tokenizer.eos_token_id)
            input_ids = torch.tensor(last_tokens, dtype=torch.long, device=self.device).unsqueeze(1)
            with torch.no_grad():
                out = self.model(input_ids=input_ids, past_key_values=self._batched_past, use_cache=True)
            self._batched_past = out.past_key_values
            logits_batch = out.logits[:, -1, :]
            for i, seq in enumerate(self.active):
                if seq.finished:
                    continue
                self._advance_sequence(seq, logits_batch[i])
            self._decoding_started = True
            self._steps_since_last_admission += 1

    def _cleanup_finished(self):
        if not self.active:
            return
        if self.use_paged_kv:
            # Remove finished sequences immediately to keep groups small
            self.active = [s for s in self.active if not s.finished]
            if not self.active:
                self._decoding_started = False
            return
        if all(s.finished for s in self.active):
            self.active = []
            self._batched_past = None
            self._decoding_started = False

    def _loop(self):
        idle_sleep = 0.01
        debug = os.getenv("CONTINUOUS_DEBUG", "0") == "1"
        while not self.kill:
            made_progress = False
            with self.lock:
                if len(self.active) < self.max_active and self.pending:
                    # In rolling or per-sequence (paged_kv) mode we can always admit immediately
                    if not self._decoding_started or self.rolling_mode or self.use_paged_kv:
                        admitted = self._admit_newcomers()
                        if debug and admitted:
                            print(f"[continuous] admitted_initial count={len(admitted)} pending={len(self.pending)} active={len(self.active)}")
                        made_progress = True
                    else:
                        # staged consolidation path
                        # If we explicitly attempted admission earlier we may wait a few steps
                        # but if capacity was freed (finished sequences) we should consolidate immediately
                        consolidate_now = False
                        if self._needs_consolidation:
                            if self._steps_since_last_admission >= self._consolidation_interval:
                                consolidate_now = True
                            else:
                                # Heuristic: if there is free capacity because sequences finished, don't delay
                                if len(self.active) < self.max_active:
                                    consolidate_now = True
                        else:
                            # We have pending + capacity but decoding already started; mark and consolidate immediately
                            self._needs_consolidation = True
                            consolidate_now = True
                        if consolidate_now:
                            if self._consolidate_admissions():
                                if debug:
                                    print(f"[continuous] consolidated pending={len(self.pending)} active={len(self.active)}")
                                made_progress = True
            # Decode step (no lock during forward)
            if self.active:
                self._decode_step()
                made_progress = True
            self._cleanup_finished()
            # Post-cleanup fast path: if capacity freed and pending remain, try immediate consolidation/admission
            if not self.kill:
                with self.lock:
                    if self.pending and len(self.active) < self.max_active:
                        if not self._decoding_started or self.rolling_mode or self.use_paged_kv:
                            admitted = self._admit_newcomers()
                            if debug and admitted:
                                print(f"[continuous] admitted_post_cleanup count={len(admitted)} pending={len(self.pending)} active={len(self.active)}")
                                made_progress = True
                        else:
                            self._needs_consolidation = True
                            if self._consolidate_admissions():
                                if debug:
                                    print(f"[continuous] consolidated_post_cleanup pending={len(self.pending)} active={len(self.active)}")
                                made_progress = True
            if not made_progress:
                time.sleep(idle_sleep)