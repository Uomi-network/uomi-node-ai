import threading
import time
import uuid
from collections import deque
from typing import Callable, Deque, Dict, List, Optional
import torch

from lib.config import BATCH_MAX_SIZE, TRANSFORMERS_INFERENCE_TEMPERATURE, TRANSFORMERS_INFERENCE_MAX_TOKENS


class SequenceState:
    __slots__ = [
        'id', 'prompt_input_ids', 'generated_ids', 'past_key_values', 'finished', 'max_new_tokens', 'tokens_generated',
        'sampling_cfg', 'enable_thinking', 'prob_tokens', 'last_emit_index', 'arrival_time', 'on_token', 'on_complete', 'is_check', 'forced_tokens', 'check_failed'
    ]

    def __init__(self, sid, prompt_input_ids, max_new_tokens, sampling_cfg, enable_thinking, on_token, on_complete, is_check=False, forced_tokens=None):
        self.id = sid
        self.prompt_input_ids = prompt_input_ids  # tensor [prompt_len]
        self.generated_ids: List[int] = []
        self.past_key_values = None  # will hold tuple of layer tuples
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
        # Batched KV cache for all active sequences (we avoid per-sequence slicing to retain HF cache object)
        self._batched_past = None
        self._decoding_started = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

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
        # Do not admit newcomers once decode iterations started (simplifies batched cache logic)
        if self._decoding_started:
            return []
        while self.pending and len(self.active) + len(admitted) < self.max_active:
            admitted.append(self.pending.popleft())
        # Batch prefill (capture KV cache)
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
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            logits = out.logits  # [B, L, V]
            # Store batched past (HF cache object) for all active sequences
            self._batched_past = out.past_key_values
            # First token sampling for each sequence
            for i, seq in enumerate(admitted):
                last_pos = input_ids.size(1) - 1  # safe because left padding -> last index is real token
                last_logits = logits[i, last_pos, :]
                self._advance_sequence(seq, last_logits)
            self.active.extend(admitted)
        return admitted

    def _advance_sequence(self, seq: SequenceState, logits):
        if seq.finished:
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
                seq.on_token(seq.id, txt, {'id': next_token_id, 'prob': token_prob, 'rank': rank})
            except Exception:
                pass
        # Termination
        if next_token_id == self.tokenizer.eos_token_id or seq.tokens_generated >= seq.max_new_tokens or (seq.is_check and (seq.check_failed or seq.tokens_generated >= len(seq.forced_tokens))):
            seq.finished = True
            if seq.on_complete:
                if seq.is_check and seq.check_failed:
                    seq.on_complete(seq.id, "", {'tokens': seq.prob_tokens, 'full_sequence_length': len(seq.prompt_input_ids) + len(seq.generated_ids), 'verified': False})
                else:
                    full_gen = self.tokenizer.decode(seq.generated_ids, skip_special_tokens=True)
                    seq.on_complete(seq.id, full_gen, {'tokens': seq.prob_tokens, 'full_sequence_length': len(seq.prompt_input_ids) + len(seq.generated_ids), 'verified': (not seq.is_check) or (seq.is_check and not seq.check_failed)})

    def _decode_step(self):
        if not self.active:
            return
        # Build batched input of last generated token (or eos for finished) to keep batch dims aligned
        last_tokens = []
        for seq in self.active:
            if seq.generated_ids:
                last_tokens.append(seq.generated_ids[-1])
            else:
                # Should not happen after prefill sampling; fallback to pad/eos
                last_tokens.append(self.tokenizer.eos_token_id)
        input_ids = torch.tensor(last_tokens, dtype=torch.long, device=self.device).unsqueeze(1)  # [B,1]
        with torch.no_grad():
            out = self.model(input_ids=input_ids, past_key_values=self._batched_past, use_cache=True)
        self._batched_past = out.past_key_values
        logits_batch = out.logits[:, -1, :]  # [B,V]
        for i, seq in enumerate(self.active):
            if seq.finished:
                continue
            self._advance_sequence(seq, logits_batch[i])
        self._decoding_started = True

    def _cleanup_finished(self):
        if all(s.finished for s in self.active):
            # Reset to allow new admissions next cycle
            self.active = []
            self._batched_past = None
            self._decoding_started = False
        else:
            # Keep finished sequences to preserve batch dimension until all done
            pass

    def _loop(self):
        idle_sleep = 0.01
        while not self.kill:
            made_progress = False
            with self.lock:
                if len(self.active) < self.max_active and self.pending:
                    self._admit_newcomers()
                    made_progress = True
            # Decode step (no lock during forward)
            if self.active:
                self._decode_step()
                made_progress = True
            self._cleanup_finished()
            if not made_progress:
                time.sleep(idle_sleep)