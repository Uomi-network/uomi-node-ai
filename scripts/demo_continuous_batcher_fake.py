import os
import sys
import time
import torch
from typing import List, Dict

# Ensure project root on path for `lib` imports when run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.continuous_batcher import ContinuousBatcher


class FakeTokenizer:
    def __init__(self):
        self.vocab = {ch: i+5 for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz")}
        self.inv = {v: k for k, v in self.vocab.items()}
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.pad_token = '<pad>'
        self.eos_token = '<eos>'
        self.bos_token = '<bos>'
        self.padding_side = 'left'

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        # Very naive: just concatenate contents
        txt = ''.join(m['content'] for m in messages if m['role'] == 'user')
        return txt

    def __call__(self, text, return_tensors=None):
        ids = [self.vocab.get(ch, 4) for ch in text.lower() if ch.isalpha()][:32]
        if not ids:
            ids = [4]
        return type('R', (), {'input_ids': torch.tensor([ids], dtype=torch.long)})

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for i in ids:
            if i == self.eos_token_id:
                continue
            out.append(self.inv.get(int(i), '?'))
        return ''.join(out)


class FakeModel(torch.nn.Module):
    def __init__(self, vocab_size=64, hidden=32, n_layers=2):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, hidden)
        self.ln = torch.nn.LayerNorm(hidden)
        self.ff = torch.nn.Linear(hidden, vocab_size)
        self._device = 'cpu'
    # mimic minimal HF-like structure with attention head metadata
        self.model = type('M', (), {'layers':[type('L', (), {'self_attn': type('A', (), {'num_heads':4,'head_dim':8})()})() for _ in range(n_layers)]})()
        self.config = type('C', (), {'num_attention_heads':4, 'hidden_size': hidden})()

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=True):
        # Very small toy causal model ignoring masking
        if input_ids is None:
            raise ValueError("input_ids required")
        x = self.emb(input_ids)
        x = self.ln(x)
        logits = self.ff(x)
        # Build fake incremental cache: past tensors [B, n_heads, seq_len, head_dim]
        new_past = None
        if use_cache:
            B, L = input_ids.size(0), input_ids.size(1)
            n_heads = 4
            head_dim = 8
            device = input_ids.device
            if past_key_values is None:
                layer_count = len(self.model.layers) if hasattr(self.model, 'layers') else 2  # type: ignore[attr-defined]
                new_past = tuple((torch.randn(B, n_heads, L, head_dim, device=device), torch.randn(B, n_heads, L, head_dim, device=device)) for _ in range(layer_count))
            else:
                # append
                new_past = []
                for l, (k, v) in enumerate(past_key_values):
                    add_k = torch.randn(k.size(0), k.size(1), 1, k.size(3), device=k.device)
                    add_v = torch.randn(v.size(0), v.size(1), 1, v.size(3), device=v.device)
                    new_past.append((torch.cat([k, add_k], dim=2), torch.cat([v, add_v], dim=2)))
                new_past = tuple(new_past)
        return type('O', (), {'logits': logits, 'past_key_values': new_past})


def main():
    os.environ.setdefault('CONTINUOUS_DEBUG', '1')
    tokenizer = FakeTokenizer()
    model = FakeModel()
    batcher = ContinuousBatcher(model, tokenizer, 'cpu', max_active=3)

    results: Dict[str, Dict] = {}

    def on_token(sid, txt, meta):
        print(f"token sid={sid[:6]} t={txt} prob={meta['prob']:.3f}")

    def on_complete(sid, response, proof):
        results[sid] = {'response': response, 'len': len(proof['tokens']) if proof else 0}
        print(f"complete sid={sid[:6]} response={response} len={results[sid]['len']}")

    # Submit first
    sid1 = batcher.submit([{'role':'user','content':'apple'}], False, {'deterministic':True}, 5, on_token, on_complete)
    time.sleep(0.05)
    # Submit second while first decoding
    sid2 = batcher.submit([{'role':'user','content':'banana'}], False, {'deterministic':True}, 5, on_token, on_complete)
    time.sleep(0.05)
    # Third
    sid3 = batcher.submit([{'role':'user','content':'citrus'}], False, {'deterministic':True}, 5, on_token, on_complete)

    deadline = time.time() + 5
    while len(results) < 3 and time.time() < deadline:
        time.sleep(0.05)

    print('FINAL STATUS', batcher.status())
    assert len(results) == 3, f"Not all sequences finished: {len(results)}"
    batcher.stop()


if __name__ == '__main__':
    main()
