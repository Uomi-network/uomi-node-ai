import os, time, threading, statistics, argparse, json, sys, pathlib

# Ensure project root (parent of scripts/) is on sys.path so `lib` imports resolve
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.continuous_batcher import ContinuousBatcher
from lib.config import TRANSFORMERS_INFERENCE_MAX_TOKENS

# Simple synthetic chat prompt
def build_messages(i):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Give me a long fact number {i}."}
    ]

def run_requests(model, tokenizer, device, n_reqs, stagger_s, max_new_tokens, rolling_mode):
    os.environ['CONTINUOUS_ROLLING'] = '1' if rolling_mode else '0'
    batcher = ContinuousBatcher(model, tokenizer, device)
    latencies = {}
    done_event = threading.Event()
    completed = 0
    lock = threading.Lock()

    def on_token(sid, txt, meta):
        pass

    def on_complete(sid, text, meta):
        nonlocal completed
        latencies[sid]['end'] = time.time()
        with lock:
            completed += 1
            if completed == n_reqs:
                done_event.set()

    for i in range(n_reqs):
        msgs = build_messages(i)
        sid = batcher.submit(
            msgs,
            enable_thinking=False,
            sampling_cfg={'deterministic': True},
            max_new_tokens=max_new_tokens,
            on_token=on_token,
            on_complete=on_complete,
        )
        latencies[sid] = {'start': time.time()}
        time.sleep(stagger_s)

    done_event.wait()
    results = []
    for sid, t in latencies.items():
        results.append(t['end'] - t['start'])
    batcher.stop()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='google/gemma-3-1b-it')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--stagger', type=float, default=0.3, help='Seconds between submissions')
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--interval', type=int, default=8, help='ROLLING_INTERVAL_STEPS for staged KV path')
    args = parser.parse_args()

    print(f"Loading model {args.model} on {args.device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if 'cuda' in args.device else torch.float32)
    model.to(args.device)
    model.eval()

    os.environ['ROLLING_INTERVAL_STEPS'] = str(args.interval)

    print('\n== Incremental KV (staged admissions) ==')
    kv_lat = run_requests(model, tokenizer, args.device, args.n, args.stagger, args.tokens, rolling_mode=False)
    print(f"latencies: {kv_lat}")
    print(f"median={statistics.median(kv_lat):.3f}s p95={statistics.quantiles(kv_lat, n=20)[18]:.3f}s")

    print('\n== Full Recompute Rolling Mode ==')
    roll_lat = run_requests(model, tokenizer, args.device, args.n, args.stagger, args.tokens, rolling_mode=True)
    print(f"latencies: {roll_lat}")
    print(f"median={statistics.median(roll_lat):.3f}s p95={statistics.quantiles(roll_lat, n=20)[18]:.3f}s")

    out = {
        'kv_median': statistics.median(kv_lat),
        'kv_p95': statistics.quantiles(kv_lat, n=20)[18],
        'rolling_median': statistics.median(roll_lat),
        'rolling_p95': statistics.quantiles(roll_lat, n=20)[18]
    }
    print('\nJSON_SUMMARY ' + json.dumps(out))

if __name__ == '__main__':
    main()
