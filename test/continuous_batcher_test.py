import time
import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.TransformersModelManager import TransformersModelManager, DEEPSEEK_MODEL_CONFIG

# NOTE: Uses the small 1B model configured in DEEPSEEK_MODEL_CONFIG; keep tokens low for speed.

def test_continuous_simple_join():
    manager = TransformersModelManager(DEEPSEEK_MODEL_CONFIG)
    manager.enable_continuous(max_active=4)
    results = {}
    tokens_streamed = {}

    def on_token(sid, txt, meta):
        tokens_streamed.setdefault(sid, []).append(meta['id'])

    def on_complete(sid, response, proof):
        results[sid] = {'response': response, 'proof': proof}

    messages1 = [{"role": "user", "content": "List three fruits."}]
    sid1 = manager.submit_continuous(messages1, enable_thinking=False, sampling_cfg={'temperature':0.6,'top_k':5}, max_new_tokens=8, on_token=on_token, on_complete=on_complete)

    time.sleep(0.05)  # Stagger second request

    messages2 = [{"role": "user", "content": "List three animals."}]
    sid2 = manager.submit_continuous(messages2, enable_thinking=False, sampling_cfg={'temperature':0.6,'top_k':5}, max_new_tokens=8, on_token=on_token, on_complete=on_complete)

    # Wait for completion (timeout safety)
    deadline = time.time() + 60
    while len(results) < 2 and time.time() < deadline:
        time.sleep(0.1)

    assert sid1 in results and sid2 in results, "Both sequences should complete"
    assert len(results[sid1]['proof']['tokens']) > 0
    assert len(results[sid2]['proof']['tokens']) > 0
    # Ensure streaming delivered some tokens before complete
    assert len(tokens_streamed[sid1]) >= 1
    assert len(tokens_streamed[sid2]) >= 1


def test_continuous_deterministic_mode():
    manager = TransformersModelManager(DEEPSEEK_MODEL_CONFIG)
    manager.enable_continuous(max_active=2)
    results_runs = []

    def run_once():
        results = {}
        def on_token(sid, txt, meta):
            pass
        def on_complete(sid, response, proof):
            results[sid] = response
        sid = manager.submit_continuous([{"role":"user","content":"Say ONE word color."}], enable_thinking=False, sampling_cfg={'deterministic':True, 'temperature':1.0}, max_new_tokens=2, on_token=on_token, on_complete=on_complete)
        deadline = time.time() + 60
        while sid not in results and time.time() < deadline:
            time.sleep(0.05)
        return results[sid]

    first = run_once()
    second = run_once()
    # Deterministic should produce identical outputs (best effort; if prompt too open this may fail)
    assert first == second


def test_continuous_check_mode():
    manager = TransformersModelManager(DEEPSEEK_MODEL_CONFIG)
    manager.enable_continuous(max_active=2)

    # First run to capture a short sequence of tokens deterministically
    collected = {}
    def on_token_capture(sid, txt, meta):
        collected.setdefault('tokens', []).append(meta['id'])
    def on_complete_capture(sid, response, proof):
        collected['done'] = True
    sid = manager.submit_continuous([{"role":"user","content":"Say hi."}], enable_thinking=False, sampling_cfg={'deterministic':True}, max_new_tokens=3, on_token=on_token_capture, on_complete=on_complete_capture)
    deadline = time.time() + 60
    while 'done' not in collected and time.time() < deadline:
        time.sleep(0.05)
    assert 'tokens' in collected and len(collected['tokens']) > 0

    forced = collected['tokens']
    results = {}
    def on_token_check(sid, txt, meta):
        pass
    def on_complete_check(sid, response, proof):
        results['verified'] = proof.get('verified', False)
    manager.submit_continuous([{"role":"user","content":"Say hi."}], enable_thinking=False, sampling_cfg={'deterministic':True}, max_new_tokens=len(forced), on_token=on_token_check, on_complete=on_complete_check, is_check=True, forced_tokens=forced)
    deadline = time.time() + 60
    while 'verified' not in results and time.time() < deadline:
        time.sleep(0.05)
    assert results.get('verified') is True
