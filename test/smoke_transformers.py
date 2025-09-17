import os, json, time, sys, pathlib

# Ensure project root on path when running via arbitrary interpreter
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.TransformersModelManager import TransformersModelManager, DEEPSEEK_MODEL_CONFIG

# Reduce tokens for quick local test
os.environ['SMOKE_MAX_NEW_TOKENS'] = os.environ.get('SMOKE_MAX_NEW_TOKENS', '32')

def simple_messages(user_text: str):
    return [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": user_text}
    ]

def run_smoke():
    mgr = TransformersModelManager(DEEPSEEK_MODEL_CONFIG)
    outputs = []
    def cb(i, out):
        outputs.append((i, out['response'][:120]))
    mgr.run_batch_executions([
        simple_messages("Explain batching in one sentence."),
        simple_messages("List three colors."),
    ], [False, False], cb)
    return outputs

if __name__ == '__main__':
    start = time.time()
    outs = run_smoke()
    for idx, resp in outs:
        print(f"Sample {idx}: {resp}")
    print(f"Total time: {time.time()-start:.2f}s")
