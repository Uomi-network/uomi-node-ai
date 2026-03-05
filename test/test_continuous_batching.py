"""
test_continuous_batching.py — verifica continuous batching e KV cache di vLLM.

Chiama l'API pubblica /run (porta 8888) — può girare da qualsiasi macchina
client con accesso al server.

Modalità:
  --ramp     : testa a più livelli di concorrenza (es. 4→8→16→24→32) e
               mostra il throughput a ogni step → evidenzia il punto di saturazione.
  (default)  : singola fase sequenziale + singola fase parallela a --concurrency.

Opzionale: --vllm-url http://<server>:8100
  Se il server vLLM è raggiungibile direttamente, interroga /metrics per
  mostrare KV-cache utilization e verificare che il dtype sia FP8.

Usage:
  python test/test_continuous_batching.py --url http://<server>:8888 --model Qwen/Qwen3.5-35B-A3B-FP8
  python test/test_continuous_batching.py --url http://localhost:8888 --ramp --max-concurrency 32
  python test/test_continuous_batching.py --url http://localhost:8888 --vllm-url http://localhost:8100
"""

import argparse
import base64
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional


# ---------------------------------------------------------------------------
# Prompts lunghi — 32 voci per supportare stress test >24 concurrenti
# ---------------------------------------------------------------------------
LONG_PROMPTS = [
    "Racconta una storia dettagliata di un astronauta che scopre una civiltà aliena su Marte. Includi dialoghi e descrizioni.",
    "Spiega passo per passo come funziona un motore a reazione: principi fisici, componenti, differenze turbojet/turbofan/turboprop.",
    "Scrivi un saggio filosofico sul rapporto tra libertà e determinismo. Cita almeno tre filosofi.",
    "Descrivi lo sviluppo di un sistema operativo da zero: memoria, scheduling, filesystem, driver.",
    "Racconta la Seconda Guerra Mondiale: cause, battaglie principali, strategie, conseguenze politiche.",
    "Spiega il machine learning dall'algebra lineare alle reti neurali transformer con esempi pratici.",
    "Scrivi un racconto noir ambientato nella Milano degli anni '50. Protagonista: un detective privato.",
    "Analizza l'architettura dei microprocessori moderni: pipeline, cache hierarchy, branch prediction, OOO, SIMD.",
    "Spiega il protocollo RAFT per il consenso distribuito: leader election, log replication, safety.",
    "Descrivi come funziona il Byzantine fault tolerance e i suoi algoritmi principali (PBFT, Tendermint).",
    "Spiega la crittografia a chiave pubblica: RSA, ECC, e perché la lunghezza della chiave importa.",
    "Descrivi l'architettura dei transformer: attention, positional encoding, feed-forward, layer norm.",
    "Spiega il garbage collection in Java e Go: mark-and-sweep, generational GC, write barriers.",
    "Descrivi il sistema TCP/IP completo: handshake, controllo della congestione, TLS, HTTP/2.",
    "Spiega come funziona un database distribuito: sharding, replication, consistency models, MVCC.",
    "Descrivi il funzionamento di una blockchain: merkle tree, proof-of-work, proof-of-stake, fork.",
    "Spiega Kubernetes in dettaglio: pod, service, deployment, ingress, etcd, scheduler.",
    "Descrivi come funziona Docker internamente: namespaces, cgroups, union filesystem, networking.",
    "Spiega il funzionamento di un compilatore: lexer, parser, AST, IR, ottimizzazioni, code gen.",
    "Descrivi l'implementazione di un sistema di file distribuito come HDFS o GFS.",
    "Spiega il CAP theorem con esempi concreti di sistemi che scelgono CP vs AP.",
    "Descrivi come funziona WebAssembly: stack machine, linear memory, interface types, WASI.",
    "Spiega il funzionamento di un motore di ricerca: crawling, indexing, ranking, query processing.",
    "Descrivi gli algoritmi di compressione: Huffman, LZ77, LZ4, Zstd — vantaggi e svantaggi.",
    "Spiega come funziona un sistema di raccomandazione: collaborative filtering, content-based, deep learning.",
    "Descrivi l'architettura di un sistema di pagamento distribuito: idempotency, saga pattern, 2PC.",
    "Spiega come funziona un'implementazione di Redis: strutture dati, persistenza RDB/AOF, replication.",
    "Descrivi il funzionamento di un sistema operativo real-time: task scheduling, interrupt latency, priority inversion.",
    "Spiega come funziona IPv6: addressing, NDP, stateless autoconfiguration, transition mechanisms.",
    "Descrivi l'implementazione di un lock-free queue: CAS, ABA problem, memory ordering.",
    "Spiega il funzionamento di WebRTC: ICE, STUN, TURN, DTLS, SRTP.",
    "Descrivi come funziona un sistema di time-series database: LSM tree, compaction, downsampling.",
]


# ---------------------------------------------------------------------------
# Helper: conta token dalla proof
# ---------------------------------------------------------------------------

def count_tokens_from_proof(proof_zipped: str) -> int:
    if not proof_zipped:
        return 0
    try:
        raw = zlib.decompress(base64.b64decode(proof_zipped.encode("ascii"))).decode("utf-8")
        return len(json.loads(raw).get("tokens", []))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# KV cache metrics da vLLM /metrics (Prometheus)
# ---------------------------------------------------------------------------

def fetch_vllm_metrics(vllm_url: str) -> dict:
    """
    Interroga vLLM /metrics e restituisce le metriche rilevanti.
    Ritorna {} se non raggiungibile.
    """
    try:
        with urllib.request.urlopen(f"{vllm_url}/metrics", timeout=5) as resp:
            text = resp.read().decode("utf-8")
    except Exception:
        return {}

    result = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for key in ("vllm:gpu_cache_usage_perc", "vllm:cpu_cache_usage_perc",
                    "vllm:num_requests_running", "vllm:num_requests_waiting",
                    "vllm:num_requests_swapped"):
            if line.startswith(key + "{") or line.startswith(key + " "):
                try:
                    result[key] = float(line.split()[-1])
                except Exception:
                    pass
    return result


def check_kv_dtype(vllm_url: str):
    """
    Tenta di verificare il KV cache dtype interrogando vLLM.
    Non c'è un endpoint esplicito, ma /v1/models può mostrare metadati utili.
    """
    try:
        with urllib.request.urlopen(f"{vllm_url}/v1/models", timeout=5) as resp:
            data = json.loads(resp.read())
            # Cerca nei metadati del modello
            for m in data.get("data", []):
                meta = str(m)
                if "fp8" in meta.lower() or "kv" in meta.lower():
                    print(f"  [kv-check] /v1/models metadata: {meta[:200]}")
    except Exception:
        pass

    # Tenta di leggere i server args via /v1/models o /info se disponibile
    for path in ["/info", "/v1/server_info"]:
        try:
            with urllib.request.urlopen(f"{vllm_url}{path}", timeout=3) as resp:
                info = json.loads(resp.read())
                print(f"  [kv-check] {path}: {json.dumps(info)[:300]}")
                return
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Chiamata HTTP a /run
# ---------------------------------------------------------------------------

def call_run(base_url: str, model: str, prompt: str,
             max_tokens: int = 600, timeout: int = 600) -> dict:
    input_payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_new_tokens": max_tokens,
        "enable_thinking": False,
    }
    body = json.dumps({
        "model": model,
        "input": json.dumps(input_payload),
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/run", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            elapsed = time.perf_counter() - start
            tokens = count_tokens_from_proof(data.get("proof", ""))
            content = data.get("response", "")
            return {
                "elapsed": elapsed,
                "completion_tokens": tokens,
                "tokens_per_sec": tokens / elapsed if elapsed > 0 else 0,
                "content_preview": (content[:100] + "...") if len(content) > 100 else content,
                "error": None,
            }
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - start
        body_str = ""
        try:
            body_str = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return {"elapsed": elapsed, "completion_tokens": 0, "tokens_per_sec": 0,
                "content_preview": "", "error": f"HTTP {e.code}: {body_str[:200]}"}
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"elapsed": elapsed, "completion_tokens": 0, "tokens_per_sec": 0,
                "content_preview": "", "error": str(e)}


# ---------------------------------------------------------------------------
# Esecuzione sequenziale e concorrente
# ---------------------------------------------------------------------------

def run_sequential(base_url: str, model: str, prompts: list,
                   max_tokens: int, timeout: int) -> list:
    results = []
    for i, prompt in enumerate(prompts):
        print(f"  [seq] {i+1}/{len(prompts)}...", flush=True)
        r = call_run(base_url, model, prompt, max_tokens, timeout)
        results.append(r)
        status = (f"OK ({r['completion_tokens']} tok, {r['elapsed']:.1f}s)"
                  if not r["error"] else f"ERR: {r['error'][:80]}")
        print(f"  [seq] {i+1} done — {status}", flush=True)
    return results


def run_concurrent(base_url: str, model: str, prompts: list,
                   max_tokens: int, concurrency: int, timeout: int,
                   stagger_sec: float = 0.0,
                   vllm_url: str = "",
                   metrics_out: Optional[dict] = None) -> list:
    """
    Esegue `len(prompts)` richieste con max `concurrency` thread paralleli.
    stagger_sec > 0: ogni worker aspetta i*stagger_sec prima di partire
    (simula arrivi sfalsati per forzare il batching su richieste già in volo).
    """
    results_list = [None] * len(prompts)
    lock = threading.Lock()
    peak_metrics = {}

    def task(idx, prompt):
        if stagger_sec > 0:
            time.sleep(idx * stagger_sec)
        print(f"  [par] avvio {idx+1}/{len(prompts)}...", flush=True)
        r = call_run(base_url, model, prompt, max_tokens, timeout)
        with lock:
            results_list[idx] = r
            # Snapshot KV metrics subito dopo la risposta
            if vllm_url:
                m = fetch_vllm_metrics(vllm_url)
                if m.get("vllm:gpu_cache_usage_perc", 0) > peak_metrics.get("vllm:gpu_cache_usage_perc", 0):
                    peak_metrics.update(m)
            status = (f"OK ({r['completion_tokens']} tok, {r['elapsed']:.1f}s)"
                      if not r["error"] else f"ERR: {r['error'][:80]}")
            print(f"  [par] {idx+1} done — {status}", flush=True)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(task, i, p): i for i, p in enumerate(prompts)}
        for f in as_completed(futs):
            f.result()

    if metrics_out is not None:
        metrics_out.update(peak_metrics)
    return results_list


def print_results(label: str, results: list, wall_time: float) -> tuple:
    errors = [r for r in results if r["error"]]
    ok = [r for r in results if not r["error"]]
    total_tokens = sum(r["completion_tokens"] for r in ok)
    latencies = [r["elapsed"] for r in ok]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Richieste OK / totali : {len(ok)} / {len(results)}")
    if errors:
        print(f"  Errori                : {len(errors)}")
        for e in errors[:3]:
            print(f"    - {e['error']}")
    if ok:
        print(f"  Wall-clock time       : {wall_time:.2f}s")
        print(f"  Token totali generati : {total_tokens}")
        if wall_time > 0:
            print(f"  Throughput globale    : {total_tokens / wall_time:.1f} tok/s")
        print(f"  Latenza media/req     : {statistics.mean(latencies):.1f}s")
        print(f"  Latenza max/req       : {max(latencies):.1f}s")
        print(f"  Latenza min/req       : {min(latencies):.1f}s")
    return wall_time, total_tokens


# ---------------------------------------------------------------------------
# Ramp test: misura throughput a diversi livelli di concorrenza
# ---------------------------------------------------------------------------

def run_ramp_test(base_url: str, model: str, prompts: list, max_tokens: int,
                  levels: list, timeout: int, vllm_url: str = ""):
    """
    Per ogni livello di concorrenza in `levels`, esegue `level` richieste in parallelo
    e misura throughput + latenza. Mostra la curva di scalabilità.
    """
    print(f"\n{'='*60}")
    print(f"  RAMP TEST — livelli: {levels}")
    print(f"{'='*60}")

    ramp_results = []
    for concurrency in levels:
        batch = (prompts * ((concurrency // len(prompts)) + 1))[:concurrency]
        print(f"\n  → concorrenza={concurrency} ({concurrency} richieste)...")
        t0 = time.perf_counter()
        peak_m: dict = {}
        res = run_concurrent(base_url, model, batch, max_tokens,
                             concurrency, timeout,
                             stagger_sec=0.5, vllm_url=vllm_url,
                             metrics_out=peak_m)
        wall = time.perf_counter() - t0

        ok = [r for r in res if r and not r["error"]]
        errors = len(res) - len(ok)
        tokens = sum(r["completion_tokens"] for r in ok)
        tps = tokens / wall if wall > 0 else 0
        lat_mean = statistics.mean(r["elapsed"] for r in ok) if ok else 0

        kv_str = ""
        if peak_m:
            kv_pct = peak_m.get("vllm:gpu_cache_usage_perc", 0) * 100
            running = peak_m.get("vllm:num_requests_running", "?")
            waiting = peak_m.get("vllm:num_requests_waiting", "?")
            kv_str = f"  KV={kv_pct:.0f}% run={running} wait={waiting}"

        print(f"  concorrenza={concurrency:2d}  wall={wall:.1f}s  "
              f"tps={tps:.0f}  lat_mean={lat_mean:.1f}s  "
              f"errors={errors}{kv_str}")
        ramp_results.append((concurrency, wall, tps, lat_mean, errors))

    # Sommario tabellare
    print(f"\n  {'Concorrenza':>12} {'Wall(s)':>9} {'TPS':>7} {'Lat.media(s)':>13} {'Errors':>7}")
    print(f"  {'-'*12} {'-'*9} {'-'*7} {'-'*13} {'-'*7}")
    for c, w, t, l, e in ramp_results:
        print(f"  {c:>12} {w:>9.1f} {t:>7.0f} {l:>13.1f} {e:>7}")

    # Analisi speedup rispetto al primo livello
    if len(ramp_results) >= 2:
        base_tps = ramp_results[0][2]
        if base_tps > 0:
            print(f"\n  Scaling (TPS relativo al livello minimo):")
            for c, _, t, _, _ in ramp_results:
                bar = "█" * int(t / base_tps * 10)
                print(f"  {c:>3}x  {bar:<30} {t/base_tps:.2f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test continuous batching via /run di uomi-ai.py")
    parser.add_argument("--url", type=str, default="http://localhost:8888")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-35B-A3B-FP8")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Concorrenza per la fase parallela standard (default 8)")
    parser.add_argument("--max-concurrency", type=int, default=32,
                        help="Concorrenza massima per --ramp (default 32)")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--prompts", type=int, default=8,
                        help="Prompt da usare nella fase standard (max 32)")
    parser.add_argument("--skip-sequential", action="store_true")
    parser.add_argument("--ramp", action="store_true",
                        help="Esegui ramp test a più livelli di concorrenza")
    parser.add_argument("--ramp-levels", type=str, default="4,8,16,24,32",
                        help="Livelli di concorrenza per --ramp (default: 4,8,16,24,32)")
    parser.add_argument("--vllm-url", type=str, default="",
                        help="URL diretto a vLLM (es. http://server:8100) per KV metrics")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    vllm_url = args.vllm_url.rstrip("/") if args.vllm_url else ""
    n_prompts = min(args.prompts, len(LONG_PROMPTS))
    prompts = LONG_PROMPTS[:n_prompts]
    concurrency = min(args.concurrency, n_prompts)

    print(f"\nuomi-ai.py Continuous Batching & KV Stress Test")
    print(f"  server        : {base_url}")
    print(f"  model         : {args.model}")
    print(f"  prompts       : {n_prompts}")
    print(f"  concurrency   : {concurrency}")
    print(f"  max_tokens    : {args.max_tokens}")
    if vllm_url:
        print(f"  vllm_url      : {vllm_url} (KV metrics)")

    # Health check
    try:
        with urllib.request.urlopen(f"{base_url}/status", timeout=5) as resp:
            d = json.loads(resp.read())
            print(f"  status        : OK — version={d.get('UOMI_ENGINE_PALLET_VERSION', '?')}")
    except Exception as e:
        print(f"\nERRORE: server non raggiungibile ({e})")
        return

    # KV cache info (se vllm_url disponibile)
    if vllm_url:
        print(f"\n--- KV cache check (vLLM diretto) ---")
        check_kv_dtype(vllm_url)
        m = fetch_vllm_metrics(vllm_url)
        if m:
            print(f"  KV gpu usage  : {m.get('vllm:gpu_cache_usage_perc', 0)*100:.1f}%")
            print(f"  Req running   : {m.get('vllm:num_requests_running', '?')}")
            print(f"  Req waiting   : {m.get('vllm:num_requests_waiting', '?')}")
        else:
            print("  /metrics non raggiungibile — verifica che vllm_url punti alla porta 8100")
        # Nota esplicativa
        print("\n  NOTA: per verificare FP8 KV cache, controlla i log di avvio vLLM:")
        print("  cerca: 'Using FP8 KV cache' o '--kv-cache-dtype fp8_e5m2' nel comando")

    # ----------------------------------------------------------------
    # RAMP TEST
    # ----------------------------------------------------------------
    if args.ramp:
        levels = [int(x) for x in args.ramp_levels.split(",")]
        levels = [l for l in levels if l <= args.max_concurrency]
        ramp_prompts = LONG_PROMPTS  # usa tutti e 32
        run_ramp_test(base_url, args.model, ramp_prompts, args.max_tokens,
                      levels, args.timeout, vllm_url)
        return

    # ----------------------------------------------------------------
    # TEST STANDARD: sequenziale + parallelo
    # ----------------------------------------------------------------
    seq_wall = seq_tokens = None

    if not args.skip_sequential:
        print(f"\n--- FASE 1: baseline sequenziale ({n_prompts} richieste) ---")
        t0 = time.perf_counter()
        seq_res = run_sequential(base_url, args.model, prompts, args.max_tokens, args.timeout)
        seq_wall = time.perf_counter() - t0
        seq_wall, seq_tokens = print_results("SEQUENZIALE", seq_res, seq_wall)
    else:
        print("\n(baseline sequenziale saltato)")

    print(f"\n--- FASE 2: {concurrency} richieste in parallelo ---")
    t0 = time.perf_counter()
    peak_m: dict = {}
    par_res = run_concurrent(base_url, args.model, prompts, args.max_tokens,
                             concurrency, args.timeout,
                             stagger_sec=0.5, vllm_url=vllm_url,
                             metrics_out=peak_m)
    par_wall = time.perf_counter() - t0
    par_wall, par_tokens = print_results(f"PARALLELO ({concurrency}x)", par_res, par_wall)

    if vllm_url and peak_m:
        print(f"\n  Peak KV usage durante parallelo: "
              f"{peak_m.get('vllm:gpu_cache_usage_perc', 0)*100:.1f}%  "
              f"run={peak_m.get('vllm:num_requests_running', '?')}  "
              f"wait={peak_m.get('vllm:num_requests_waiting', '?')}")

    # ----------------------------------------------------------------
    # SOMMARIO
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  SOMMARIO")
    print(f"{'='*60}")
    if seq_wall and seq_tokens:
        speedup = seq_wall / par_wall if par_wall > 0 else 0
        seq_tps = seq_tokens / seq_wall
        par_tps = par_tokens / par_wall if par_wall > 0 else 0
        print(f"  Speedup wall-clock    : {speedup:.2f}x")
        print(f"  Throughput seq        : {seq_tps:.1f} tok/s")
        print(f"  Throughput par        : {par_tps:.1f} tok/s")
        if seq_tps > 0:
            print(f"  Throughput improvement: {par_tps/seq_tps:.2f}x")
        print()
        if speedup > 1.5:
            print(f"  ✓ Continuous batching ATTIVO: {speedup:.1f}x speedup")
        elif speedup > 1.1:
            print("  ~ Batching parziale")
        else:
            print("  ? Speedup marginale — GPU già satura o prompt troppo brevi")
    else:
        par_tps = par_tokens / par_wall if par_wall > 0 else 0
        print(f"  Throughput parallelo  : {par_tps:.1f} tok/s")
        print(f"  Wall-clock            : {par_wall:.2f}s per {n_prompts} richieste")
    print()


if __name__ == "__main__":
    main()
