"""
test_proof_verification_client.py — verifica il contratto blockchain chiamando
l'API pubblica /run di uomi-ai.py (porta 8888).

Simula esattamente il ruolo del VALIDATORE nella blockchain:
  1. Riceve una risposta + proof da un nodo generate
  2. Ri-sottomette con la stessa proof → deve ottenere verified=True
  3. Tenta con proof manomesse → deve ottenere verified=False / HTTP 400

NON richiede accesso diretto a vLLM (porta 8100) né il tokenizer installato.
Può girare da qualsiasi macchina client con accesso alla porta 8888.

Usage:
  python test/test_proof_verification_client.py --url http://<server-ip>:8888 --model Qwen/Qwen3.5-35B-A3B-FP8
"""

import argparse
import base64
import json
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers — stessa logica di zipper.py
# ---------------------------------------------------------------------------

def zip_proof(proof_dict: dict) -> str:
    raw = json.dumps(proof_dict)
    compressed = zlib.compress(raw.encode("utf-8"))
    return base64.b64encode(compressed).decode("ascii")


def unzip_proof(s: str) -> dict:
    compressed = base64.b64decode(s.encode("ascii"))
    raw = zlib.decompress(compressed).decode("utf-8")
    return json.loads(raw)


def call_run(base_url: str, model: str, messages: list,
             proof: Optional[str] = None,
             enable_thinking: bool = False,
             timeout: int = 300) -> tuple[int, dict]:
    """
    Chiama POST /run e ritorna (http_status_code, response_dict).
    Non lancia eccezione su 4xx — li gestisce il chiamante.
    """
    input_payload = {
        "messages": messages,
        "enable_thinking": enable_thinking,
        "sampling": {"temperature": 0.6, "top_k": 5},
    }
    body = {"model": model, "input": json.dumps(input_payload)}
    if proof is not None:
        body["proof"] = proof

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/run",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body_bytes = b""
        try:
            body_bytes = e.read()
        except Exception:
            pass
        try:
            return e.code, json.loads(body_bytes)
        except Exception:
            return e.code, {"_raw": body_bytes.decode("utf-8", errors="replace")}


# ---------------------------------------------------------------------------
# Stato test
# ---------------------------------------------------------------------------

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results = []
results_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Grid di test: (label, enable_thinking, messages)
# Dimensioni coperte: thinking on/off, lunghezza corta/media/lunga,
#   tipo (fatto, ragionamento, codice, multi-turn, inglese)
# ---------------------------------------------------------------------------

PROMPT_GRID = [
    # ── Corto ──────────────────────────────────────────────────────────────
    ("corto/no-think",
     False,
     [{"role": "user", "content": "Cos'è il Byzantine fault tolerance? Rispondi in 1 frase."}]),

    ("corto/think",
     True,
     [{"role": "user", "content": "Cos'è il Byzantine fault tolerance? Rispondi in 1 frase."}]),

    # ── Medio ───────────────────────────────────────────────────────────────
    ("medio/no-think",
     False,
     [{"role": "user", "content": "Spiega cos'è la crittografia asimmetrica e come funziona RSA. Rispondi in 3-4 frasi."}]),

    ("medio/think",
     True,
     [{"role": "user", "content": "Spiega cos'è la crittografia asimmetrica e come funziona RSA. Rispondi in 3-4 frasi."}]),

    # ── Lungo ───────────────────────────────────────────────────────────────
    ("lungo/no-think",
     False,
     [{"role": "user", "content": "Spiega dettagliatamente il protocollo RAFT: leader election, log replication, safety e liveness."}]),

    ("lungo/think",
     True,
     [{"role": "user", "content": "Spiega dettagliatamente il protocollo RAFT: leader election, log replication, safety e liveness."}]),

    # ── Ragionamento matematico ─────────────────────────────────────────────
    ("math/no-think",
     False,
     [{"role": "user", "content": "Un treno parte alle 9:15 e arriva alle 11:42. Quanti minuti dura il viaggio?"}]),

    ("math/think",
     True,
     [{"role": "user", "content": "Un treno parte alle 9:15 e arriva alle 11:42. Quanti minuti dura il viaggio?"}]),

    # ── Codice ─────────────────────────────────────────────────────────────
    ("code/no-think",
     False,
     [{"role": "user", "content": "Scrivi una funzione Python che verifica se un numero è primo. Solo codice, niente spiegazioni."}]),

    ("code/think",
     True,
     [{"role": "user", "content": "Scrivi una funzione Python che verifica se un numero è primo. Solo codice, niente spiegazioni."}]),

    # ── Multi-turn ─────────────────────────────────────────────────────────
    ("multiturn/no-think",
     False,
     [
         {"role": "user",      "content": "Cos'è una blockchain?"},
         {"role": "assistant", "content": "Una blockchain è un registro distribuito e immutabile."},
         {"role": "user",      "content": "E cos'è un nodo validatore?"},
     ]),

    # ── Inglese ────────────────────────────────────────────────────────────
    ("en/no-think",
     False,
     [{"role": "user", "content": "What is a Merkle tree and why is it used in blockchains? Answer in 2-3 sentences."}]),

    ("en/think",
     True,
     [{"role": "user", "content": "What is a Merkle tree and why is it used in blockchains? Answer in 2-3 sentences."}]),
]


def record(name: str, status: str, detail: str = ""):
    symbol = "✓" if status == PASS else ("~" if status == SKIP else "✗")
    with results_lock:
        print(f"  [{symbol}] {name}", end="")
        if detail:
            print(f" — {detail}")
        else:
            print()
        results.append((name, status, detail))


# ---------------------------------------------------------------------------
# Suite di test
# ---------------------------------------------------------------------------

def test_generate_returns_proof(status: int, resp: dict):
    """T1: la chiamata generate ritorna HTTP 200 con proof non vuota."""
    try:
        assert status == 200, f"HTTP {status}"
        assert resp.get("result") is True, f"result=False: {resp}"
        assert "proof" in resp, "campo 'proof' assente"
        assert isinstance(resp["proof"], str) and len(resp["proof"]) > 0, "proof vuota"
        assert "response" in resp, "campo 'response' assente"
        record("T1 generate → HTTP 200 + proof", PASS,
               f"response='{resp['response'][:60]}...'")
    except AssertionError as e:
        record("T1 generate → HTTP 200 + proof", FAIL, str(e))


def test_proof_is_valid_zlib_json(proof_zipped: str):
    """T2: la proof è zlib+base64 di un JSON {"tokens": [{"id": int}]}."""
    try:
        proof = unzip_proof(proof_zipped)
        assert isinstance(proof, dict), "non è un dict"
        assert "tokens" in proof, "manca 'tokens'"
        assert isinstance(proof["tokens"], list), "'tokens' non è lista"
        assert len(proof["tokens"]) > 0, "tokens lista vuota"
        for tok in proof["tokens"]:
            assert isinstance(tok, dict) and "id" in tok and isinstance(tok["id"], int), \
                f"token malformato: {tok}"
        record("T2 struttura proof (zlib+JSON)", PASS,
               f"{len(proof['tokens'])} tokens")
    except Exception as e:
        record("T2 struttura proof (zlib+JSON)", FAIL, str(e))


def test_check_valid_proof(base_url: str, model: str, messages: list,
                           proof_zipped: str, response_text: str):
    """T3: check con proof originale → HTTP 200, verified implicito."""
    try:
        status, resp = call_run(base_url, model, messages, proof=proof_zipped)
        if status == 200 and resp.get("result") is True:
            record("T3 check proof valida → verified=True", PASS,
                   f"response='{resp.get('response','')[:60]}'")
        else:
            err = resp.get("error", resp.get("_raw", ""))
            record("T3 check proof valida → verified=True", FAIL,
                   f"HTTP {status}, result={resp.get('result')}, error={err}")
    except Exception as e:
        record("T3 check proof valida → verified=True", FAIL, str(e))


def test_check_wrong_ids_proof(base_url: str, model: str, messages: list,
                                proof_zipped: str):
    """T4: check con tutti gli IDs sostituiti da 1 → HTTP 400 + error."""
    try:
        proof = unzip_proof(proof_zipped)
        tampered = {"tokens": [{"id": 1} for _ in proof["tokens"]]}
        tampered_zipped = zip_proof(tampered)

        status, resp = call_run(base_url, model, messages, proof=tampered_zipped)
        if status == 400 and resp.get("result") is False:
            record("T4 check IDs sbagliati → HTTP 400 + result=False", PASS,
                   f"error='{resp.get('error', '')}'")
        elif status == 400:
            record("T4 check IDs sbagliati → HTTP 400 + result=False", PASS,
                   f"HTTP 400 (result={resp.get('result')})")
        else:
            record("T4 check IDs sbagliati → HTTP 400 + result=False", FAIL,
                   f"HTTP {status}, result={resp.get('result')} (proof manomessa ACCETTATA)")
    except Exception as e:
        record("T4 check IDs sbagliati → HTTP 400 + result=False", FAIL, str(e))


def test_check_extra_tokens_proof(base_url: str, model: str, messages: list,
                                   proof_zipped: str):
    """T5: proof con token extra aggiunti in coda → HTTP 400."""
    try:
        proof = unzip_proof(proof_zipped)
        original_ids = [t["id"] for t in proof["tokens"]]
        tampered = {"tokens": [{"id": i} for i in original_ids] + [{"id": 999}, {"id": 999}]}
        tampered_zipped = zip_proof(tampered)

        status, resp = call_run(base_url, model, messages, proof=tampered_zipped)
        if status == 400:
            record("T5 check proof con token extra → HTTP 400", PASS,
                   f"+2 token fasulli rigettati")
        else:
            record("T5 check proof con token extra → HTTP 400", FAIL,
                   f"HTTP {status} (proof inflazionata ACCETTATA)")
    except Exception as e:
        record("T5 check proof con token extra → HTTP 400", FAIL, str(e))


def test_check_truncated_proof(base_url: str, model: str, messages: list,
                                proof_zipped: str):
    """T6: proof troncata a metà → HTTP 400."""
    try:
        proof = unzip_proof(proof_zipped)
        original_ids = [t["id"] for t in proof["tokens"]]
        if len(original_ids) < 4:
            record("T6 check proof troncata → HTTP 400", SKIP,
                   "proof troppo corta per troncare")
            return
        half = len(original_ids) // 2
        tampered = {"tokens": [{"id": i} for i in original_ids[:half]]}
        tampered_zipped = zip_proof(tampered)

        status, resp = call_run(base_url, model, messages, proof=tampered_zipped)
        if status == 400:
            record("T6 check proof troncata → HTTP 400", PASS,
                   f"troncata a {half}/{len(original_ids)} token rigettata")
        else:
            record("T6 check proof troncata → HTTP 400", FAIL,
                   f"HTTP {status} (proof troncata ACCETTATA)")
    except Exception as e:
        record("T6 check proof troncata → HTTP 400", FAIL, str(e))


def test_check_invalid_base64_proof(base_url: str, model: str, messages: list):
    """T7: proof con base64 non valido → HTTP 400 (non crash del server)."""
    try:
        status, resp = call_run(base_url, model, messages, proof="NOT_VALID_BASE64!!!")
        if status == 400:
            record("T7 check proof base64 non valida → HTTP 400", PASS)
        else:
            record("T7 check proof base64 non valida → HTTP 400", FAIL,
                   f"HTTP {status} (proof garbage ACCETTATA)")
    except Exception as e:
        record("T7 check proof base64 non valida → HTTP 400", FAIL, str(e))


def test_check_empty_proof(base_url: str, model: str, messages: list):
    """T8: proof stringa vuota → HTTP 400."""
    try:
        status, resp = call_run(base_url, model, messages, proof="")
        if status == 400:
            record("T8 check proof vuota → HTTP 400", PASS)
        else:
            record("T8 check proof vuota → HTTP 400", FAIL,
                   f"HTTP {status}")
    except Exception as e:
        record("T8 check proof vuota → HTTP 400", FAIL, str(e))


def test_proof_not_reusable_on_different_prompt(base_url: str, model: str,
                                                 proof_zipped: str):
    """T9: proof di prompt A usata su prompt B diverso → HTTP 400."""
    try:
        different_messages = [
            {"role": "user", "content": "Rispondi solo con: QUARANTADUE"}
        ]
        status, resp = call_run(base_url, model, different_messages, proof=proof_zipped)
        if status == 400:
            record("T9 proof non trasferibile tra prompt → HTTP 400", PASS,
                   "proof di prompt A rigettata su prompt B")
        else:
            record("T9 proof non trasferibile tra prompt → HTTP 400", FAIL,
                   f"HTTP {status} (proof riusata su altro prompt — PROBLEMA SICUREZZA)")
    except Exception as e:
        record("T9 proof non trasferibile tra prompt → HTTP 400", FAIL, str(e))


def test_generate_check_roundtrip_twice(base_url: str, model: str, messages: list):
    """T10: generate due volte con stessa domanda → entrambe le proof passano il check."""
    try:
        results_inner = []
        for i in range(2):
            s, r = call_run(base_url, model, messages)
            if s != 200 or not r.get("result"):
                record(f"T10 double generate+check roundtrip", FAIL,
                       f"generate {i+1} fallita: HTTP {s}")
                return
            proof = r["proof"]
            cs, cr = call_run(base_url, model, messages, proof=proof)
            results_inner.append((s, cs, cr.get("result"), proof))

        ok = all(cs == 200 and cr_result is True for _, cs, cr_result, _ in results_inner)
        if ok:
            record("T10 double generate+check roundtrip", PASS,
                   "entrambe le proof verificate correttamente")
        else:
            record("T10 double generate+check roundtrip", FAIL,
                   str([(cs, cr) for _, cs, cr, _ in results_inner]))
    except Exception as e:
        record("T10 double generate+check roundtrip", FAIL, str(e))


# ---------------------------------------------------------------------------
# Suite multi-prompt — sfrutta il continuous batching
# ---------------------------------------------------------------------------

def _parallel(fn_args: list, n_workers: int) -> list:
    """Esegue [(fn, args), ...] in parallelo e restituisce i risultati in ordine."""
    results_out = [None] * len(fn_args)
    def task(i, fn, args):
        results_out[i] = fn(*args)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(task, i, fn, args): i for i, (fn, args) in enumerate(fn_args)}
        for f in as_completed(futs):
            f.result()
    return results_out


def run_suite_for_prompt(base_url: str, model: str, messages: list,
                         enable_thinking: bool, tag: str, timeout: int):
    """Esegue la suite completa (generate + tutti i check) per un singolo prompt."""

    def r(name, status, detail=""):
        record(f"{name} [{tag}]", status, detail)

    # --- T1: generate ---
    gen_status, gen_resp = call_run(base_url, model, messages,
                                    enable_thinking=enable_thinking, timeout=timeout)
    if gen_status != 200 or not gen_resp.get("result") or not gen_resp.get("proof"):
        r("T1 generate → proof", FAIL, f"HTTP {gen_status}")
        return
    proof_zipped = gen_resp["proof"]
    r("T1 generate → proof", PASS, f"response='{gen_resp.get('response','')[:50]}...'")

    # --- T2: struttura proof ---
    try:
        proof_obj = unzip_proof(proof_zipped)
        assert isinstance(proof_obj, dict) and "tokens" in proof_obj
        assert len(proof_obj["tokens"]) > 0
        r("T2 struttura proof", PASS, f"{len(proof_obj['tokens'])} tokens")
    except Exception as e:
        r("T2 struttura proof", FAIL, str(e))
        return

    ids = [t["id"] for t in proof_obj["tokens"]]

    # --- T3: check proof valida ---
    s, resp = call_run(base_url, model, messages, proof=proof_zipped,
                       enable_thinking=enable_thinking, timeout=timeout)
    if s == 200 and resp.get("result"):
        r("T3 check valida → 200", PASS)
    else:
        r("T3 check valida → 200", FAIL,
          f"HTTP {s} result={resp.get('result')} error={resp.get('error','')}")

    # --- T4: IDs tutti sbagliati ---
    tampered = zip_proof({"tokens": [{"id": 1} for _ in ids]})
    s, _ = call_run(base_url, model, messages, proof=tampered,
                    enable_thinking=enable_thinking, timeout=timeout)
    if s == 400:
        r("T4 IDs sbagliati → 400", PASS)
    else:
        r("T4 IDs sbagliati → 400", FAIL, f"HTTP {s} (manomessa ACCETTATA)")

    # --- T5: token extra in coda ---
    tampered = zip_proof({"tokens": [{"id": i} for i in ids] + [{"id": 999}, {"id": 999}]})
    s, _ = call_run(base_url, model, messages, proof=tampered,
                    enable_thinking=enable_thinking, timeout=timeout)
    if s == 400:
        r("T5 token extra → 400", PASS)
    else:
        r("T5 token extra → 400", FAIL, f"HTTP {s} (inflazionata ACCETTATA)")

    # --- T6: proof troncata ---
    if len(ids) >= 4:
        half = len(ids) // 2
        tampered = zip_proof({"tokens": [{"id": i} for i in ids[:half]]})
        s, _ = call_run(base_url, model, messages, proof=tampered,
                        enable_thinking=enable_thinking, timeout=timeout)
        if s == 400:
            r("T6 troncata → 400", PASS, f"{half}/{len(ids)} token")
        else:
            r("T6 troncata → 400", FAIL, f"HTTP {s} (troncata ACCETTATA)")
    else:
        r("T6 troncata → 400", SKIP, "proof troppo corta")

    # --- T9: proof non trasferibile tra prompt ---
    diff_messages = [{"role": "user", "content": "Rispondi solo con: QUARANTADUE"}]
    s, _ = call_run(base_url, model, diff_messages, proof=proof_zipped, timeout=timeout)
    if s == 400:
        r("T9 prompt diverso → 400", PASS)
    else:
        r("T9 prompt diverso → 400", FAIL, f"HTTP {s} (PROBLEMA SICUREZZA)")

    # --- T10: double generate+check roundtrip ---
    s2, r2 = call_run(base_url, model, messages,
                      enable_thinking=enable_thinking, timeout=timeout)
    if s2 == 200 and r2.get("result") and r2.get("proof"):
        s3, r3 = call_run(base_url, model, messages, proof=r2["proof"],
                          enable_thinking=enable_thinking, timeout=timeout)
        if s3 == 200 and r3.get("result"):
            r("T10 second roundtrip", PASS)
        else:
            r("T10 second roundtrip", FAIL, f"check HTTP {s3} result={r3.get('result')}")
    else:
        r("T10 second roundtrip", FAIL, f"second generate HTTP {s2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test proof blockchain — client mode (porta 8888)")
    parser.add_argument("--url", type=str, default="http://localhost:8888",
                        help="URL base del server uomi-ai.py (es. http://1.2.3.4:8888)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-35B-A3B-FP8")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout per singola chiamata HTTP in secondi")
    parser.add_argument("--workers", type=int, default=8,
                        help="Max richieste parallele nella fase grid (default 8)")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    print(f"\nvLLM Proof Verification Test — CLIENT MODE (via /run)")
    print(f"  server  : {base_url}")
    print(f"  model   : {args.model}")
    print(f"  timeout : {args.timeout}s per request")
    print(f"  workers : {args.workers} (fase concorrente)")

    # Health check via /status
    try:
        with urllib.request.urlopen(f"{base_url}/status", timeout=5) as resp:
            status_data = json.loads(resp.read())
            print(f"  status  : OK — version={status_data.get('UOMI_ENGINE_PALLET_VERSION', '?')}")
    except Exception as e:
        print(f"\nERRORE: server non raggiungibile ({e})")
        print(f"Verifica che uomi-ai.py sia in esecuzione su {base_url}")
        sys.exit(1)

    # Prompt principale per i test
    messages = [
        {"role": "user",
         "content": "Cos'è il Byzantine fault tolerance? Rispondi in 2-3 frasi."}
    ]

    # --- GENERATE iniziale ---
    print(f"\n--- Generazione proof iniziale ---")
    t0 = time.perf_counter()
    gen_status, gen_resp = call_run(base_url, args.model, messages, timeout=args.timeout)
    elapsed = time.perf_counter() - t0
    print(f"  HTTP status : {gen_status}")
    if gen_status == 200:
        print(f"  response    : '{gen_resp.get('response', '')[:100]}'")
        print(f"  proof size  : {len(gen_resp.get('proof', ''))} chars")
    else:
        print(f"  response    : {gen_resp}")
    print(f"  elapsed     : {elapsed:.1f}s")

    proof_zipped = gen_resp.get("proof", "") if gen_status == 200 else ""
    response_text = gen_resp.get("response", "") if gen_status == 200 else ""

    # --- SUITE DI TEST ---
    print(f"\n--- Suite di test ---")
    test_generate_returns_proof(gen_status, gen_resp)

    if proof_zipped:
        test_proof_is_valid_zlib_json(proof_zipped)
        test_check_valid_proof(base_url, args.model, messages, proof_zipped, response_text)
        test_check_wrong_ids_proof(base_url, args.model, messages, proof_zipped)
        test_check_extra_tokens_proof(base_url, args.model, messages, proof_zipped)
        test_check_truncated_proof(base_url, args.model, messages, proof_zipped)
        test_check_invalid_base64_proof(base_url, args.model, messages)
        test_check_empty_proof(base_url, args.model, messages)
        test_proof_not_reusable_on_different_prompt(base_url, args.model, proof_zipped)
        test_generate_check_roundtrip_twice(base_url, args.model, messages)
    else:
        print("  (test saltati: generate iniziale fallita)")

    # --- FASE 2: GRID DI TEST IN PARALLELO ---
    total = len(PROMPT_GRID)
    print(f"\n--- Fase 2: grid {total} combinazioni "
          f"(thinking×lunghezza×tipo), max {args.workers} in parallelo ---")
    for label, _, _ in PROMPT_GRID:
        print(f"    {label}")
    print()
    t0 = time.perf_counter()
    jobs = [
        (run_suite_for_prompt,
         (base_url, args.model, msgs, thinking, f"{label}", args.timeout))
        for label, thinking, msgs in PROMPT_GRID
    ]
    _parallel(jobs, args.workers)
    elapsed = time.perf_counter() - t0
    print(f"\n  (grid completata in {elapsed:.1f}s)")

    # --- SOMMARIO ---
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    skipped = sum(1 for _, s, _ in results if s == SKIP)

    print(f"\n{'='*60}")
    print(f"  RISULTATI: {passed} PASS  {failed} FAIL  {skipped} SKIP")
    print(f"{'='*60}")
    if failed:
        print("  Test falliti:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"    ✗ {name}: {detail}")
    print()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
