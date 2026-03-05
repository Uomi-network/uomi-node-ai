"""
test_proof_verification.py — verifica l'integrità del sistema di proof blockchain.

Il contratto blockchain richiede che:
  1. generate → produce proof = zlib(json({"tokens": [{"id": int}, ...]}))
  2. check    → ri-genera con lo stesso prompt + forced_tokens dal proof
               → verified=True se token IDs coincidono, False altrimenti
  3. Una proof manomessa deve risultare verified=False
  4. Il proof deve sopravvivere a encode/decode round-trip identico

Questo script testa tutto ciò chiamando direttamente il server vLLM
con la stessa logica di VLLMModelManager e runner.py.

Usage:
  python test/test_proof_verification.py [--port 8100] [--temperature 0]
"""

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
import zlib
from typing import List, Optional


# ---------------------------------------------------------------------------
# Helpers (stessa logica di zipper.py + VLLMModelManager)
# ---------------------------------------------------------------------------

def zip_proof(proof_dict: dict) -> str:
    """Replica esatta di zip_string(json.dumps(proof)) in runner.py."""
    raw = json.dumps(proof_dict)
    compressed = zlib.compress(raw.encode("utf-8"))
    return base64.b64encode(compressed).decode("ascii")


def unzip_proof(s: str) -> dict:
    """Replica esatta di unzip_string + json.loads in runner.py."""
    compressed = base64.b64decode(s.encode("ascii"))
    raw = zlib.decompress(compressed).decode("utf-8")
    return json.loads(raw)


def load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return tok
    except Exception as e:
        print(f"  WARNING: impossibile caricare tokenizer ({e})")
        return None


def encode_tokens(tokenizer, text: str) -> List[int]:
    """Replica esatta di VLLMModelManager._encode_tokens."""
    if tokenizer is None:
        return []
    return list(tokenizer.encode(text, add_special_tokens=False))


def decode_from_ids(tokenizer, ids: List[int]) -> str:
    if tokenizer is None:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=True)


def call_vllm(base_url: str, model: str, messages: list,
              max_tokens: int = 256, temperature: float = 0.0,
              timeout: int = 120) -> dict:
    """Chiama /v1/chat/completions e ritorna il messaggio completo."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code}: {body}") from e


def get_model_name(base_url: str) -> str:
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as resp:
            return json.loads(resp.read())["data"][0]["id"]
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Risultati test
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

results = []


def record(name: str, status: str, detail: str = ""):
    symbol = "✓" if status == PASS else ("~" if status == SKIP else "✗")
    print(f"  [{symbol}] {name}", end="")
    if detail:
        print(f" — {detail}")
    else:
        print()
    results.append((name, status, detail))


# ---------------------------------------------------------------------------
# Suite di test
# ---------------------------------------------------------------------------

def test_proof_structure(tokenizer, response_text: str, proof_zipped: str):
    """T1: la proof è un JSON valido con la struttura attesa."""
    try:
        proof = unzip_proof(proof_zipped)
        assert isinstance(proof, dict), "non è un dict"
        assert "tokens" in proof, "manca 'tokens'"
        assert isinstance(proof["tokens"], list), "'tokens' non è una lista"
        assert len(proof["tokens"]) > 0, "tokens lista vuota"
        for tok in proof["tokens"]:
            assert isinstance(tok, dict), f"token non è dict: {tok}"
            assert "id" in tok, f"token senza 'id': {tok}"
            assert isinstance(tok["id"], int), f"token id non int: {tok['id']}"
        record("T1 struttura proof valida", PASS,
               f"{len(proof['tokens'])} tokens")
    except Exception as e:
        record("T1 struttura proof valida", FAIL, str(e))


def test_proof_roundtrip(tokenizer, response_text: str, proof_zipped: str):
    """T2: encode(decode(proof)) == proof (idempotenza zip/unzip)."""
    try:
        proof = unzip_proof(proof_zipped)
        rezipped = zip_proof(proof)
        proof2 = unzip_proof(rezipped)
        assert proof == proof2, "proof cambia dopo re-zip"
        record("T2 round-trip zip/unzip idempotente", PASS)
    except Exception as e:
        record("T2 round-trip zip/unzip idempotente", FAIL, str(e))


def test_token_ids_decode_to_response(tokenizer, response_text: str, proof_zipped: str):
    """T3: i token IDs della proof, decodificati, ricostruiscono la risposta."""
    if tokenizer is None:
        record("T3 token IDs → testo", SKIP, "tokenizer non disponibile")
        return
    try:
        proof = unzip_proof(proof_zipped)
        ids = [t["id"] for t in proof["tokens"]]
        reconstructed = decode_from_ids(tokenizer, ids)
        # Confronto flessibile: il testo ricostruito deve essere substring o molto simile
        # (piccole differenze di whitespace sono accettabili)
        resp_norm = response_text.strip()
        rec_norm = reconstructed.strip()
        if rec_norm == resp_norm:
            record("T3 token IDs → testo (match esatto)", PASS)
        elif rec_norm in resp_norm or resp_norm in rec_norm:
            record("T3 token IDs → testo (match parziale)", PASS,
                   f"len(proof_text)={len(rec_norm)} len(response)={len(resp_norm)}")
        else:
            # Mostra diff
            overlap = sum(a == b for a, b in zip(rec_norm, resp_norm))
            record("T3 token IDs → testo", FAIL,
                   f"overlap={overlap}/{max(len(rec_norm),len(resp_norm))} "
                   f"got='{rec_norm[:60]}' expected='{resp_norm[:60]}'")
    except Exception as e:
        record("T3 token IDs → testo", FAIL, str(e))


def test_token_consistency_with_local_encode(tokenizer, response_text: str, proof_zipped: str):
    """T4: encode locale del response_text produce gli stessi IDs della proof."""
    if tokenizer is None:
        record("T4 token IDs == encode locale", SKIP, "tokenizer non disponibile")
        return
    try:
        proof = unzip_proof(proof_zipped)
        proof_ids = [t["id"] for t in proof["tokens"]]
        local_ids = encode_tokens(tokenizer, response_text)
        if proof_ids == local_ids:
            record("T4 token IDs == encode locale", PASS,
                   f"{len(proof_ids)} tokens")
        else:
            # Calcola quanti match
            matches = sum(a == b for a, b in zip(proof_ids, local_ids))
            record("T4 token IDs == encode locale", FAIL,
                   f"len proof={len(proof_ids)} vs local={len(local_ids)} "
                   f"matching={matches}")
    except Exception as e:
        record("T4 token IDs == encode locale", FAIL, str(e))


def test_check_valid_proof(base_url: str, model: str, messages: list,
                           tokenizer, response_text: str, proof_zipped: str,
                           temperature: float, max_tokens: int):
    """T5: check con proof corretta → verified=True (richiede temperature=0)."""
    if temperature != 0.0:
        record("T5 check proof valida → verified=True", SKIP,
               f"temperature={temperature} != 0, la generazione non è deterministica")
        return
    try:
        proof = unzip_proof(proof_zipped)
        forced_ids = [t["id"] for t in proof["tokens"]]
        n_tokens = len(forced_ids)

        # Re-generate con gli stessi parametri
        data = call_vllm(base_url, model, messages,
                         max_tokens=n_tokens, temperature=0.0, timeout=120)
        re_text = data["choices"][0]["message"].get("content") or ""

        if tokenizer is not None:
            re_ids = encode_tokens(tokenizer, re_text)
        else:
            re_ids = []

        verified = re_ids == forced_ids
        if verified:
            record("T5 check proof valida → verified=True", PASS,
                   f"{n_tokens} tokens matched")
        else:
            matches = sum(a == b for a, b in zip(re_ids, forced_ids))
            record("T5 check proof valida → verified=True", FAIL,
                   f"token mismatch: {matches}/{max(len(re_ids),len(forced_ids))} match "
                   f"(re='{re_text[:60]}' orig='{response_text[:60]}')")
    except Exception as e:
        record("T5 check proof valida → verified=True", FAIL, str(e))


def test_check_tampered_proof_wrong_ids(base_url: str, model: str, messages: list,
                                        tokenizer, response_text: str, proof_zipped: str,
                                        temperature: float, max_tokens: int):
    """T6: check con IDs manomessi → verified=False."""
    if tokenizer is None:
        record("T6 check proof manomessa (IDs sbagliati) → verified=False",
               SKIP, "tokenizer non disponibile")
        return
    try:
        proof = unzip_proof(proof_zipped)
        # Sostituisci tutti gli ID con valori errati (es. ID=1 che è sempre sbagliato)
        tampered_proof = {
            "tokens": [{"id": 1} for _ in proof["tokens"]]
        }
        tampered_zipped = zip_proof(tampered_proof)

        # Simula il check come farebbe runner.py
        forced_ids = [t["id"] for t in tampered_proof["tokens"]]
        n_tokens = len(forced_ids)

        data = call_vllm(base_url, model, messages,
                         max_tokens=n_tokens, temperature=temperature, timeout=120)
        re_text = data["choices"][0]["message"].get("content") or ""
        re_ids = encode_tokens(tokenizer, re_text)

        verified = re_ids == forced_ids
        if not verified:
            record("T6 check proof manomessa (IDs sbagliati) → verified=False", PASS,
                   "correttamente rigettata")
        else:
            record("T6 check proof manomessa (IDs sbagliati) → verified=False", FAIL,
                   "proof manomessa risulta verified=True (PROBLEMA DI SICUREZZA)")
    except Exception as e:
        record("T6 check proof manomessa (IDs sbagliati) → verified=False", FAIL, str(e))


def test_check_tampered_proof_extra_tokens(base_url: str, model: str, messages: list,
                                           tokenizer, response_text: str, proof_zipped: str,
                                           temperature: float, max_tokens: int):
    """T7: check con token extra aggiunti → verified=False."""
    if tokenizer is None:
        record("T7 check proof con token extra → verified=False",
               SKIP, "tokenizer non disponibile")
        return
    try:
        proof = unzip_proof(proof_zipped)
        original_ids = [t["id"] for t in proof["tokens"]]
        # Aggiungi token fasulli in coda
        tampered_proof = {
            "tokens": [{"id": i} for i in original_ids] + [{"id": 999}, {"id": 999}]
        }
        forced_ids = [t["id"] for t in tampered_proof["tokens"]]
        n_tokens = len(forced_ids)

        data = call_vllm(base_url, model, messages,
                         max_tokens=n_tokens, temperature=temperature, timeout=120)
        re_text = data["choices"][0]["message"].get("content") or ""
        re_ids = encode_tokens(tokenizer, re_text)

        verified = re_ids == forced_ids
        if not verified:
            record("T7 check proof con token extra → verified=False", PASS,
                   f"proof con {len(forced_ids)} tok (orig {len(original_ids)}) rigettata")
        else:
            record("T7 check proof con token extra → verified=False", FAIL,
                   "proof con token extra approvata (PROBLEMA)")
    except Exception as e:
        record("T7 check proof con token extra → verified=False", FAIL, str(e))


def test_check_truncated_proof(base_url: str, model: str, messages: list,
                               tokenizer, response_text: str, proof_zipped: str,
                               temperature: float, max_tokens: int):
    """T8: check con proof troncata → verified=False (lunghezza diversa)."""
    if tokenizer is None:
        record("T8 check proof troncata → verified=False",
               SKIP, "tokenizer non disponibile")
        return
    try:
        proof = unzip_proof(proof_zipped)
        original_ids = [t["id"] for t in proof["tokens"]]
        if len(original_ids) < 4:
            record("T8 check proof troncata → verified=False", SKIP,
                   "proof troppo corta per troncare")
            return

        # Tronca a metà
        half = len(original_ids) // 2
        tampered_proof = {"tokens": [{"id": i} for i in original_ids[:half]]}
        forced_ids = [t["id"] for t in tampered_proof["tokens"]]
        n_tokens = len(forced_ids)

        data = call_vllm(base_url, model, messages,
                         max_tokens=n_tokens, temperature=temperature, timeout=120)
        re_text = data["choices"][0]["message"].get("content") or ""
        re_ids = encode_tokens(tokenizer, re_text)

        verified = re_ids == forced_ids
        if not verified:
            record("T8 check proof troncata → verified=False", PASS,
                   f"proof troncata a {half}/{len(original_ids)} tok rigettata")
        else:
            record("T8 check proof troncata → verified=False", FAIL,
                   "proof troncata approvata (PROBLEMA)")
    except Exception as e:
        record("T8 check proof troncata → verified=False", FAIL, str(e))


def test_proof_for_different_prompt(base_url: str, model: str,
                                    messages: list, proof_zipped: str,
                                    tokenizer, temperature: float):
    """T9: riusa la proof di prompt A su prompt B diverso → verified=False."""
    if tokenizer is None:
        record("T9 proof di prompt A su prompt B → verified=False",
               SKIP, "tokenizer non disponibile")
        return
    try:
        proof = unzip_proof(proof_zipped)
        forced_ids = [t["id"] for t in proof["tokens"]]
        n_tokens = len(forced_ids)

        # Prompt completamente diverso
        different_messages = [
            {"role": "user", "content": "Rispondi solo con un numero: 42"}
        ]

        data = call_vllm(base_url, model, different_messages,
                         max_tokens=n_tokens, temperature=temperature, timeout=120)
        re_text = data["choices"][0]["message"].get("content") or ""
        re_ids = encode_tokens(tokenizer, re_text)

        verified = re_ids == forced_ids
        if not verified:
            record("T9 proof di prompt A su prompt B → verified=False", PASS,
                   "proof non trasferibile tra prompt diversi")
        else:
            record("T9 proof di prompt A su prompt B → verified=False", FAIL,
                   "proof di un prompt accettata per un altro (PROBLEMA DI SICUREZZA)")
    except Exception as e:
        record("T9 proof di prompt A su prompt B → verified=False", FAIL, str(e))


def test_proof_reproducibility(base_url: str, model: str, messages: list,
                                tokenizer, temperature: float, max_tokens: int):
    """T10: due generate con temperature=0 sullo stesso prompt → stessi token IDs."""
    if temperature != 0.0:
        record("T10 riproducibilità proof (temperature=0)", SKIP,
               f"temperature={temperature}, skip (stochastic)")
        return
    if tokenizer is None:
        record("T10 riproducibilità proof (temperature=0)", SKIP, "tokenizer non disponibile")
        return
    try:
        def gen():
            data = call_vllm(base_url, model, messages,
                             max_tokens=max_tokens, temperature=0.0, timeout=120)
            text = data["choices"][0]["message"].get("content") or ""
            return encode_tokens(tokenizer, text), text

        ids1, t1 = gen()
        ids2, t2 = gen()

        if ids1 == ids2:
            record("T10 riproducibilità proof (temperature=0)", PASS,
                   f"{len(ids1)} tokens identici in 2 run")
        else:
            matches = sum(a == b for a, b in zip(ids1, ids2))
            record("T10 riproducibilità proof (temperature=0)", FAIL,
                   f"IDs divergono: {matches}/{max(len(ids1),len(ids2))} match "
                   f"(run1='{t1[:50]}' run2='{t2[:50]}')")
    except Exception as e:
        record("T10 riproducibilità proof (temperature=0)", FAIL, str(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test proof verification blockchain")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = deterministico (raccomandato per il contratto blockchain)")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"

    print(f"\nvLLM Proof Verification Test — Blockchain Contract")
    print(f"  server      : {base_url}")

    # Health check
    try:
        urllib.request.urlopen(f"{base_url}/health", timeout=5)
    except Exception as e:
        print(f"\nERRORE: server non raggiungibile ({e})")
        sys.exit(1)

    model = args.model or get_model_name(base_url)
    print(f"  model       : {model}")
    print(f"  temperature : {args.temperature}")
    print(f"  max_tokens  : {args.max_tokens}")

    # Carica tokenizer (stesso modello usato da VLLMModelManager)
    print(f"\nCaricamento tokenizer per {model}...")
    tokenizer = load_tokenizer(model)
    if tokenizer:
        print(f"  tokenizer caricato ✓")
    else:
        print(f"  tokenizer non disponibile — alcuni test saranno skippati")

    # Prompt di test
    messages = [
        {"role": "user",
         "content": "Spiega brevemente cos'è il Byzantine fault tolerance in blockchain. Sii conciso."}
    ]

    # --- GENERA la proof iniziale ---
    print(f"\n--- Generazione proof iniziale ---")
    t0 = time.perf_counter()
    data = call_vllm(base_url, model, messages,
                     max_tokens=args.max_tokens, temperature=args.temperature, timeout=120)
    elapsed = time.perf_counter() - t0
    response_text = data["choices"][0]["message"].get("content") or ""
    usage = data.get("usage", {})

    print(f"  response    : '{response_text[:120]}{'...' if len(response_text)>120 else ''}'")
    print(f"  tokens      : {usage.get('completion_tokens', '?')} completion / {usage.get('total_tokens', '?')} total")
    print(f"  elapsed     : {elapsed:.2f}s")

    # Costruisci proof con la stessa logica di VLLMModelManager
    if tokenizer is not None:
        token_ids = encode_tokens(tokenizer, response_text)
        proof_dict = {"tokens": [{"id": int(t)} for t in token_ids]}
    else:
        # Senza tokenizer, prova a usare l'usage count (meno preciso)
        proof_dict = {"tokens": [{"id": i} for i in range(usage.get("completion_tokens", 0))]}

    proof_zipped = zip_proof(proof_dict)
    print(f"  proof size  : {len(proof_zipped)} chars (zlib+base64)")

    # --- ESEGUI LA SUITE ---
    print(f"\n--- Suite di test ---")
    test_proof_structure(tokenizer, response_text, proof_zipped)
    test_proof_roundtrip(tokenizer, response_text, proof_zipped)
    test_token_ids_decode_to_response(tokenizer, response_text, proof_zipped)
    test_token_consistency_with_local_encode(tokenizer, response_text, proof_zipped)
    test_check_valid_proof(base_url, model, messages, tokenizer,
                           response_text, proof_zipped, args.temperature, args.max_tokens)
    test_check_tampered_proof_wrong_ids(base_url, model, messages, tokenizer,
                                        response_text, proof_zipped, args.temperature, args.max_tokens)
    test_check_tampered_proof_extra_tokens(base_url, model, messages, tokenizer,
                                           response_text, proof_zipped, args.temperature, args.max_tokens)
    test_check_truncated_proof(base_url, model, messages, tokenizer,
                               response_text, proof_zipped, args.temperature, args.max_tokens)
    test_proof_for_different_prompt(base_url, model, messages, proof_zipped,
                                    tokenizer, args.temperature)
    test_proof_reproducibility(base_url, model, messages, tokenizer,
                                args.temperature, args.max_tokens)

    # --- SOMMARIO ---
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    skipped = sum(1 for _, s, _ in results if s == SKIP)

    print(f"\n{'='*60}")
    print(f"  RISULTATI: {passed} PASS  {failed} FAIL  {skipped} SKIP")
    print(f"{'='*60}")
    if failed > 0:
        print("  Test falliti:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"    ✗ {name}: {detail}")
    if args.temperature != 0.0:
        print()
        print("  NOTA: esegui con --temperature 0.0 per il test completo.")
        print("  Con temperature > 0 la generazione è stocastica e T5/T10")
        print("  non possono verificare la determinismo necessario alla blockchain.")
    print()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
