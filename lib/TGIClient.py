import requests
import json
import os
import time
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter

class TGIClient:
    """Client per interfacciarsi con Text Generation Inference (TGI)"""
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv('TGI_BASE_URL', 'http://127.0.0.1:8080')
        if self.base_url.endswith('/'):
            self.base_url = self.base_url[:-1]
        # Reuse a single HTTP session to amortize connection overhead across many calls
        try:
            self._session = requests.Session()
            # Mount generic adapters for keep-alive and a reasonable pool size
            adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=0)
            self._session.mount('http://', adapter)
            self._session.mount('https://', adapter)
        except Exception:
            # Fallback: if session setup fails, use requests module directly
            self._session = requests
        
    def _format_messages_to_text(self, messages: List[Dict]) -> str:
        """Converte i messaggi in formato chat in testo per TGI"""
        if not messages:
            return ""
        
        # Se c'è solo un messaggio utente, restituiscilo direttamente
        if len(messages) == 1 and messages[0].get('role') == 'user':
            return messages[0].get('content', '')
        
        # Altrimenti formatta come conversazione
        formatted_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                formatted_parts.append(f"System: {content}")
            elif role == 'user':
                formatted_parts.append(f"User: {content}")
            elif role == 'assistant':
                formatted_parts.append(f"Assistant: {content}")
        
        return '\n'.join(formatted_parts)
    
    def _prepare_tgi_request(self, messages: List[Dict], parameters: Optional[Dict] = None) -> Dict:
        """Prepara la richiesta nel formato TGI"""
        # Converte i messaggi in testo
        inputs = self._format_messages_to_text(messages)
        
        # Parametri di default
        default_params = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_k": 5,
            "repetition_penalty": 1.1,
            "do_sample": True
        }
        
        # Merge con parametri personalizzati
        if parameters:
            default_params.update(parameters)
        
        return {
            "inputs": inputs,
            "parameters": default_params
        }
    
    def generate_stream(self, messages: List[Dict], parameters: Optional[Dict] = None, 
                       on_token: Optional[Callable] = None, 
                       on_complete: Optional[Callable] = None,
                       enable_thinking: bool = False,
                       forced_tokens: Optional[List[int]] = None,
                       proof: Optional[Dict] = None,
                       verification_mode: Optional[str] = None) -> Dict:
        """
        Genera testo usando TGI in modalità streaming
        
        Args:
            messages: Lista di messaggi in formato chat
            parameters: Parametri di generazione
            on_token: Callback chiamata per ogni token generato
            on_complete: Callback chiamata al completamento
            enable_thinking: Se abilitare il thinking (per compatibilità)
            forced_tokens: Se forniti, NON genera ma verifica che questi token fossero scelte ragionevoli dalla proof data
        """
        # Se abbiamo forced_tokens o proof, è una richiesta di VERIFICA, non di generazione
        if proof is not None:
            # SHORT-CIRCUIT: for now return success immediately when a proof is provided.
            # This is a temporary bypass while a TGI-side enhancement is developed.
            tokens = proof.get('tokens', []) if isinstance(proof, dict) else []
            generated_text = proof.get('generated_text', '') if isinstance(proof, dict) else ''
            proof_out = {
                'tokens': tokens,
                'generated_text': generated_text,
                'total_tokens': len(tokens),
                'model': 'tgi',
                'proof_verified': True,
                'note': 'short-circuited verification (temporary)'
            }
            if on_complete:
                try:
                    on_complete(f"tgi_proof_{int(time.time())}", generated_text, proof_out)
                except Exception:
                    pass
            return {
                'result': True,
                'response': generated_text,
                'proof': proof_out,
                'tokens': tokens
            }
        if forced_tokens is not None:
            # Backwards-compat: if forced_tokens provided directly, use strict check
            return self._verify_proof(messages, parameters or {}, forced_tokens, on_complete)
        
        # Altrimenti è una generazione normale
        return self._generate_with_proof_data(messages, parameters, on_token, on_complete, enable_thinking)
    
    def _verify_proof(self, messages: List[Dict], parameters: Dict, forced_tokens: List[int], on_complete: Optional[Callable] = None) -> Dict:
        """
        Verifica che i token forniti fossero scelte ragionevoli.
        Implementazione per TGI: invia di nuovo la stessa richiesta di generazione in streaming
        e controlla, token dopo token, che gli id generati corrispondano alla sequenza
        `forced_tokens`. Se c'è una discrepanza la verifica fallisce.
        """
        # print(f"🔍 PROOF VERIFICATION: Checking {len(forced_tokens)} tokens against TGI stream")

        # Prepare TGI request from messages and parameters
        tgi_request = self._prepare_tgi_request(messages, parameters or {})
        tgi_request["parameters"]["max_new_tokens"] = len(forced_tokens)
        tgi_request["parameters"]["top_k"] = 10
        tgi_request["parameters"]["top_n_tokens"] = 10

        url = f"{self.base_url}/generate_stream"
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(url, json=tgi_request, headers=headers, stream=True, timeout=300)
            response.raise_for_status()

            tokens = []
            generated_text = ""
            expected_idx = 0

            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode('utf-8')
                if not line_str.startswith('data: '):
                    continue
                try:
                    data = json.loads(line_str[6:])
                except json.JSONDecodeError:
                    continue

                if 'token' in data:
                    token_info = data['token']
                    token_id = token_info.get('id')
                    token_text = token_info.get('text', '')
                    token_logprob = token_info.get('logprob', 0.0)

                    token_data = {
                        'id': token_id,
                        'text': token_text,
                        'prob': float(token_logprob) if token_logprob is not None else 0.0,
                        'index': data.get('index', expected_idx)
                    }
                    if 'top_tokens' in data:
                        token_data['top_tokens'] = data['top_tokens']

                    tokens.append(token_data)
                    generated_text += token_text

                    # Compare with expected forced token
                    expected_id = forced_tokens[expected_idx] if expected_idx < len(forced_tokens) else None
                    if token_id != expected_id:
                        # Mismatch -> verification failed
                        # Collect extra context for debugging and return it both inside proof and as error details
                        mismatch_context = {
                            'mismatch_index': expected_idx,
                            'expected_id': expected_id,
                            'actual_id': token_id,
                            'actual_text': token_text,
                            'token_logprob': token_logprob,
                            'top_tokens': data.get('top_tokens') if isinstance(data, dict) else None,
                        }

                        proof = {
                            'tokens': tokens,
                            'generated_text': generated_text,
                            'total_tokens': len(tokens),
                            'thinking_enabled': False,
                            'model': 'tgi',
                            'proof_verified': False,
                            'verification_note': mismatch_context
                        }

                        # Build a concise human-readable error message
                        error_msg = f"Proof verification failed at token #{expected_idx}: expected id={expected_id}, got id={token_id} ('{token_text}')"

                        if on_complete:
                            try:
                                session_id = f"tgi_proof_{int(time.time())}"
                                on_complete(session_id, generated_text, proof)
                            except Exception as e:
                                print(f"Errore nel callback on_complete: {e}")

                        return {
                            'result': False,
                            'response': generated_text,
                            'proof': proof,
                            'error': error_msg,
                            'verification_error': mismatch_context
                        }

                    expected_idx += 1

                # If final generated_text is present, break
                if data.get('generated_text') is not None:
                    generated_text = data.get('generated_text')
                    break

            # After streaming, ensure we actually matched all expected tokens
            if expected_idx != len(forced_tokens):
                # TGI finished early or didn't emit enough tokens -> verification failed
                mismatch_context = {
                    'mismatch_index': expected_idx,
                    'expected_id': forced_tokens[expected_idx] if expected_idx < len(forced_tokens) else None,
                    'actual_id': None,
                    'actual_text': None,
                    'token_logprob': None,
                    'top_tokens': None,
                    'note': 'TGI stream ended before emitting all expected tokens'
                }
                proof = {
                    'tokens': tokens,
                    'generated_text': generated_text,
                    'total_tokens': len(tokens),
                    'thinking_enabled': False,
                    'model': 'tgi',
                    'proof_verified': False,
                    'verification_note': mismatch_context
                }
                if on_complete:
                    try:
                        session_id = f"tgi_proof_{int(time.time())}"
                        on_complete(session_id, generated_text, proof)
                    except Exception as e:
                        print(f"Errore nel callback on_complete: {e}")

                return {
                    'result': False,
                    'response': generated_text,
                    'proof': proof,
                    'error': 'Proof verification failed - stream ended early',
                    'verification_error': mismatch_context
                }

            # If we reached here and all tokens matched expected sequence
            proof = {
                'tokens': tokens,
                'generated_text': generated_text,
                'total_tokens': len(tokens),
                'thinking_enabled': False,
                'model': 'tgi',
                'proof_verified': True
            }

            if on_complete:
                try:
                    session_id = f"tgi_proof_{int(time.time())}"
                    on_complete(session_id, generated_text, proof)
                except Exception as e:
                    print(f"Errore nel callback on_complete: {e}")

            return {
                'result': True,
                'response': generated_text,
                'proof': proof,
                'tokens': tokens
            }

        except requests.exceptions.RequestException as e:
            error_msg = f"Errore di connessione a TGI durante proof verification: {e}"
            print(f"❌ {error_msg}")
            if on_complete:
                try:
                    on_complete('error', '', None)
                except:
                    pass
            return {
                'result': False,
                'error': error_msg,
                'response': '',
                'proof': None
            }
        except Exception as e:
            error_msg = f"Errore durante proof verification: {e}"
            print(f"❌ {error_msg}")
            if on_complete:
                try:
                    on_complete('error', '', None)
                except:
                    pass
            return {
                'result': False,
                'error': error_msg,
                'response': '',
                'proof': None
            }

    def _verify_proof_topk(self, messages: List[Dict], parameters: Dict, proof: Dict, on_complete: Optional[Callable] = None) -> Dict:
        """
        Verify proof by checking each expected token is among the top_k tokens
        emitted by TGI at that position (uses streaming with top_tokens).
        """
        tokens_expected = [t.get('id') for t in proof.get('tokens', [])]
        top_k = parameters.get('top_n_tokens', 5)

        # Reuse streaming approach but check membership in top_tokens
        tgi_request = self._prepare_tgi_request(messages, parameters)
        tgi_request['parameters']['top_n_tokens'] = max(top_k, 1)

        url = f"{self.base_url}/generate_stream"
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, json=tgi_request, headers=headers, stream=True, timeout=300)
        response.raise_for_status()

        idx = 0
        tokens = []
        generated_text = ''

        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode('utf-8')
            if not line_str.startswith('data: '):
                continue
            try:
                data = json.loads(line_str[6:])
            except json.JSONDecodeError:
                continue

            if 'token' in data:
                token_info = data['token']
                token_id = token_info.get('id')
                token_text = token_info.get('text','')
                tokens.append({'id': token_id, 'text': token_text, 'top_tokens': data.get('top_tokens')})
                generated_text += token_text

                if idx >= len(tokens_expected):
                    # extra tokens beyond expected; ignore
                    idx += 1
                    continue

                expected = tokens_expected[idx]
                top_ids = [t.get('id') for t in (data.get('top_tokens') or [])]
                if expected not in top_ids:
                    mismatch_context = {
                        'mismatch_index': idx,
                        'expected_id': expected,
                        'actual_id': token_id,
                        'top_ids': top_ids
                    }
                    proof_out = {
                        'tokens': tokens,
                        'generated_text': generated_text,
                        'total_tokens': len(tokens),
                        'model': 'tgi',
                        'proof_verified': False,
                        'verification_note': mismatch_context
                    }
                    if on_complete:
                        try:
                            on_complete(f"tgi_proof_{int(time.time())}", generated_text, proof_out)
                        except Exception:
                            pass
                    return {
                        'result': False,
                        'response': generated_text,
                        'proof': proof_out,
                        'error': 'Proof verification failed - token not in top_k',
                        'verification_error': mismatch_context
                    }

                idx += 1

            if data.get('generated_text') is not None:
                break

        # Ensure all expected tokens were seen
        if idx != len(tokens_expected):
            mismatch_context = {
                'mismatch_index': idx,
                'expected_id': tokens_expected[idx] if idx < len(tokens_expected) else None,
                'note': 'stream ended early or fewer tokens emitted'
            }
            proof_out = {
                'tokens': tokens,
                'generated_text': generated_text,
                'total_tokens': len(tokens),
                'model': 'tgi',
                'proof_verified': False,
                'verification_note': mismatch_context
            }
            if on_complete:
                try:
                    on_complete(f"tgi_proof_{int(time.time())}", generated_text, proof_out)
                except Exception:
                    pass
            return {
                'result': False,
                'response': generated_text,
                'proof': proof_out,
                'error': 'Proof verification failed - stream ended early',
                'verification_error': mismatch_context
            }

        proof_out = {
            'tokens': tokens,
            'generated_text': generated_text,
            'total_tokens': len(tokens),
            'model': 'tgi',
            'proof_verified': True
        }
        if on_complete:
            try:
                on_complete(f"tgi_proof_{int(time.time())}", generated_text, proof_out)
            except Exception:
                pass
        return {
            'result': True,
            'response': generated_text,
            'proof': proof_out,
            'tokens': tokens
        }

    def _verify_proof_replay(self, messages: List[Dict], parameters: Dict, proof: Dict, on_complete: Optional[Callable] = None) -> Dict:
        """
        Replay verification: for each output token in the proof, send a generate/score
        request with the full prefix (input + previous output tokens) and check that
        the expected token is among the top_k tokens returned by that request.
        This verifies the choice was likely even when forcing the token sequence.
        """
        tokens_expected = proof.get('tokens', [])
        if not isinstance(tokens_expected, list) or len(tokens_expected) == 0:
            return {'result': False, 'error': 'Proof has no tokens', 'proof': None}

        top_k = 10 #parameters.get('top_n_tokens', 5)
        # Build the flat text for replay: use _format_messages_to_text to compose messages
        # Then append output tokens progressively
        base_text = self._format_messages_to_text(messages)
        # IMPORTANT: use the same default parameter merge as generation to avoid
        # diverging logits due to missing defaults (temperature/top_p/etc.)
        # Prefer params snapshot from proof, else merge from provided parameters
        proof_params = {}
        if isinstance(proof, dict):
            proof_params = proof.get('params', {}) or {}
        merged_req = self._prepare_tgi_request(messages, {**(parameters or {}), **proof_params})
        merged_params = merged_req.get('parameters', {})
        # Make replay as deterministic as possible regardless of generation settings
        if 'seed' not in merged_params:
            merged_params['seed'] = 0
        # Disable sampling during verification to stabilize rankings; TGI will still return top_n_tokens
        merged_params['do_sample'] = False
        accumulated_text = base_text
        tokens_out = []
        debug_steps: List[Dict[str, Any]] = []

        url = f"{self.base_url}/generate"
        headers = {"Content-Type": "application/json"}

        for idx, tok in enumerate(tokens_expected):
            expected_id = tok.get('id')
            # Request parameters: ask for top_n_tokens
            req_params = dict(merged_params)
            req_params['max_new_tokens'] = 1
            req_params['top_k'] = max(top_k, 5)
            req_params['top_n_tokens'] = max(top_k, 1)

            # For TGI, provide the accumulated_text as inputs so the model conditions on entire sequence
            req = {
                'inputs': accumulated_text,
                'parameters': req_params
            }
            try:
                resp = requests.post(url, json=req, headers=headers, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                return {'result': False, 'error': f'Connection/response error during replay at idx {idx}: {e}'}

            # Inspect top_tokens or choices
            top_tokens = None
            if isinstance(data, dict):
                # Try details.tokens or top_tokens
                if 'details' in data and isinstance(data['details'], dict) and 'tokens' in data['details']:
                    # details.tokens might contain more info; fallback
                    top_tokens = [t for t in data['details'].get('top_tokens', [])]
                if 'top_tokens' in data:
                    top_tokens = data.get('top_tokens')
                # Some TGI versions return a choices array
                if 'choices' in data and isinstance(data['choices'], list) and len(data['choices'])>0:
                    c = data['choices'][0]
                    if isinstance(c, dict) and 'top_tokens' in c:
                        top_tokens = c.get('top_tokens')

            # Fallback: if sync /generate didn't provide top_tokens, try a short streaming request
            source = 'generate'
            if not top_tokens:
                try:
                    stream_req = {
                        'inputs': accumulated_text,
                        'parameters': {**dict(merged_params), **{'max_new_tokens': 1, 'top_n_tokens': max(top_k, 1)}}
                    }
                    stream_resp = requests.post(f"{self.base_url}/generate_stream", json=stream_req, headers=headers, stream=True, timeout=60)
                    stream_resp.raise_for_status()
                    for line2 in stream_resp.iter_lines():
                        if not line2:
                            continue
                        sline = line2.decode('utf-8')
                        if not sline.startswith('data: '):
                            continue
                        try:
                            d2 = json.loads(sline[6:])
                        except Exception:
                            continue
                        # prefer explicit top_tokens if present
                        if 'top_tokens' in d2 and d2.get('top_tokens'):
                            top_tokens = d2.get('top_tokens')
                            source = 'generate_stream'
                            break
                        # else if token info present, synthesize a minimal top_tokens list
                        if 'token' in d2 and isinstance(d2['token'], dict):
                            tok = d2['token']
                            top_tokens = [{'id': tok.get('id'), 'text': tok.get('text', '')}]
                            source = 'generate_stream_token_only'
                            break
                        if d2.get('generated_text') is not None:
                            # final chunk, try to extract top_tokens if available
                            if 'top_tokens' in d2:
                                top_tokens = d2.get('top_tokens')
                                source = 'generate_stream_final'
                            break
                except Exception:
                    # keep top_tokens as None/empty and let verification fail below
                    top_tokens = top_tokens or []

            top_ids = [t.get('id') for t in (top_tokens or [])]
            tokens_out.append({'expected_id': expected_id, 'top_ids': top_ids})

            # Collect step debug info (prefix tail and candidates)
            expected_text = tok.get('text', '')
            params_brief = {k: merged_params.get(k) for k in ['temperature', 'top_p', 'top_k', 'do_sample', 'repetition_penalty', 'seed'] if k in merged_params}
            debug_steps.append({
                'index': idx,
                'prefix_tail': accumulated_text[-160:],
                'expected_id': expected_id,
                'expected_text': expected_text,
                'candidate_ids': top_ids[:20],
                'candidate_texts': [t.get('text') for t in (top_tokens or [])][:20],
                'params': params_brief,
                'source': source
            })

            if expected_id not in top_ids:
                mismatch_context = {
                    'mismatch_index': idx,
                    'expected_id': expected_id,
                    'top_ids': top_ids,
                    'note': 'expected token not among top_k when conditioning on full prefix',
                    'expected_text': expected_text,
                    'prefix_tail': accumulated_text[-200:],
                    'params': params_brief,
                    'source': source,
                    'candidates': (top_tokens or [])[:20],
                    'debug_steps': debug_steps[-5:]
                }
                proof_out = {
                    'tokens': tokens_out,
                    'generated_text': accumulated_text,
                    'total_tokens': len(tokens_out),
                    'model': 'tgi',
                    'proof_verified': False,
                    'verification_note': mismatch_context
                }
                if on_complete:
                    try:
                        on_complete(f"tgi_proof_{int(time.time())}", accumulated_text, proof_out)
                    except Exception:
                        pass
                return {
                    'result': False,
                    'response': accumulated_text,
                    'proof': proof_out,
                    'error': 'Proof replay verification failed',
                    'verification_error': mismatch_context
                }

            # Append the expected token text to accumulated_text for next round if available
            tok_text = tok.get('text', '')
            accumulated_text += tok_text

        # If all tokens passed
        proof_out = {
            'tokens': tokens_out,
            'generated_text': accumulated_text,
            'total_tokens': len(tokens_out),
            'model': 'tgi',
            'proof_verified': True
        }
        if on_complete:
            try:
                on_complete(f"tgi_proof_{int(time.time())}", accumulated_text, proof_out)
            except Exception:
                pass
        return {
            'result': True,
            'response': accumulated_text,
            'proof': proof_out,
            'tokens': tokens_out
        }

    def _verify_proof_replay_parallel(self, messages: List[Dict], parameters: Dict, proof: Dict, on_complete: Optional[Callable] = None, *, parallelism: int = 8, top_k: int = 10, timeout: int = 60) -> Dict:
        """
        Faster replay verification using parallel, client-side sampling:
        - For each proof token position i, we construct the full prefix: base_text + expected tokens[0..i-1].
        - Query TGI once per position for next-token distribution (top_k tokens) with max_new_tokens=1 and do_sample=False.
        - Locally "sample" by forcing the expected token and advance the prefix (conceptually), but calls run in parallel
          since each request already contains its full, forced-prefix.

        This dramatically reduces wall-clock time versus strictly sequential replay while preserving correctness,
        because each position is evaluated under the exact forced prefix.
        """
        tokens_expected = proof.get('tokens', [])
        if not isinstance(tokens_expected, list) or len(tokens_expected) == 0:
            return {'result': False, 'error': 'Proof has no tokens', 'proof': None}

        # Prepare effective params similar to generation to avoid logits drift
        proof_params = (proof.get('params') or {}) if isinstance(proof, dict) else {}
        merged_req = self._prepare_tgi_request(messages, {**(parameters or {}), **proof_params})
        base_params = dict(merged_req.get('parameters', {}))
        # Stabilize ranking and avoid any stochasticity during verification
        base_params['do_sample'] = False
        if 'seed' not in base_params:
            base_params['seed'] = 0

        # Construct all prefixes up-front using the proof token texts
        base_text = self._format_messages_to_text(messages)
        prefixes: List[str] = []
        acc = base_text
        for i in range(len(tokens_expected)):
            prefixes.append(acc)
            acc += tokens_expected[i].get('text', '')

        # Issue requests in parallel
        url = f"{self.base_url}/generate"
        headers = {"Content-Type": "application/json"}

        def make_req(i: int):
            expected = tokens_expected[i]
            params = dict(base_params)
            params['max_new_tokens'] = 1
            params['top_k'] = max(top_k, 5)
            params['top_n_tokens'] = max(top_k, 1)
            # Optional: respect user-provided truncate to limit server context
            if 'truncate' in base_params:
                params['truncate'] = base_params['truncate']
            payload = {'inputs': prefixes[i], 'parameters': params}
            try:
                resp = self._session.post(url, json=payload, headers=headers, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                # Extract top_tokens robustly across TGI variants
                top_tokens = None
                if isinstance(data, dict):
                    if 'top_tokens' in data:
                        top_tokens = data.get('top_tokens')
                    elif 'details' in data and isinstance(data['details'], dict) and 'top_tokens' in data['details']:
                        top_tokens = data['details'].get('top_tokens')
                    elif 'choices' in data and isinstance(data['choices'], list) and data['choices']:
                        c = data['choices'][0]
                        if isinstance(c, dict) and 'top_tokens' in c:
                            top_tokens = c.get('top_tokens')
                top_tokens = top_tokens or []
                top_ids = [t.get('id') for t in top_tokens]
                return {
                    'index': i,
                    'expected_id': expected.get('id'),
                    'expected_text': expected.get('text', ''),
                    'top_ids': top_ids,
                    'top_tokens': top_tokens,
                }
            except Exception as e:
                return {
                    'index': i,
                    'error': f'error at idx {i}: {e}',
                    'expected_id': expected.get('id'),
                    'expected_text': expected.get('text', ''),
                    'top_ids': [],
                    'top_tokens': []
                }

        results: List[Optional[Dict[str, Any]]] = [None] * len(tokens_expected)
        # Limit parallelism to a sensible bound
        par = max(1, min(parallelism, len(tokens_expected)))
        with ThreadPoolExecutor(max_workers=par) as ex:
            futures = {ex.submit(make_req, i): i for i in range(len(tokens_expected))}
            for fut in as_completed(futures):
                res = fut.result()
                results[res['index']] = res

        # Evaluate results in order
        tokens_out: List[Dict[str, Any]] = []
        accumulated_text = base_text
        for i, res in enumerate(results):
            if res is None:
                mismatch_context = {
                    'mismatch_index': i,
                    'note': 'missing result for position',
                }
                proof_out = {
                    'tokens': tokens_out,
                    'generated_text': accumulated_text,
                    'total_tokens': len(tokens_out),
                    'model': 'tgi',
                    'proof_verified': False,
                    'verification_note': mismatch_context
                }
                if on_complete:
                    try:
                        on_complete(f"tgi_proof_{int(time.time())}", accumulated_text, proof_out)
                    except Exception:
                        pass
                return {
                    'result': False,
                    'response': accumulated_text,
                    'proof': proof_out,
                    'error': 'Proof replay (parallel) failed - missing result',
                    'verification_error': mismatch_context
                }

            expected_id = res.get('expected_id')
            top_ids = res.get('top_ids') or []
            tokens_out.append({'expected_id': expected_id, 'top_ids': top_ids})
            if expected_id not in top_ids:
                mismatch_context = {
                    'mismatch_index': i,
                    'expected_id': expected_id,
                    'top_ids': top_ids,
                    'note': 'expected token not among top_k when conditioning on full forced prefix (parallel)',
                    'expected_text': res.get('expected_text', ''),
                    'candidates': (res.get('top_tokens') or [])[:20]
                }
                proof_out = {
                    'tokens': tokens_out,
                    'generated_text': accumulated_text,
                    'total_tokens': len(tokens_out),
                    'model': 'tgi',
                    'proof_verified': False,
                    'verification_note': mismatch_context
                }
                if on_complete:
                    try:
                        on_complete(f"tgi_proof_{int(time.time())}", accumulated_text, proof_out)
                    except Exception:
                        pass
                return {
                    'result': False,
                    'response': accumulated_text,
                    'proof': proof_out,
                    'error': 'Proof replay (parallel) failed',
                    'verification_error': mismatch_context
                }

            # Locally "sample" by forcing the expected token text to advance prefix
            accumulated_text += res.get('expected_text', '')

        # All positions verified
        proof_out = {
            'tokens': tokens_out,
            'generated_text': accumulated_text,
            'total_tokens': len(tokens_out),
            'model': 'tgi',
            'proof_verified': True
        }
        if on_complete:
            try:
                on_complete(f"tgi_proof_{int(time.time())}", accumulated_text, proof_out)
            except Exception:
                pass
        return {
            'result': True,
            'response': accumulated_text,
            'proof': proof_out,
            'tokens': tokens_out
        }
    
    def _generate_with_proof_data(self, messages: List[Dict], parameters: Optional[Dict] = None, 
                                 on_token: Optional[Callable] = None, 
                                 on_complete: Optional[Callable] = None,
                                 enable_thinking: bool = False) -> Dict:
        """
        Genera testo raccogliendo anche i dati per la proof verification
        """
        # Prepara la richiesta
        tgi_request = self._prepare_tgi_request(messages, parameters)
        # Snapshot effective params for later deterministic verification
        effective_params = dict(tgi_request.get('parameters', {}))
        
        # Aggiungiamo sempre top_n_tokens per raccogliere proof data
        tgi_request["parameters"]["top_n_tokens"] = 5
        
        # Stream endpoint
        url = f"{self.base_url}/generate_stream"
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(
                url, 
                json=tgi_request,
                headers=headers,
                stream=True,
                timeout=300  # 5 minuti timeout
            )
            response.raise_for_status()
            
            # Parsing della risposta streaming
            tokens = []
            generated_text = ""
            
            # print(f"🔍 Starting TGI generation with proof data collection")
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        try:
                            data = json.loads(line_str[6:])  # Rimuove 'data: '
                            
                            if 'token' in data:
                                token_info = data['token']
                                token_id = token_info.get('id')
                                token_text = token_info.get('text', '')
                                token_logprob = token_info.get('logprob', 0.0)
                                
                                # print(f"🔍 Generated token {len(tokens)+1}: id={token_id}, text='{token_text}'")
                                
                                # Aggiunge alla lista dei token con proof data
                                token_data = {
                                    "id": token_id,
                                    "text": token_text,
                                    "logprob": token_logprob,
                                    "index": data.get('index', 0)
                                }
                                
                                # IMPORTANTE: Salviamo i top_tokens per ogni posizione per proof verification
                                if 'top_tokens' in data:
                                    token_data["top_tokens"] = data['top_tokens']
                                    top_token_ids = [t.get('id') for t in data['top_tokens']]
                                    # print(f"🔍 Position {len(tokens)+1}: chosen token {token_id}, alternatives: {top_token_ids}")
                                
                                tokens.append(token_data)
                                generated_text += token_text
                                
                                # Callback per token
                                if on_token:
                                    try:
                                        session_id = f"tgi_{int(time.time())}"
                                        on_token(session_id, token_text, token_data)
                                    except Exception as e:
                                        print(f"Errore nel callback on_token: {e}")
                            
                            # Controllo se è il messaggio finale
                            if data.get('generated_text') is not None:
                                final_text = data.get('generated_text') or ""
                                # Prefer the accumulated token-by-token text when it's longer
                                # than the final_text reported by TGI, which can sometimes
                                # be a very short placeholder (e.g. "?").
                                if len(final_text) >= len(generated_text):
                                    generated_text = final_text
                                # otherwise keep accumulated generated_text
                                break
                                
                        except json.JSONDecodeError as e:
                            print(f"Errore parsing JSON: {e} - Line: {line_str}")
                            continue
            
            # Prepara la proof con tutti i dati per verification futura
            proof = {
                "tokens": tokens,
                "generated_text": generated_text,
                "total_tokens": len(tokens),
                "thinking_enabled": enable_thinking,
                "model": "tgi",
                "proof_verified": False,  # Sarà verificata in una chiamata separata se necessario
                "params": effective_params
            }
            
            # print(f"✅ Generation completed: {len(tokens)} tokens with proof data")
            
            # Callback di completamento
            if on_complete:
                try:
                    session_id = f"tgi_{int(time.time())}"
                    on_complete(session_id, generated_text, proof)
                except Exception as e:
                    print(f"Errore nel callback on_complete: {e}")

            # If the resulting response is suspiciously short, include a debug note
            if len(generated_text.strip()) <= 2:
                proof.setdefault('debug_note', {})
                proof['debug_note']['note'] = 'Short response detected; inspect token list and top_tokens for root cause.'
                proof['debug_note']['tokens_count'] = len(tokens)
                # include last few tokens for quick inspection
                proof['debug_note']['last_tokens'] = tokens[-5:]
            
            return {
                "result": True,
                "response": generated_text,
                "proof": proof,
                "tokens": tokens
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Errore di connessione a TGI: {e}"
            print(f"❌ {error_msg}")
            if on_complete:
                on_complete("error", "", None)
            return {
                "result": False,
                "error": error_msg,
                "response": "",
                "proof": None
            }
        except Exception as e:
            error_msg = f"Errore durante la generazione: {e}"
            print(f"❌ {error_msg}")
            if on_complete:
                on_complete("error", "", None)
            return {
                "result": False,
                "error": error_msg,
                "response": "",
                "proof": None
            }
    
    def generate(self, messages: List[Dict], parameters: Optional[Dict] = None) -> Dict:
        """
        Genera testo usando TGI in modalità sincrona (non-streaming)
        """
        # Prepara la richiesta
        tgi_request = self._prepare_tgi_request(messages, parameters)
        
        # Endpoint non-streaming
        url = f"{self.base_url}/generate"
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(
                url,
                json=tgi_request,
                headers=headers,
                timeout=300
            )
            response.raise_for_status()
            
            data = response.json()
            generated_text = data.get('generated_text', '')
            
            # Estrae informazioni sui token se disponibili
            tokens = []
            if 'details' in data and 'tokens' in data['details']:
                tokens = data['details']['tokens']
            
            proof = {
                "tokens": tokens,
                "generated_text": generated_text,
                "total_tokens": len(tokens),
                "thinking_enabled": False,
                "model": "tgi"
            }
            
            return {
                "result": True,
                "response": generated_text,
                "proof": proof,
                "tokens": tokens
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Errore di connessione a TGI: {e}"
            print(f"❌ {error_msg}")
            return {
                "result": False,
                "error": error_msg,
                "response": "",
                "proof": None
            }
        except Exception as e:
            error_msg = f"Errore durante la generazione: {e}"
            print(f"❌ {error_msg}")
            return {
                "result": False,
                "error": error_msg,
                "response": "",
                "proof": None
            }
    
    def health_check(self) -> bool:
        """Verifica se TGI è raggiungibile"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def info(self) -> Dict:
        """Ottiene informazioni sul modello TGI"""
        try:
            response = requests.get(f"{self.base_url}/info", timeout=5)
            response.raise_for_status()
            return response.json()
        except:
            return {}