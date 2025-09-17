import threading
import time
import uuid
import os
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Any
from dataclasses import dataclass
from lib.config import BATCH_MAX_SIZE, TRANSFORMERS_INFERENCE_TEMPERATURE, TRANSFORMERS_INFERENCE_MAX_TOKENS

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: torch not available, FastContinuousBatcher will not work")


@dataclass
class BatchRequest:
    id: str
    messages: List[Dict]
    enable_thinking: bool
    sampling_cfg: Dict
    max_new_tokens: int
    on_token: Callable
    on_complete: Callable
    is_check: bool = False
    forced_tokens: Optional[List[int]] = None
    arrival_time: float = 0.0


class FastContinuousBatcher:
    """
    High-performance continuous batcher inspired by TGI patterns but preserving probability collection.
    Uses transformers.generate() efficiently while supporting continuous batching.
    """
    
    def __init__(self, model, tokenizer, device, max_active: int = BATCH_MAX_SIZE):
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for FastContinuousBatcher")
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_active = max_active
        self.pending: Deque[BatchRequest] = deque()
        self.active: Dict[str, BatchRequest] = {}
        self.lock = threading.Lock()
        self.kill = False
        
        # Batch processing config
        self.batch_window_ms = int(os.getenv("BATCH_WINDOW_MS", "50"))  # Window to collect requests
        self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", str(max_active)))
        
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
    
    def submit(self, messages, enable_thinking: bool, sampling_cfg: Dict, max_new_tokens: Optional[int], 
               on_token: Callable, on_complete: Callable, is_check: bool = False, forced_tokens: Optional[List[int]] = None):
        """Submit a new request for processing"""
        request_id = uuid.uuid4().hex
        request = BatchRequest(
            id=request_id,
            messages=messages,
            enable_thinking=enable_thinking,
            sampling_cfg=sampling_cfg,
            max_new_tokens=max_new_tokens or TRANSFORMERS_INFERENCE_MAX_TOKENS,
            on_token=on_token,
            on_complete=on_complete,
            is_check=is_check,
            forced_tokens=forced_tokens,
            arrival_time=time.time()
        )
        
        with self.lock:
            self.pending.append(request)
        
        return request_id
    
    def stop(self):
        self.kill = True
        self.thread.join()
    
    def _processing_loop(self):
        """Main processing loop - inspired by TGI's continuous batching"""
        while not self.kill:
            with self.lock:
                # Collect pending requests for batching
                batch_requests = self._collect_batch()
            
            if not batch_requests:
                time.sleep(0.001)  # 1ms sleep when idle
                continue
            
            try:
                # Process the batch using transformers.generate()
                self._process_batch(batch_requests)
            except Exception as e:
                print(f"[fast_batcher] Error processing batch: {e}")
                # Mark all requests as failed
                for req in batch_requests:
                    req.on_complete(req.id, "", {"error": str(e), "tokens": []})
    
    def _collect_batch(self) -> List[BatchRequest]:
        """Collect requests for batching within the time window"""
        if not self.pending:
            return []
        
        batch = []
        deadline = time.time() + (self.batch_window_ms / 1000.0)
        
        # Collect requests until deadline or max batch size
        while self.pending and len(batch) < self.max_batch_size:
            batch.append(self.pending.popleft())
            
            # If we have some requests and deadline passed, break
            if batch and time.time() >= deadline:
                break
        
        return batch
    
    def _process_batch(self, requests: List[BatchRequest]):
        """Process a batch of requests using transformers.generate()"""
        if not requests:
            return
        
        # Prepare batch inputs
        texts = []
        for req in requests:
            text = self.tokenizer.apply_chat_template(
                req.messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=req.enable_thinking
            )
            texts.append(text)
        
        # Tokenize with padding
        tokenized = self.tokenizer(texts, padding=True, return_tensors="pt")
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        
        # Determine generation parameters (use the most common settings from the batch)
        max_new_tokens = max(req.max_new_tokens for req in requests)
        temperature = requests[0].sampling_cfg.get('temperature', TRANSFORMERS_INFERENCE_TEMPERATURE)
        top_k = requests[0].sampling_cfg.get('top_k', 5)
        do_sample = not all(req.sampling_cfg.get('deterministic', False) for req in requests)
        
        # Generation config
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "do_sample": do_sample,
            "use_cache": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,  # CRITICAL: This gives us the probabilities!
        }
        
        # Run generation
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch required for generation")
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                **generation_config
            )
        
        # Extract results
        generated_sequences = outputs.sequences
        scores = outputs.scores  # This contains the logits for each step
        
        # Process each request's results
        for i, req in enumerate(requests):
            self._process_request_result(req, i, input_ids[i], generated_sequences[i], scores)
    
    def _process_request_result(self, request: BatchRequest, batch_idx: int, 
                               input_ids, generated_sequence, 
                               scores):
        """Process the result for a single request and extract probabilities"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch required for processing results")
            
        prompt_length = len(input_ids)
        generated_tokens = generated_sequence[prompt_length:]
        
        # Extract probabilities for each generated token
        prob_tokens = []
        for token_idx, token_id in enumerate(generated_tokens):
            if token_idx < len(scores):
                # Get logits for this position and batch index
                token_logits = scores[token_idx][batch_idx]
                
                # Convert to probabilities
                token_probs = torch.softmax(token_logits, dim=-1)
                
                # Get token probability and rank
                token_prob = token_probs[token_id].item()
                
                # Get top tokens to find rank
                top_probs, top_indices = token_probs.topk(10)
                rank = -1
                for r, idx in enumerate(top_indices):
                    if idx.item() == token_id.item():
                        rank = r
                        break
                
                prob_tokens.append({
                    "id": token_id.item(),
                    "prob": token_prob,
                    "index": rank
                })
            else:
                # Fallback for tokens beyond available scores
                prob_tokens.append({
                    "id": token_id.item(),
                    "prob": 0.0,
                    "index": -1
                })
        
        # Decode the response
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Stream tokens if callback provided
        if request.on_token:
            for i, token_info in enumerate(prob_tokens):
                token_text = self.tokenizer.decode([token_info["id"]], skip_special_tokens=True)
                request.on_token(request.id, token_text, token_info)
        
        # Complete the request
        proof = {
            "tokens": prob_tokens,
            "full_sequence_length": len(generated_sequence)
        }
        
        request.on_complete(request.id, response, proof)
    
    def status(self):
        """Get current batcher status"""
        with self.lock:
            return {
                'pending': len(self.pending),
                'active': len(self.active),
                'batch_window_ms': self.batch_window_ms,
                'max_batch_size': self.max_batch_size
            }