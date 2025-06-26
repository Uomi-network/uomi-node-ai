#!/usr/bin/env python3
"""
Debug script to monitor lock contention and deadlock detection.

This script helps identify potential deadlock situations by monitoring
lock acquisition times and thread states.

Usage:
    python deadlock_monitor.py
"""

import threading
import time
import sys
import traceback
from contextlib import contextmanager

class LockMonitor:
    def __init__(self, name, timeout=30.0):
        self.name = name
        self.timeout = timeout
        self.lock = threading.Lock()
        self.acquisition_times = []
        self.current_holder = None
        
    @contextmanager
    def monitored_lock(self):
        """Context manager that monitors lock acquisition and warns about long holds"""
        start_time = time.time()
        thread_name = threading.current_thread().name
        
        print(f"[{thread_name}] Attempting to acquire {self.name} lock...")
        
        # Try to acquire with timeout
        acquired = self.lock.acquire(timeout=self.timeout)
        
        if not acquired:
            print(f"[{thread_name}] WARNING: Failed to acquire {self.name} lock within {self.timeout}s!")
            print(f"[{thread_name}] Current holder: {self.current_holder}")
            print(f"[{thread_name}] Stack trace:")
            traceback.print_stack()
            raise TimeoutError(f"Failed to acquire {self.name} lock within {self.timeout}s")
        
        acquisition_time = time.time() - start_time
        self.acquisition_times.append(acquisition_time)
        self.current_holder = thread_name
        
        if acquisition_time > 1.0:
            print(f"[{thread_name}] WARNING: {self.name} lock took {acquisition_time:.2f}s to acquire")
        
        print(f"[{thread_name}] Acquired {self.name} lock in {acquisition_time:.3f}s")
        
        try:
            hold_start = time.time()
            yield
        finally:
            hold_time = time.time() - hold_start
            if hold_time > 5.0:
                print(f"[{thread_name}] WARNING: Held {self.name} lock for {hold_time:.2f}s")
            
            self.current_holder = None
            self.lock.release()
            print(f"[{thread_name}] Released {self.name} lock after {hold_time:.3f}s")
    
    def get_stats(self):
        if not self.acquisition_times:
            return "No lock acquisitions recorded"
        
        avg_time = sum(self.acquisition_times) / len(self.acquisition_times)
        max_time = max(self.acquisition_times)
        total_acquisitions = len(self.acquisition_times)
        
        return f"{self.name} Lock Stats: {total_acquisitions} acquisitions, avg: {avg_time:.3f}s, max: {max_time:.3f}s"

def detect_deadlock_in_runner():
    """
    Example of how to detect potential deadlock patterns in the runner
    """
    print("Deadlock Detection Tips for Runner:")
    print("1. Monitor lock hold times - should be < 1s typically")
    print("2. Watch for nested lock acquisitions")
    print("3. Check if sleep() calls are inside locks")
    print("4. Verify exception handling releases locks")
    print("5. Use timeouts on lock acquisitions")
    print()
    
    # Example patterns to watch for:
    deadlock_patterns = [
        "Lock held during model inference (can be minutes)",
        "Sleep inside lock (blocks other threads unnecessarily)", 
        "Nested locks: executor -> queue -> executor",
        "Exception during batch processing (lock not released)",
        "Long-running callbacks holding locks"
    ]
    
    for i, pattern in enumerate(deadlock_patterns, 1):
        print(f"{i}. {pattern}")

if __name__ == "__main__":
    detect_deadlock_in_runner()
    
    # Example usage of LockMonitor
    print("\nExample of monitored lock usage:")
    
    queue_lock = LockMonitor("queue", timeout=5.0)
    executor_lock = LockMonitor("executor", timeout=5.0)
    
    def worker_thread(name, delay):
        try:
            with queue_lock.monitored_lock():
                print(f"Worker {name} doing queue work...")
                time.sleep(delay)
                
                with executor_lock.monitored_lock():
                    print(f"Worker {name} doing executor work...")
                    time.sleep(delay)
        except Exception as e:
            print(f"Worker {name} error: {e}")
    
    # Simulate concurrent access
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker_thread, args=(f"T{i}", 0.1), name=f"Worker-{i}")
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    print("\nLock Statistics:")
    print(queue_lock.get_stats())
    print(executor_lock.get_stats())
