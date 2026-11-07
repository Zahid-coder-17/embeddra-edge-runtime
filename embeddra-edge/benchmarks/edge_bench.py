"""
Embeddra EdgeBench - Automated Edge AI Benchmarking Suite
Measures Time to First Token (TTFT), tokens/sec, KV cache growth, prompt processing throughput,
and memory utilization on the host and edge targets.
"""

import os
import sys
import time
import psutil
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig


def run_benchmark(checkpoint_path: str, prompt_lens=[16, 64, 128], gen_len=30):
    print("=" * 65)
    print("EMBEDDRA EDGEBENCH: MARIA-1B (HYPERGQE) PERFORMANCE AUDIT")
    print("=" * 65)

    process = psutil.Process(os.getpid())
    ram_initial_mb = process.memory_info().rss / (1024 * 1024)

    # 1. Model Loading Benchmark
    t0 = time.time()
    model = MariaForCausalLM.from_pretrained_checkpoint(checkpoint_path, device="cpu", dtype=torch.float32)
    load_time_sec = time.time() - t0
    ram_post_load_mb = process.memory_info().rss / (1024 * 1024)
    model_ram_mb = ram_post_load_mb - ram_initial_mb

    print(f"Model Load Time:          {load_time_sec:.2f} s")
    print(f"Host Initial RAM:         {ram_initial_mb:.1f} MB")
    print(f"Model Memory Allocation:  {model_ram_mb:.1f} MB ({model_ram_mb / 1024:.2f} GB)")

    results = []

    # 2. Prefill & Generation Benchmark across prompt lengths
    for p_len in prompt_lens:
        print(f"\n--- Benchmarking Prompt Length: {p_len} tokens ---")
        prompt_ids = torch.randint(100, 5000, (1, p_len), dtype=torch.long)

        # Measure TTFT (Prompt processing)
        t_prefill_start = time.time()
        with torch.no_grad():
            logits, kv_cache = model(prompt_ids, use_cache=True)
        t_first_token = time.time() - t_prefill_start
        ttft_ms = t_first_token * 1000
        prefill_tok_per_sec = p_len / t_first_token if t_first_token > 0 else 0

        print(f"  TTFT (Time to First Token): {ttft_ms:.1f} ms ({prefill_tok_per_sec:.1f} prompt tok/s)")

        # Measure Decode Throughput
        current_input = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        t_gen_start = time.time()
        tokens_generated = 0
        offset = p_len

        for step in range(gen_len):
            with torch.no_grad():
                logits, kv_cache = model(current_input, position_offset=offset, past_key_values=kv_cache, use_cache=True)
            current_input = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tokens_generated += 1
            offset += 1

        t_gen_total = time.time() - t_gen_start
        decode_tok_per_sec = tokens_generated / t_gen_total if t_gen_total > 0 else 0
        print(f"  Decode Throughput:          {decode_tok_per_sec:.2f} tokens/sec")
        print(f"  Total Decode Time:          {t_gen_total:.2f} s for {tokens_generated} tokens")

        results.append({
            "prompt_length": p_len,
            "ttft_ms": ttft_ms,
            "prefill_tok_sec": prefill_tok_per_sec,
            "decode_tok_sec": decode_tok_per_sec,
            "tokens_generated": tokens_generated
        })

    # 3. KV Cache Footprint Calculation
    # 18 layers * 2 (K, V) * 4 heads * 128 dim * seq_len * 4 bytes (FP32)
    max_ctx = 2048
    kv_cache_bytes = 18 * 2 * 4 * 128 * max_ctx * 4
    kv_cache_mb = kv_cache_bytes / (1024 * 1024)
    print(f"\nKV Cache Footprint @ {max_ctx} tokens: {kv_cache_mb:.2f} MB (FP32) / {kv_cache_mb / 2:.2f} MB (FP16)")

    print("\n" + "=" * 65)
    print("EDGEBENCH SUMMARY MATRIX")
    print("=" * 65)
    print(f"{'Prompt Len':<12} | {'TTFT (ms)':<12} | {'Prefill (tok/s)':<18} | {'Decode (tok/s)':<15}")
    print("-" * 65)
    for r in results:
        print(f"{r['prompt_length']:<12} | {r['ttft_ms']:<12.1f} | {r['prefill_tok_sec']:<18.1f} | {r['decode_tok_sec']:<15.2f}")
    print("=" * 65)


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\checkpoints\maria_1b_20B_weights.pt"
    run_benchmark(ckpt)

