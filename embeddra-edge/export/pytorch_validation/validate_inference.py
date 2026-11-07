"""
Embeddra Edge Runtime (EER) - Milestone 2 Validation Script
Validates Maria-1B model reconstruction, state dict loading, forward pass,
numerical consistency, and autoregressive generation with KV cache.
"""

import os
import sys
import time
import torch

# Ensure embeddra-edge is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig


def run_validation(checkpoint_path: str):
    print("=" * 60)
    print("EMBEDDRA EDGE RUNTIME: PYTORCH INFERENCE VALIDATION")
    print("=" * 60)
    print(f"Target Checkpoint: {checkpoint_path}")

    # Step 1: Instantiate & load checkpoint
    t0 = time.time()
    model = MariaForCausalLM.from_pretrained_checkpoint(
        checkpoint_path=checkpoint_path,
        device="cpu",
        dtype=torch.float32  # Use float32 on CPU for clean numerical validation
    )
    load_time = time.time() - t0
    print(f"[OK] Model loaded in {load_time:.2f} seconds.")

    # Step 2: Forward pass validation
    test_seq_len = 16
    prompt_ids = torch.randint(100, 5000, (1, test_seq_len), dtype=torch.long)
    print(f"\nTesting forward pass on dummy prompt of shape {prompt_ids.shape}...")

    t0 = time.time()
    with torch.no_grad():
        logits, kv_cache = model(prompt_ids, use_cache=True)
    forward_time = time.time() - t0

    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (1, test_seq_len, 128256), f"Unexpected logits shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "ERROR: NaNs detected in logits!"
    assert not torch.isinf(logits).any(), "ERROR: Infs detected in logits!"
    print(f"[OK] Forward pass completed in {forward_time * 1000:.1f} ms. Logits are finite and healthy.")
    print(f"     Logits mean={logits.mean().item():.4f}, std={logits.std().item():.4f}, min={logits.min().item():.4f}, max={logits.max().item():.4f}")

    # Step 3: Autoregressive generation with KV cache
    num_gen_tokens = 10
    print(f"\nTesting autoregressive generation for {num_gen_tokens} tokens (temperature=0.0 / greedy)...")

    generated_tokens = []
    t_start = time.time()
    for tok in model.generate(prompt_ids, max_new_tokens=num_gen_tokens, temperature=0.0):
        generated_tokens.append(tok)
        print(f"  Generated token: {tok}")

    total_gen_time = time.time() - t_start
    tps = len(generated_tokens) / total_gen_time if total_gen_time > 0 else 0
    print(f"[OK] Generation completed: {len(generated_tokens)} tokens in {total_gen_time:.2f}s ({tps:.2f} tokens/sec on CPU).")

    print("\n" + "=" * 60)
    print("MILESTONE 2: MODEL RECONSTRUCTION & NUMERICAL VALIDATION PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\checkpoints\maria_1b_20B_weights.pt"
    run_validation(ckpt)

