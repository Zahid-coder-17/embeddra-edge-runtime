# Embeddra Edge Runtime (EER) - Technical Checkpoint Inspection Report
**Model:** Maria-1B (Pretrained on 20B Tokens)  
**Architecture:** HyperGQE (Hypernetwork-based Grouped Query Experts)  
**Date:** November 2026  
**Status:** Validated  

---

## 1. Artifact Verification
* **Weights Checkpoint:** `checkpoints/maria_1b_20B_weights.pt`
  * **Size:** 2,147,165,531 bytes (2.000 GiB)
  * **SHA-256:** `8d1e0fe4acef9dcea73471f47047f907e246f230190451a06f116e5c1f0081d1`
  * **Integrity:** Verified match against `checkpoints/checksums.txt`.
* **Full Training Checkpoint:** `checkpoints/maria_1b_20B_original_full.pt`
  * **Size:** 6,441,557,581 bytes (6.000 GiB)
  * **SHA-256:** `b6aad87cb3673079564b4292c19f0ef604ac059e697ebbd95fc1278576c65198`
  * **Integrity:** Verified match against `checkpoints/checksums.txt`.

---

## 2. Parameter Attribution
* **Total Parameters in State Dict:** 1,336,184,192 (1.336 Billion)
* **Unique Physical Parameters:** 1,073,515,904 (1.074 Billion)
* **Weight Tying:** `embed.weight` and `lm_head.weight` share identical underlying storage pointer (`data_ptr()`).
* **Active Parameters per Token:** ~1.017 Billion (with ~56.6M query weights conditionally routed/pruned).
* **Storage Precision:** BFloat16 (`torch.bfloat16`).

---

## 3. Structural Specification
| Parameter | Value | Details |
| :--- | :--- | :--- |
| **Layers ($N_{\text{layers}}$)** | 18 | `blocks.0` through `blocks.17` |
| **Hidden Dimension ($d_{\text{model}}$)** | 2048 | Residual stream and normalization width |
| **Vocab Size ($V$)** | 128,256 | Llama-3 BPE tokenizer compatible |
| **Max Context ($L_{\text{max}}$)** | 8,192 | Rotary Positional Embedding (RoPE) |
| **Query Heads ($H_Q$)** | 16 | $16 \times 128 = 2048$ |
| **Key/Value Heads ($H_{KV}$)** | 4 | $4 \times 128 = 512$ (4:1 GQA compression ratio) |
| **Head Dimension ($d_{\text{head}}$)** | 128 | Standard 128-dim head size |
| **FFN Activation** | SwiGLU | $\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}}) W_{\text{down}}$ |
| **FFN Hidden Dimension** | 5632 | $\approx 2.75 \times d_{\text{model}}$ |
| **Normalization** | RMSNorm | Pre-normalization with $\epsilon = 10^{-5}$ |

---

## 4. HyperGQE Breakdown (Per Layer)
Each attention block consists of 15 parameter tensors:
1. `expert_embeddings`: `[4, 4, 64]` (4 KV groups, 4 expert vectors of dim 64)
2. `q_proj.weight`: `[2048, 2048]`
3. `k_proj.weight`: `[512, 2048]`
4. `v_proj.weight`: `[512, 2048]`
5. `q_shared.weight`: `[128, 2048]` (1 shared anchor head)
6. `o_proj.weight`: `[2048, 768]` (output projection taking 768 to 2048)
7. `router.weight`: `[16, 2048]` (produces routing logits for 16 query heads)
8. `selection_mlp.0.weight` / `bias`: `[128, 64]` / `[128]`
9. `selection_mlp.2.weight` / `bias`: `[64, 128]` / `[64]`
10. `hyper_W_D.weight` / `bias`: `[32768, 64]` / `[32768]` ($2048 \times 16$)
11. `hyper_W_U.weight` / `bias`: `[2048, 64]` / `[2048]` ($16 \times 128$)

