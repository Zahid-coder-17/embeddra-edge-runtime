"""
Embeddra Edge Runtime (EER) - Checkpoint Inspection Utility
Analyzes Maria-1B PyTorch checkpoints, layer geometry, parameter attribution,
quantization footprint, and operator requirements.
"""

import os
import sys
import hashlib
import json
import torch

# Ensure model module can be resolved for unpickling
import types
if "model" not in sys.modules:
    m = types.ModuleType("model")
    class MariaConfig: pass
    m.MariaConfig = MariaConfig
    sys.modules["model"] = m


def compute_sha256(filepath: str, block_size=65536) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_checkpoint(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    file_size_bytes = os.path.getsize(checkpoint_path)
    file_size_gb = file_size_bytes / (1024 ** 3)
    print(f"Inspecting: {checkpoint_path}")
    print(f"File Size: {file_size_gb:.3f} GiB ({file_size_bytes:,} bytes)")

    print("Computing SHA-256 checksum...")
    sha256 = compute_sha256(checkpoint_path)
    print(f"SHA-256: {sha256}")

    print("Loading PyTorch checkpoint dictionary...")
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    keys = list(data.keys())
    print(f"Checkpoint Keys: {keys}")

    sd = data.get("model_state_dict", data.get("state_dict", data))
    print(f"Total Tensors in State Dict: {len(sd)}")

    # Parameter accounting
    total_elements = 0
    unique_storage_ptrs = set()
    unique_elements = 0
    tensor_inventory = []

    for name, tensor in sd.items():
        numel = tensor.numel()
        shape = list(tensor.shape)
        dtype = str(tensor.dtype)
        total_elements += numel

        ptr = tensor.data_ptr()
        if ptr not in unique_storage_ptrs:
            unique_storage_ptrs.add(ptr)
            unique_elements += numel

        tensor_inventory.append({
            "name": name,
            "shape": shape,
            "dtype": dtype,
            "numel": numel,
            "shared": (ptr in unique_storage_ptrs and unique_elements < total_elements)
        })

    is_tied = False
    if "embed.weight" in sd and "lm_head.weight" in sd:
        is_tied = sd["embed.weight"].data_ptr() == sd["lm_head.weight"].data_ptr()

    # Block analysis
    block_names = [k for k in sd.keys() if k.startswith("blocks.")]
    block_indices = sorted(list(set(int(k.split(".")[1]) for k in block_names)))

    report = {
        "checkpoint_path": checkpoint_path,
        "file_size_bytes": file_size_bytes,
        "file_size_gb": file_size_gb,
        "sha256": sha256,
        "step": data.get("step", None),
        "tokens_seen": data.get("tokens_seen", None),
        "loss": data.get("loss", None),
        "total_parameters": total_elements,
        "unique_parameters": unique_elements,
        "tied_embedding_head": is_tied,
        "num_layers": len(block_indices),
        "tensor_count": len(sd),
    }

    print("\n--- Checkpoint Summary ---")
    print(f"Total Parameters: {total_elements:,} ({total_elements / 1e9:.3f}B)")
    print(f"Unique Parameters: {unique_elements:,} ({unique_elements / 1e9:.3f}B)")
    print(f"Embedding & LM Head Tied: {is_tied}")
    print(f"Transformer Layers: {len(block_indices)}")
    if report["step"] is not None:
        print(f"Pretraining Step: {report['step']}")
    if report["tokens_seen"] is not None:
        print(f"Tokens Seen: {report['tokens_seen']:,}")

    return report, tensor_inventory


if __name__ == "__main__":
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\checkpoints\maria_1b_20B_weights.pt"
    inspect_checkpoint(ckpt_path)

