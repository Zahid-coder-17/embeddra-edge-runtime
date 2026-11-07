"""
Embeddra Edge Runtime (EER) - Binary Model Serializer (Streamed)
Converts Maria-1B PyTorch checkpoints into the zero-copy, memory-mapped
Embeddra Edge Binary Format (.eer) with minimal memory footprint.
"""

import os
import sys
import struct
import time
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Handle unpickling MariaConfig
import types
if "model" not in sys.modules:
    m = types.ModuleType("model")
    class MariaConfig: pass
    m.MariaConfig = MariaConfig
    sys.modules["model"] = m

MAGIC = b"EER1"
VERSION = 1

DTYPE_FP32 = 0
DTYPE_FP16 = 1
DTYPE_INT8 = 2
DTYPE_BF16 = 3


def serialize_checkpoint_to_eer(
    checkpoint_path: str,
    output_eer_path: str,
    target_dtype: str = "fp16"
):
    print("=" * 65)
    print("EMBEDDRA EDGE RUNTIME: STREAMED BINARY SERIALIZER (.eer)")
    print("=" * 65)
    print(f"Source Checkpoint: {checkpoint_path}")
    print(f"Target Output:     {output_eer_path}")
    print(f"Target Precision:  {target_dtype.upper()}")

    t0 = time.time()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    sd = ckpt.get("model_state_dict", ckpt)
    print(f"[OK] Checkpoint mapped in {time.time() - t0:.2f}s ({len(sd)} tensors).")

    # Geometry
    vocab_size = 128256
    hidden_size = 2048
    intermediate_size = 5632
    num_layers = 18
    num_heads = 16
    num_kv_heads = 4
    head_dim = 128
    max_seq_len = 2048
    dtype_code = DTYPE_FP16

    header_pack = struct.pack(
        "<4sIIIIIIIIIII",
        MAGIC,
        VERSION,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        max_seq_len,
        dtype_code,
        len(sd)
    )

    # 1. Pre-calculate tensor metadata & offsets
    current_offset = 0
    tensor_entries = []
    for name, tensor in sd.items():
        size_bytes = tensor.numel() * 2 # 2 bytes per float16
        align_pad = (64 - (current_offset % 64)) % 64
        current_offset += align_pad
        tensor_entries.append({
            "name": name,
            "shape": list(tensor.shape),
            "dtype": dtype_code,
            "offset": current_offset,
            "size_bytes": size_bytes,
            "align_pad": align_pad
        })
        current_offset += size_bytes

    # 2. Build index table
    index_buffer = bytearray()
    for entry in tensor_entries:
        name_bytes = entry["name"].encode("utf-8")
        entry_header = struct.pack("<HB", len(name_bytes), len(entry["shape"]))
        index_buffer.extend(entry_header)
        index_buffer.extend(name_bytes)
        for d in entry["shape"]:
            index_buffer.extend(struct.pack("<I", d))
        index_buffer.extend(struct.pack("<BQQ", entry["dtype"], entry["offset"], entry["size_bytes"]))

    header_and_index = bytearray(header_pack + struct.pack("<I", len(index_buffer)) + index_buffer)
    page_pad = (4096 - (len(header_and_index) % 4096)) % 4096
    header_and_index.extend(b"\x00" * page_pad)
    payload_start = len(header_and_index)

    # Re-encode index with absolute offsets
    adjusted_index = bytearray()
    for entry in tensor_entries:
        name_bytes = entry["name"].encode("utf-8")
        adjusted_index.extend(struct.pack("<HB", len(name_bytes), len(entry["shape"])))
        adjusted_index.extend(name_bytes)
        for d in entry["shape"]:
            adjusted_index.extend(struct.pack("<I", d))
        abs_off = payload_start + entry["offset"]
        adjusted_index.extend(struct.pack("<BQQ", entry["dtype"], abs_off, entry["size_bytes"]))

    final_header = bytearray(header_pack + struct.pack("<I", len(adjusted_index)) + adjusted_index)
    page_pad = (4096 - (len(final_header) % 4096)) % 4096
    final_header.extend(b"\x00" * page_pad)

    print(f"\nWriting binary stream to {output_eer_path}...")
    with open(output_eer_path, "wb") as f:
        f.write(final_header)
        for entry in tensor_entries:
            if entry["align_pad"] > 0:
                f.write(b"\x00" * entry["align_pad"])
            t = sd[entry["name"]].to(torch.float16).contiguous()
            # Convert to numpy bytes
            f.write(t.numpy().tobytes())

    total_size = os.path.getsize(output_eer_path)
    print(f"[OK] Serialization complete: {total_size / (1024**3):.2f} GB ({total_size:,} bytes).")
    return output_eer_path

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\checkpoints\maria_1b_20B_weights.pt"
    dst = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\checkpoints\maria_1b.eer"
    serialize_checkpoint_to_eer(src, dst)
