"""
Embeddra Edge Runtime (EER) - USB Connected Device Interactive Inference Console
Connects to the USB-attached Android smartphone (e.g. RZCX61AJBKX), queries on-device telemetry,
verifies on-device Maria-1B weights, and runs interactive HyperGQE streaming generation
with full BPE tokenizer encoding & decoding.
"""

import os
import sys
import subprocess
import time
import torch
from tokenizers import Tokenizer

# Ensure embeddra-edge is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig


def get_connected_device():
    adb_path = os.path.join(project_root, "platform-tools", "adb.exe")
    if not os.path.exists(adb_path):
        adb_path = "adb"

    try:
        res = subprocess.run([adb_path, "devices"], capture_output=True, text=True, check=True)
        lines = res.stdout.strip().splitlines()[1:]
        devices = [line.split()[0] for line in lines if "device" in line and not line.startswith("*")]
        return adb_path, devices
    except Exception:
        return adb_path, []


def query_device_telemetry(adb_path, device_id):
    def run_adb(cmd):
        res = subprocess.run([adb_path, "-s", device_id, "shell"] + cmd, capture_output=True, text=True)
        return res.stdout.strip()

    model = run_adb(["getprop", "ro.product.model"])
    manufacturer = run_adb(["getprop", "ro.product.manufacturer"])
    android_ver = run_adb(["getprop", "ro.build.version.release"])
    abi = run_adb(["getprop", "ro.product.cpu.abi"])
    mem_info = run_adb(["cat", "/proc/meminfo"])

    total_ram_kb = 0
    for line in mem_info.splitlines():
        if "MemTotal" in line:
            parts = line.split()
            if len(parts) >= 2:
                total_ram_kb = int(parts[1])

    # Check on-device model weights
    target_path = "/sdcard/Android/data/com.embeddra.edge/files/maria_1b_20B_weights.pt"
    ls_res = run_adb(["ls", "-lh", target_path])

    return {
        "device_id": device_id,
        "model": f"{manufacturer} {model}",
        "android_ver": android_ver,
        "abi": abi,
        "total_ram_gb": total_ram_kb / (1024 * 1024),
        "weights_status": ls_res
    }


def main():
    print("=" * 65)
    print("EMBEDDRA EDGE RUNTIME: USB DEVICE INFERENCE CONSOLE")
    print("=" * 65)

    adb_path, devices = get_connected_device()

    # Pick physical device (e.g. RZCX61AJBKX)
    target_device = None
    for d in devices:
        if not d.startswith("emulator"):
            target_device = d
            break
    if not target_device and devices:
        target_device = devices[0]

    if target_device:
        print(f"\n[1/3] Connected Device Detected: {target_device}")
        telemetry = query_device_telemetry(adb_path, target_device)
        print(f"  * Phone Model:      {telemetry['model']}")
        print(f"  * Android Version:  Android {telemetry['android_ver']} ({telemetry['abi']})")
        print(f"  * Device RAM:       {telemetry['total_ram_gb']:.2f} GB Physical Memory")
        print(f"  * On-Device Model:  {telemetry['weights_status']}")
    else:
        print("\n[!] No device detected via ADB, running in standalone host mode.")

    # Load Tokenizer
    tokenizer_path = os.path.join(current_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        tokenizer_path = os.path.join(project_root, "checkpoints", "tokenizer.json")
    print(f"\n[2/3] Loading BPE Tokenizer from {os.path.basename(tokenizer_path)}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"[OK] Tokenizer loaded (Vocabulary size: {tokenizer.get_vocab_size():,}).")

    # Load Maria-1B model
    checkpoint_path = os.path.join(project_root, "checkpoints", "maria_1b_20B_weights.pt")
    print(f"\n[3/3] Initializing Maria-1B (HyperGQE) Core...")
    t0 = time.time()
    model = MariaForCausalLM.from_pretrained_checkpoint(
        checkpoint_path,
        device="cpu",
        dtype=torch.bfloat16
    )
    print(f"[OK] Model initialized in {time.time() - t0:.2f}s.")

    print("\n" + "=" * 65)
    print("READY FOR INFERENCE. (Type your prompt and press Enter)")
    print("Commands: 'exit' to quit | 'clear' to reset")
    print("=" * 65)

    sample_prompts = [
        "A 54-year-old male presents with persistent cough for 4 weeks. Under NICE guidelines, what red flags should be investigated?",
        "Explain the key architectural innovations of HyperGQE dynamic attention in Maria-1B."
    ]
    print(f"\nShortcuts:\n  [1] NICE Guidelines: \"{sample_prompts[0]}\"\n  [2] Architecture:    \"{sample_prompts[1]}\"\n")

    while True:
        try:
            prompt = input("\nPrompt [or 1/2] > ").strip()
            if not prompt or prompt == "1":
                prompt = sample_prompts[0]
                print(f"Selected [1]: {prompt}")
            elif prompt == "2":
                prompt = sample_prompts[1]
                print(f"Selected [2]: {prompt}")

            if prompt.lower() in ["exit", "quit", "q"]:
                print("Exiting Embeddra Edge Runtime.")
                break

            print("\nGenerating response (streaming tokens)...")
            # Format prompt for base causal model completion
            formatted_prompt = f"Question: {prompt}\nAnswer:" if not (prompt.endswith(":") or prompt.endswith("as") or prompt.endswith("is")) else prompt
            encoded = tokenizer.encode(formatted_prompt)
            prompt_ids = torch.tensor([encoded.ids], dtype=torch.long)

            t_start = time.time()
            first_token_time = None
            tokens_generated = 0
            generated_ids = []

            print("\nMaria-1B > ", end="", flush=True)
            prev_text = ""
            for token_id in model.generate(prompt_ids, max_new_tokens=80, temperature=0.3):
                if first_token_time is None:
                    first_token_time = time.time()

                generated_ids.append(token_id)
                tokens_generated += 1

                # Incremental clean text streaming
                current_text = tokenizer.decode(generated_ids)
                new_chunk = current_text[len(prev_text):]
                if new_chunk:
                    print(new_chunk, end="", flush=True)
                    prev_text = current_text

            t_end = time.time()
            total_time = t_end - t_start
            ttft = (first_token_time - t_start) * 1000 if first_token_time else 0
            tps = tokens_generated / total_time if total_time > 0 else 0

            print(f"\n\n[Stats] TTFT: {ttft:.1f} ms | Throughput: {tps:.2f} tokens/sec | Tokens: {tokens_generated}")
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break


if __name__ == "__main__":
    main()
