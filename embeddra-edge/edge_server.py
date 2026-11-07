"""
Embeddra Edge - Real-Time USB Bridge Server for Maria-1B (HyperGQE)
Bridges phone screen app to the full 1.074 Billion parameter Maria-1B model
via USB port reverse (adb reverse tcp:8080 tcp:8080).
Streams fluent English words in real-time directly to the phone screen.
"""

import os
import sys
import time
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
from tokenizers import Tokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
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

from hypergqe.model_adapter.maria_model import MariaForCausalLM

print("=" * 65)
print("  EMBEDDRA EDGE: REAL-TIME MARIA-1B USB PHONE SERVER")
print("=" * 65)

# Load Tokenizer
tok_path = os.path.join(current_dir, "tokenizer.json")
print(f"Loading official BPE vocabulary from {os.path.basename(tok_path)}...")
tokenizer = Tokenizer.from_file(tok_path)
print(f"[OK] Tokenizer loaded ({tokenizer.get_vocab_size():,} words).")

# Load Model
ckpt_path = os.path.join(project_root, "checkpoints", "maria_1b_20B_weights.pt")
print(f"Loading Maria-1B (1.074B params, HyperGQE) from {os.path.basename(ckpt_path)}...")
t0 = time.time()
model = MariaForCausalLM.from_pretrained_checkpoint(
    ckpt_path,
    device="cpu",
    dtype=torch.bfloat16
)
print(f"[OK] Maria-1B initialized in {time.time() - t0:.2f}s.")
print("=" * 65)


class MariaHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # Clean logging
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ready", "model": "Maria-1B HyperGQE"}).encode())
            return

        if parsed.path == "/generate":
            prompt = params.get("prompt", [""])[0]
            if not prompt:
                self.send_response(400)
                self.end_headers()
                return

            print(f"\n[Phone Query Received]: \"{prompt}\"")
            print("Streaming fluent English response to phone screen > ", end="", flush=True)

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Format prompt for base causal model completion
            formatted_prompt = f"Question: {prompt}\nAnswer:" if not (prompt.endswith(":") or prompt.endswith("as") or prompt.endswith("is")) else prompt
            encoded = tokenizer.encode(formatted_prompt)
            prompt_ids = torch.tensor([encoded.ids], dtype=torch.long)

            t_start = time.time()
            first_token_time = None
            generated_ids = []
            prev_text = ""
            tokens_count = 0

            for token_id in model.generate(prompt_ids, max_new_tokens=80, temperature=0.3):
                if first_token_time is None:
                    first_token_time = time.time()

                generated_ids.append(token_id)
                tokens_count += 1

                # Incremental subword decoding
                current_text = tokenizer.decode(generated_ids)
                new_chunk = current_text[len(prev_text):]
                if new_chunk:
                    print(new_chunk, end="", flush=True)
                    prev_text = current_text
                    # Write HTTP chunk
                    chunk_bytes = new_chunk.encode("utf-8")
                    self.wfile.write(f"{len(chunk_bytes):X}\r\n".encode())
                    self.wfile.write(chunk_bytes)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()

            # End of chunked stream
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

            total_time = time.time() - t_start
            ttft = (first_token_time - t_start) * 1000 if first_token_time else 0
            tps = tokens_count / total_time if total_time > 0 else 0
            print(f"\n[Complete] TTFT: {ttft:.1f}ms | Speed: {tps:.2f} tok/s | Tokens: {tokens_count}\n")


def main():
    port = 8080
    server = HTTPServer(("0.0.0.0", port), MariaHandler)
    print(f"\nSERVER READY on http://localhost:{port}")
    print("Phone can now send prompts and stream fluent English directly!\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    main()

