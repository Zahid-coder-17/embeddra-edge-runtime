"""
Embeddra Edge Runtime (EER) - Maria-1B Full Architecture & Model Adapter
Reconstructs the exact HyperGQE architecture matching the original PyTorch checkpoint.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Generator
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from hypergqe.attention.hypergqe_attention import HyperGQEAttention
except (ImportError, ModuleNotFoundError):
    from embeddra_edge.hypergqe.attention.hypergqe_attention import HyperGQEAttention


@dataclass
class MariaConfig:
    vocab_size: int = 128256
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 18
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    max_position_embeddings: int = 8192
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True


class MariaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_f = x.float()
        variance = x_f.pow(2).mean(-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(variance + self.eps)
        return (self.weight * x_norm).to(input_dtype)


class MariaMLP(nn.Module):
    """SwiGLU feed-forward network matching checkpoint parameter names."""
    def __init__(self, hidden_size: int = 2048, intermediate_size: int = 5632):
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MariaBlock(nn.Module):
    """Transformer block with HyperGQE attention and SwiGLU FFN."""
    def __init__(self, config: MariaConfig):
        super().__init__()
        self.attn_norm = MariaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = HyperGQEAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )
        self.ffn_norm = MariaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = MariaMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_offset: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm Attention with residual
        norm_x = self.attn_norm(hidden_states)
        attn_out, new_cache = self.attn(
            norm_x,
            position_offset=position_offset,
            kv_cache=kv_cache,
            use_cache=use_cache
        )
        hidden_states = hidden_states + attn_out

        # Pre-norm FFN with residual
        norm_ffn = self.ffn_norm(hidden_states)
        ffn_out = self.ffn(norm_ffn)
        hidden_states = hidden_states + ffn_out

        return hidden_states, new_cache


class MariaForCausalLM(nn.Module):
    """Complete Maria-1B Causal Language Model."""
    def __init__(self, config: Optional[MariaConfig] = None):
        super().__init__()
        self.config = config or MariaConfig()

        self.embed = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.blocks = nn.ModuleList([MariaBlock(self.config) for _ in range(self.config.num_hidden_layers)])
        self.norm = MariaRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_offset: int = 0,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        B, S = input_ids.shape
        hidden_states = self.embed(input_ids)

        new_key_values = [] if use_cache else None

        for idx, block in enumerate(self.blocks):
            layer_cache = past_key_values[idx] if past_key_values is not None else None
            hidden_states, new_cache = block(
                hidden_states,
                position_offset=position_offset,
                kv_cache=layer_cache,
                use_cache=use_cache
            )
            if use_cache:
                new_key_values.append(new_cache)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, new_key_values

    @classmethod
    def from_pretrained_checkpoint(cls, checkpoint_path: str, device: str = "cpu", dtype: torch.dtype = torch.bfloat16):
        """Loads and verifies original PyTorch checkpoint without altering architecture."""
        config = MariaConfig()
        model = cls(config).to(device=device, dtype=dtype)

        # Allow unpickling model.MariaConfig
        import sys, types, gc
        if "model" not in sys.modules:
            m = types.ModuleType("model")
            m.MariaConfig = MariaConfig
            sys.modules["model"] = m

        print(f"Loading checkpoint weights from {checkpoint_path} (mmap=True)...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)

        # Support dequantization for Embeddra INT8/INT4 checkpoints
        is_quantized = ckpt.get("format") == "embeddra_quantized_v1"
        clean_sd = {}
        for k in list(sd.keys()):
            if k.endswith("_scale"):
                continue
            scale_key = f"{k}_scale"
            if is_quantized and scale_key in sd:
                scale = sd[scale_key].unsqueeze(1) if sd[k].dim() > 1 else sd[scale_key]
                clean_sd[k] = (sd[k].float() * scale.float()).to(dtype=dtype)
            else:
                clean_sd[k] = sd[k].to(dtype=dtype)

        missing, unexpected = model.load_state_dict(clean_sd, strict=True)
        print(f"State dict loaded successfully! (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
        del ckpt, sd, clean_sd
        gc.collect()

        model.eval()
        return model

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.25,
        eos_token_id: int = 128001,
    ) -> Generator[int, None, None]:
        """Streaming autoregressive token generation with full KV cache reuse."""
        past_key_values = None
        current_input = input_ids
        position_offset = 0
        generated_tokens = list(input_ids[0].tolist())

        for step in range(max_new_tokens):
            logits, past_key_values = self.forward(
                current_input,
                position_offset=position_offset,
                past_key_values=past_key_values,
                use_cache=True
            )

            next_token_logits = logits[:, -1, :].clone()

            # Apply repetition penalty to prevent phrase loops
            if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                for prev_tok in set(generated_tokens):
                    val = next_token_logits[0, prev_tok].item()
                    if val > 0:
                        next_token_logits[0, prev_tok] = val / repetition_penalty
                    else:
                        next_token_logits[0, prev_tok] = val * repetition_penalty

            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            token_id = next_token.item()
            generated_tokens.append(token_id)
            yield token_id

            if token_id == eos_token_id:
                break

            position_offset += current_input.shape[1]
            current_input = next_token
