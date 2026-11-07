"""
Embeddra Edge Runtime (EER) - HyperGQE Attention Implementation
Implements Hypernetwork-based Grouped Query Experts (HyperGQE).
Preserves exact operator behavior, RoPE, dynamic routing, and KV caching semantics.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoPE(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(offset, offset + seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        # Reshape for broadcasting [1, seq_len, 1, dim]
        return cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2)


class HyperGQEAttention(nn.Module):
    """
    Hypernetwork-based Grouped Query Experts (HyperGQE) Module.
    Combines grouped-query attention, token-conditional expert routing,
    and hypernetwork-generated low-rank query adaptation.
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        max_position_embeddings: int = 8192,
        rope_theta: float = 500000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.experts_per_group = num_attention_heads // num_key_value_heads  # 16 / 4 = 4
        self.expert_dim = 64
        self.hyper_rank = 16

        # Standard linear projections
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)  # [2048, 2048]
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)  # [512, 2048]
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)  # [512, 2048]
        self.q_shared = nn.Linear(hidden_size, head_dim, bias=False)                     # [128, 2048]

        # Compressed output projection (4 group heads + 2 shared/hyper heads = 6 heads @ 128 = 768)
        self.o_proj = nn.Linear(768, hidden_size, bias=False)                            # [2048, 768]

        # Dynamic Router
        self.router = nn.Linear(hidden_size, num_attention_heads, bias=False)            # [16, 2048]

        # Expert Embeddings: [num_groups=4, experts_per_group=4, expert_dim=64]
        self.expert_embeddings = nn.Parameter(torch.empty(self.num_key_value_heads, self.experts_per_group, self.expert_dim))

        # Selection MLP: [64 -> 128 -> 64]
        self.selection_mlp = nn.Sequential(
            nn.Linear(self.expert_dim, 128),
            nn.SiLU(),
            nn.Linear(128, self.expert_dim)
        )

        # Hypernetworks: generate low-rank down- and up-projections
        self.hyper_W_D = nn.Linear(self.expert_dim, hidden_size * self.hyper_rank)       # [32768, 64]
        self.hyper_W_U = nn.Linear(self.expert_dim, self.hyper_rank * head_dim)          # [2048, 64]

        # Rotary Positional Embeddings
        self.rotary_emb = RoPE(dim=head_dim, max_position_embeddings=max_position_embeddings, base=rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_offset: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, S, C = hidden_states.shape

        # 1. Base Q, K, V Projections
        q_base = self.q_proj(hidden_states).view(B, S, self.num_key_value_heads, self.experts_per_group, self.head_dim)
        k = self.k_proj(hidden_states).view(B, S, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, S, self.num_key_value_heads, self.head_dim)
        q_shared = self.q_shared(hidden_states).view(B, S, 1, self.head_dim)

        # 2. Dynamic Router & Expert Blending
        router_logits = self.router(hidden_states).view(B, S, self.num_key_value_heads, self.experts_per_group)
        routing_weights = F.softmax(router_logits, dim=-1)  # [B, S, 4, 4]

        # Blend query experts per group
        # q_base: [B, S, 4, 4, 128] * routing_weights: [B, S, 4, 4, 1] -> sum over experts -> [B, S, 4, 128]
        q_group = torch.sum(q_base * routing_weights.unsqueeze(-1), dim=3)  # [B, S, 4, 128]

        # 3. Hypernetwork Dynamic Query Generation
        group_emb = torch.einsum("bsge,ged->bsgd", routing_weights, self.expert_embeddings)  # [B, S, 4, 64]
        context_vec = group_emb.mean(dim=2)  # [B, S, 64]
        sel = self.selection_mlp(context_vec)  # [B, S, 64]

        W_D = self.hyper_W_D(sel).view(B, S, C, self.hyper_rank)       # [B, S, 2048, 16]
        W_U = self.hyper_W_U(sel).view(B, S, self.hyper_rank, self.head_dim)  # [B, S, 16, 128]

        # Dynamic low-rank projection: x @ W_D @ W_U
        # [B, S, 1, 2048] @ [B, S, 2048, 16] -> [B, S, 1, 16]
        h_mid = torch.matmul(hidden_states.unsqueeze(2), W_D)
        # [B, S, 1, 16] @ [B, S, 16, 128] -> [B, S, 1, 128]
        q_dyn = torch.matmul(h_mid, W_U).view(B, S, 1, self.head_dim)

        # Concatenate dynamic head with shared head -> [B, S, 2, 128]
        q_aux = torch.cat([q_shared, q_dyn], dim=2)

        # 4. RoPE
        cos, sin = self.rotary_emb(hidden_states, seq_len=S, offset=position_offset)
        # q_group: [B, S, 4, 128]
        q_group_rot, k_rot = apply_rotary_pos_emb(q_group, k, cos, sin)
        q_aux_rot, _ = apply_rotary_pos_emb(q_aux, q_aux, cos, sin)

        # 5. KV Cache Management
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k_rot = torch.cat([prev_k, k_rot], dim=1)
            v = torch.cat([prev_v, v], dim=1)

        new_cache = (k_rot, v) if use_cache else None
        total_k_len = k_rot.shape[1]

        # 6. Scaled Dot-Product Attention
        scale = 1.0 / math.sqrt(self.head_dim)

        # Group Attention: [B, 4, S, 128] x [B, 4, total_k_len, 128]^T
        q_g = q_group_rot.transpose(1, 2)  # [B, 4, S, 128]
        k_g = k_rot.transpose(1, 2)        # [B, 4, total_k_len, 128]
        v_g = v.transpose(1, 2)            # [B, 4, total_k_len, 128]

        attn_scores_g = torch.matmul(q_g, k_g.transpose(-1, -2)) * scale  # [B, 4, S, total_k_len]
        if S > 1:
            causal_mask = torch.triu(torch.full((S, total_k_len), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype), diagonal=position_offset + 1)
            attn_scores_g = attn_scores_g + causal_mask.unsqueeze(0).unsqueeze(0)
        attn_weights_g = F.softmax(attn_scores_g, dim=-1).to(v_g.dtype)
        out_g = torch.matmul(attn_weights_g, v_g).transpose(1, 2).reshape(B, S, 512)  # [B, S, 512]

        # Aux (Shared + Hyper) Attention: [B, 2, S, 128] attends to mean-pooled / shared KV
        q_a = q_aux_rot.transpose(1, 2)  # [B, 2, S, 128]
        k_a = k_g.mean(dim=1, keepdim=True)  # [B, 1, total_k_len, 128]
        v_a = v_g.mean(dim=1, keepdim=True)  # [B, 1, total_k_len, 128]

        attn_scores_a = torch.matmul(q_a, k_a.transpose(-1, -2)) * scale  # [B, 2, S, total_k_len]
        if S > 1:
            attn_scores_a = attn_scores_a + causal_mask.unsqueeze(0).unsqueeze(0)
        attn_weights_a = F.softmax(attn_scores_a, dim=-1).to(v_a.dtype)
        out_a = torch.matmul(attn_weights_a, v_a.expand(-1, 2, -1, -1)).transpose(1, 2).reshape(B, S, 256)  # [B, S, 256]

        # 7. Concatenate features to 768 dimensions
        out_concat = torch.cat([out_g, out_a], dim=-1)  # [B, S, 768]

        # 8. Output Projection
        output = self.o_proj(out_concat)  # [B, S, 2048]

        return output, new_cache

