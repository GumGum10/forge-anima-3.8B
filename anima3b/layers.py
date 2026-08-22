from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


# ============================================================================
# Model Architecture Components
# ============================================================================

class RMSNorm(nn.Module):
    """Direct-scale RMSNorm used by Qwen3.5's gated delta block."""
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        y = y * self.weight.float()
        return y.to(dtype=x.dtype)


class Qwen35RMSNorm(nn.Module):
    """Qwen3.5 RMSNorm whose checkpoint weights are deltas around one."""
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        y = y * (1.0 + self.weight.float())
        return y.to(dtype=x.dtype)


class SSMBlock(nn.Module):
    """Reference-equivalent Qwen3.5 GatedDeltaNet linear-attention block."""
    def __init__(self, hidden_size=2560, d_inner=8192, n_groups=32,
                 d_gate=4096, conv_kernel=4, norm_dim=128,
                 device=None, dtype=None, ops=None):
        super().__init__()
        ops = ops or nn
        self.num_v_heads = n_groups
        self.num_k_heads = 16
        self.head_k_dim = norm_dim
        self.head_v_dim = norm_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = ops.Linear(hidden_size, self.conv_dim, bias=False, device=device, dtype=dtype)
        self.in_proj_z = ops.Linear(hidden_size, self.value_dim, bias=False, device=device, dtype=dtype)
        self.in_proj_a = ops.Linear(hidden_size, self.num_v_heads, bias=False, device=device, dtype=dtype)
        self.in_proj_b = ops.Linear(hidden_size, self.num_v_heads, bias=False, device=device, dtype=dtype)
        self.conv1d = ops.Conv1d(
            self.conv_dim, self.conv_dim, conv_kernel, groups=self.conv_dim,
            padding=conv_kernel - 1, bias=False, device=device, dtype=dtype
        )
        self.out_proj = ops.Linear(self.value_dim, hidden_size, bias=False, device=device, dtype=dtype)
        self.norm = RMSNorm(self.head_v_dim, device=device, dtype=dtype)
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, device=device, dtype=dtype))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads, device=device, dtype=dtype))

    def _delta_rule_scan(self, query, key, value, g, beta):
        initial_dtype = query.dtype
        query = l2norm(query, dim=-1).transpose(1, 2).contiguous().float()
        key = l2norm(key, dim=-1).transpose(1, 2).contiguous().float()
        value = value.transpose(1, 2).contiguous().float()
        beta = beta.transpose(1, 2).contiguous().float()
        g = g.transpose(1, 2).contiguous().float()

        batch, heads, seq_len, k_dim = key.shape
        v_dim = value.shape[-1]
        query = query * (1.0 / (k_dim ** 0.5))
        state = torch.zeros(batch, heads, k_dim, v_dim, device=query.device, dtype=torch.float32)
        outputs = []

        for i in range(seq_len):
            q_t = query[:, :, i]
            k_t = key[:, :, i]
            v_t = value[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)

            state = state * g_t
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            outputs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))

        return torch.stack(outputs, dim=2).transpose(1, 2).contiguous().to(initial_dtype)

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            hidden_states = hidden_states * attention_mask[:, :, None].to(dtype=hidden_states.dtype)

        batch, seq_len, _ = hidden_states.shape
        z = self.in_proj_z(hidden_states).reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len]).transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)

        beta = self.in_proj_b(hidden_states).sigmoid()
        g = -self.A_log.float().exp().to(hidden_states.device) * F.softplus(
            self.in_proj_a(hidden_states).float() + self.dt_bias.to(hidden_states.device).float()
        )
        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        y = self._delta_rule_scan(query, key, value, g, beta)
        y = self.norm(y.reshape(-1, self.head_v_dim))
        y = y * F.silu(z.reshape(-1, self.head_v_dim).float()).to(y.dtype)
        return self.out_proj(y.reshape(batch, seq_len, self.value_dim))


class GatedSelfAttention(nn.Module):
    """
    Self-attention with gated Q projection.

    q_proj outputs Q(4096) + gate(4096) = 8192:
    - 16 attention heads with 256 head_dim
    - 4 KV heads with 256 head_dim (GQA ratio 4)
    - Q and gate are interleaved by head in the checkpoint
    - After attention: [B, L, 4096] gated by sigmoid(gate) -> o_proj
    """
    def __init__(self, hidden_size=2560, num_heads=16, num_kv_heads=4,
                 head_dim=256, rope_theta=10000000.0,
                 device=None, dtype=None, ops=None):
        super().__init__()
        ops = ops or nn
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gqa_ratio = num_heads // num_kv_heads
        self.inner_dim = num_heads * head_dim  # 4096

        self.q_proj = ops.Linear(hidden_size, 2 * self.inner_dim, bias=False, device=device, dtype=dtype)
        self.k_proj = ops.Linear(hidden_size, num_kv_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = ops.Linear(hidden_size, num_kv_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = ops.Linear(self.inner_dim, hidden_size, bias=False, device=device, dtype=dtype)

        self.q_norm = Qwen35RMSNorm(head_dim, device=device, dtype=dtype)
        self.k_norm = Qwen35RMSNorm(head_dim, device=device, dtype=dtype)

    def forward(self, hidden_states, attention_mask=None, freqs_cis=None):
        B, L, _ = hidden_states.shape

        # Q projection with gate
        qg = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim * 2)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(B, L, self.inner_dim)

        # Reshape to heads
        q = q.view(B, L, self.num_heads, self.head_dim)  # [B, L, 16, 256]
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)  # [B, L, 4, 256]
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)  # [B, L, 4, 256]

        # Per-head norms
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose for attention: [B, H, L, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        if freqs_cis is not None:
            cos, sin = freqs_cis
            q = _apply_rotary_emb(q, cos, sin)
            k = _apply_rotary_emb(k, cos, sin)

        # GQA: expand K, V
        k = k.repeat_interleave(self.gqa_ratio, dim=1)  # [B, 16, L, 256]
        v = v.repeat_interleave(self.gqa_ratio, dim=1)  # [B, 16, L, 256]

        # Attention (ensure mask dtype matches query)
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.to(dtype=q.dtype)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=(attention_mask is None)
        )  # [B, 16, L, 256]

        # Reshape and gate
        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.inner_dim)  # [B, L, 4096]
        attn_out = attn_out * torch.sigmoid(gate)

        return self.o_proj(attn_out)


def _apply_rotary_emb(x, cos, sin):
    """Apply rotary position embeddings."""
    rotary_dim = cos.shape[-1]
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    x1 = x_rot[..., : rotary_dim // 2]
    x2 = x_rot[..., rotary_dim // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    x_embed = (x_rot * cos) + (rotated * sin)
    return torch.cat((x_embed, x_pass), dim=-1)


def _precompute_freqs_cis(head_dim, max_seq_len, theta=10000000.0, device=None, dtype=None):
    """Precompute RoPE frequencies."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, L, D/2]
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)  # [1, 1, L, D/2]
    # Duplicate for full head_dim
    cos = cos.repeat(1, 1, 1, 2)  # [1, 1, L, D]
    sin = sin.repeat(1, 1, 1, 2)  # [1, 1, L, D]
    if dtype is not None:
        cos = cos.to(dtype)
        sin = sin.to(dtype)
    return cos, sin


class MLP(nn.Module):
    """SwiGLU MLP."""
    def __init__(self, hidden_size=2560, intermediate_size=9216,
                 device=None, dtype=None, ops=None):
        super().__init__()
        ops = ops or nn
        self.gate_proj = ops.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = ops.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HybridBlock(nn.Module):
    """
    A single transformer block that uses either SSM or self-attention.
    """
    def __init__(self, hidden_size=2560, intermediate_size=9216,
                 use_ssm=True, has_mlp=True,
                 device=None, dtype=None, ops=None):
        super().__init__()
        self.use_ssm = use_ssm
        self.has_mlp = has_mlp

        self.input_layernorm = Qwen35RMSNorm(hidden_size, device=device, dtype=dtype)

        if use_ssm:
            self.linear_attn = SSMBlock(
                hidden_size=hidden_size,
                device=device, dtype=dtype, ops=ops
            )
        else:
            self.self_attn = GatedSelfAttention(
                hidden_size=hidden_size,
                device=device, dtype=dtype, ops=ops
            )

        if has_mlp:
            self.post_attention_layernorm = Qwen35RMSNorm(hidden_size, device=device, dtype=dtype)
            self.mlp = MLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                device=device, dtype=dtype, ops=ops
            )

    def forward(self, x, attention_mask=None, freqs_cis=None, linear_attention_mask=None):
        # Pre-norm + attention/SSM
        residual = x
        x_norm = self.input_layernorm(x)

        if self.use_ssm:
            x = residual + self.linear_attn(x_norm, attention_mask=linear_attention_mask)
        else:
            x = residual + self.self_attn(x_norm, attention_mask=attention_mask, freqs_cis=freqs_cis)

        # Pre-norm + MLP
        if self.has_mlp:
            residual = x
            x = residual + self.mlp(self.post_attention_layernorm(x))

        return x

