from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from backend.nn.anima import Attention

from .adapter import ProgressiveCrossAdapter


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = timesteps.float().reshape(-1, 1) * 1_000.0 * frequencies.reshape(1, -1)
    embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return F.pad(embedding, (0, dim % 2)) if dim % 2 else embedding


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if query_dim % num_heads:
            raise ValueError("query_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.query_dim = query_dim
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.o_proj = nn.Linear(query_dim, query_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, query_tokens, _ = query.shape
        context_tokens = context.shape[1]

        def heads(value: torch.Tensor, tokens: int) -> torch.Tensor:
            return value.reshape(
                batch,
                tokens,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        q = heads(self.q_proj(query), query_tokens)
        k = heads(self.k_proj(context), context_tokens)
        v = heads(self.v_proj(context), context_tokens)
        mask = None
        if context_mask is not None:
            mask = context_mask.to(torch.bool).reshape(
                batch,
                1,
                1,
                context_tokens,
            )
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attended = attended.transpose(1, 2).reshape(
            batch,
            query_tokens,
            self.query_dim,
        )
        return self.o_proj(attended)


class TimestepModulatedNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(
        self,
        value: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return self.norm(value) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SemanticResamplerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        qwen_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.cross_norm = TimestepModulatedNorm(dim)
        self.self_norm = TimestepModulatedNorm(dim)
        self.mlp_norm = TimestepModulatedNorm(dim)
        self.source_norm = nn.LayerNorm(qwen_dim)
        self.cross_attention = MultiHeadAttention(dim, qwen_dim, num_heads)
        self.self_attention = MultiHeadAttention(dim, dim, num_heads)
        self.mlp_in = nn.Linear(dim, 2 * mlp_hidden_dim, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden_dim, dim, bias=False)
        self.time_modulation = nn.Linear(dim, 6 * dim, bias=True)

    def forward(
        self,
        queries: torch.Tensor,
        qwen_features: torch.Tensor,
        timestep_embedding: torch.Tensor,
        qwen_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        modulation = self.time_modulation(F.silu(timestep_embedding))
        cross_scale, cross_shift, self_scale, self_shift, mlp_scale, mlp_shift = (
            modulation.chunk(6, dim=-1)
        )
        queries = queries + self.cross_attention(
            self.cross_norm(queries, cross_scale, cross_shift),
            self.source_norm(qwen_features),
            qwen_mask,
        )
        normalized = self.self_norm(queries, self_scale, self_shift)
        queries = queries + self.self_attention(normalized, normalized)
        normalized = self.mlp_norm(queries, mlp_scale, mlp_shift)
        gate, value = self.mlp_in(normalized).chunk(2, dim=-1)
        return queries + self.mlp_out(F.silu(gate) * value)


class _ResamplerParameterBank(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_queries: int,
        model_dim: int,
        qwen_dim: int,
    ) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(
            torch.empty(1, num_queries, model_dim)
        )
        self.layer_embeddings = nn.Parameter(
            torch.empty(num_layers, 1, qwen_dim)
        )


class TimestepAwareSemanticResampler(nn.Module):
    def __init__(
        self,
        qwen_dim: int,
        output_dim: int,
        num_layers: int,
        num_queries: int,
        num_blocks: int,
        model_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.parameter_bank = _ResamplerParameterBank(
            num_layers,
            num_queries,
            model_dim,
            qwen_dim,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.blocks = nn.ModuleList(
            SemanticResamplerBlock(
                model_dim,
                qwen_dim,
                num_heads,
                mlp_hidden_dim,
            )
            for _ in range(num_blocks)
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(
            model_dim,
            output_dim,
            bias=False,
        )

    @property
    def query_tokens(self) -> torch.Tensor:
        return self.parameter_bank.query_tokens

    @property
    def layer_embeddings(self) -> torch.Tensor:
        return self.parameter_bank.layer_embeddings

    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        timesteps: torch.Tensor,
        source_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if len(hidden_states) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} Qwen layers, "
                f"got {len(hidden_states)}"
            )
        batch = hidden_states[0].shape[0]
        layer_streams = [
            hidden
            + self.layer_embeddings[index].to(
                device=hidden.device,
                dtype=hidden.dtype,
            ).unsqueeze(0)
            for index, hidden in enumerate(hidden_states)
        ]
        qwen_features = torch.cat(layer_streams, dim=1)
        qwen_mask = None
        if source_mask is not None:
            qwen_mask = torch.cat([source_mask] * self.num_layers, dim=1)
        time = sinusoidal_timestep_embedding(timesteps, self.model_dim).to(
            dtype=qwen_features.dtype
        )
        time = self.time_mlp(time)
        queries = self.query_tokens.to(
            device=qwen_features.device,
            dtype=qwen_features.dtype,
        ).expand(batch, -1, -1)
        for block in self.blocks:
            queries = block(queries, qwen_features, time, qwen_mask)
        return self.output_projection(self.output_norm(queries))


class QualityAnchoredSemanticConnectorV2(nn.Module):
    architecture = "anima_qwen35_quality_anchored_semantic_connector_v2"

    def __init__(
        self,
        native_adapter: nn.Module,
        semantic_source_dim: int = 2560,
        layer_indices: Sequence[int] = (7, 15, 23, 31),
        num_queries: int = 64,
        resampler_blocks: int = 6,
        resampler_dim: int = 2048,
        resampler_heads: int = 16,
        mlp_hidden_dim: int = 5632,
    ) -> None:
        super().__init__()
        self.layer_indices = tuple(int(index) for index in layer_indices)
        self.quality_anchor = ProgressiveCrossAdapter(
            native_adapter,
            semantic_source_dim=semantic_source_dim,
            layer_indices=self.layer_indices,
        )
        model_dim = native_adapter.embed.weight.shape[1]
        num_heads = native_adapter.blocks[0].self_attn.n_heads
        head_dim = model_dim // num_heads
        self.semantic_resampler = TimestepAwareSemanticResampler(
            qwen_dim=semantic_source_dim,
            output_dim=model_dim,
            num_layers=len(self.layer_indices),
            num_queries=num_queries,
            num_blocks=resampler_blocks,
            model_dim=resampler_dim,
            num_heads=resampler_heads,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self.v2_query_norms = nn.ModuleList(
            nn.RMSNorm(model_dim, eps=1e-6) for _ in native_adapter.blocks
        )
        self.v2_semantic_norms = nn.ModuleList(
            nn.RMSNorm(model_dim, eps=1e-6) for _ in native_adapter.blocks
        )
        self.v2_attentions = nn.ModuleList(
            Attention(model_dim, model_dim, num_heads, head_dim)
            for _ in native_adapter.blocks
        )

    @property
    def native_adapter(self) -> nn.Module:
        return self.quality_anchor.native_adapter

    @staticmethod
    def _mask(mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None
        mask = mask.to(torch.bool)
        return mask.unsqueeze(1).unsqueeze(1) if mask.ndim == 2 else mask

    def forward(
        self,
        native_source: torch.Tensor,
        target_input_ids: torch.Tensor,
        semantic_hidden_states: Sequence[torch.Tensor],
        target_attention_mask: torch.Tensor | None = None,
        native_source_mask: torch.Tensor | None = None,
        semantic_source_mask: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        include_inserted_blocks: bool = True,
        include_v2: bool = True,
        **_ignored,
    ) -> torch.Tensor:
        if timesteps is None:
            raise ValueError("Semantic Connector v2 requires diffusion timesteps")
        if len(semantic_hidden_states) != len(self.layer_indices):
            raise ValueError(
                f"Expected {len(self.layer_indices)} semantic layers, "
                f"got {len(semantic_hidden_states)}"
            )

        target_attention_mask = self._mask(target_attention_mask)
        native_source_mask = self._mask(native_source_mask)
        semantic_attention_mask = self._mask(semantic_source_mask)
        semantic_bank = None
        if include_inserted_blocks and include_v2:
            semantic_bank = self.semantic_resampler(
                semantic_hidden_states,
                timesteps,
                semantic_source_mask,
            )

        native = self.native_adapter
        x = native.in_proj(native.embed(target_input_ids))
        query_positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        native_positions = torch.arange(
            native_source.shape[1], device=x.device
        ).unsqueeze(0)
        anchor_positions = torch.arange(
            semantic_hidden_states[0].shape[1], device=x.device
        ).unsqueeze(0)
        bank_rope = None
        if semantic_bank is not None:
            bank_positions = torch.arange(
                semantic_bank.shape[1], device=x.device
            ).unsqueeze(0)
            bank_rope = native.rotary_emb(x, bank_positions)
        query_rope = native.rotary_emb(x, query_positions)
        native_rope = native.rotary_emb(x, native_positions)
        anchor_rope = native.rotary_emb(x, anchor_positions)
        anchor_mix = self.quality_anchor.layer_mix_logits.float().softmax(
            dim=-1
        ).to(device=x.device, dtype=x.dtype)

        for index, native_block in enumerate(native.blocks):
            x = native_block(
                x,
                native_source,
                target_attention_mask=target_attention_mask,
                source_attention_mask=native_source_mask,
                position_embeddings=query_rope,
                position_embeddings_context=native_rope,
            )
            if not include_inserted_blocks:
                continue
            anchor_source = self.quality_anchor.source_norms[index](
                self.quality_anchor._mixed_source(
                    semantic_hidden_states,
                    anchor_mix,
                    index,
                )
            )
            x = x + self.quality_anchor.semantic_attentions[index](
                self.quality_anchor.query_norms[index](x),
                mask=semantic_attention_mask,
                context=anchor_source,
                position_embeddings=query_rope,
                position_embeddings_context=anchor_rope,
            )
            if semantic_bank is not None:
                x = x + self.v2_attentions[index](
                    self.v2_query_norms[index](x),
                    context=self.v2_semantic_norms[index](semantic_bank),
                    position_embeddings=query_rope,
                    position_embeddings_context=bank_rope,
                )
        return native.norm(native.out_proj(x))


class BundledV2Models(nn.Module):
    """One Forge-managed sampling unit with no duplicate native adapter."""

    def __init__(self, native_adapter: nn.Module, connector: nn.Module) -> None:
        super().__init__()
        self.native_adapter = native_adapter
        self.connector = connector
