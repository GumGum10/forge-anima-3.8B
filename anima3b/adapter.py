from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from backend.nn.anima import Attention


class _AdapterParameterBank(nn.Module):
    def __init__(self, blocks: int, layers: int) -> None:
        super().__init__()
        self.layer_mix_logits = nn.Parameter(torch.zeros(blocks, layers))


class ProgressiveCrossAdapter(nn.Module):
    """Inference-only Forge port of the trained progressive cross adapter."""

    def __init__(
        self,
        native_adapter: nn.Module,
        semantic_source_dim: int = 2560,
        layer_indices: Sequence[int] = (7, 15, 23, 31),
    ) -> None:
        super().__init__()
        object.__setattr__(self, "native_adapter", native_adapter)
        self.layer_indices = tuple(layer_indices)
        model_dim = native_adapter.embed.weight.shape[1]
        num_heads = native_adapter.blocks[0].self_attn.n_heads
        head_dim = model_dim // num_heads

        self.query_norms = nn.ModuleList(
            nn.RMSNorm(model_dim, eps=1e-6) for _ in native_adapter.blocks
        )
        self.source_norms = nn.ModuleList(
            nn.RMSNorm(semantic_source_dim, eps=1e-6)
            for _ in native_adapter.blocks
        )
        self.semantic_attentions = nn.ModuleList(
            Attention(model_dim, semantic_source_dim, num_heads, head_dim)
            for _ in native_adapter.blocks
        )
        self.parameter_bank = _AdapterParameterBank(
            len(native_adapter.blocks),
            len(self.layer_indices),
        )

    @property
    def layer_mix_logits(self) -> torch.Tensor:
        return self.parameter_bank.layer_mix_logits

    @staticmethod
    def _mask(mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None
        mask = mask.to(torch.bool)
        return mask.unsqueeze(1).unsqueeze(1) if mask.ndim == 2 else mask

    @staticmethod
    def _mixed_source(
        semantic_states: Sequence[torch.Tensor],
        mix: torch.Tensor,
        block_index: int,
    ) -> torch.Tensor:
        return sum(
            state * mix[block_index, state_index]
            for state_index, state in enumerate(semantic_states)
        )

    def forward(
        self,
        native_source: torch.Tensor,
        target_ids: torch.Tensor,
        semantic_states: Sequence[torch.Tensor],
        semantic_mask: torch.Tensor | None = None,
        target_attention_mask: torch.Tensor | None = None,
        native_source_mask: torch.Tensor | None = None,
        semantic_source_mask: torch.Tensor | None = None,
        **_ignored,
    ) -> torch.Tensor:
        if len(semantic_states) != len(self.layer_indices):
            raise ValueError(
                f"Expected {len(self.layer_indices)} semantic layers, "
                f"got {len(semantic_states)}"
            )

        native = self.native_adapter
        target_attention_mask = self._mask(target_attention_mask)
        native_source_mask = self._mask(native_source_mask)
        semantic_mask = self._mask(
            semantic_source_mask
            if semantic_source_mask is not None
            else semantic_mask
        )
        x = native.in_proj(native.embed(target_ids))
        query_positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        native_positions = torch.arange(
            native_source.shape[1], device=x.device
        ).unsqueeze(0)
        semantic_positions = torch.arange(
            semantic_states[0].shape[1], device=x.device
        ).unsqueeze(0)
        query_rope = native.rotary_emb(x, query_positions)
        native_rope = native.rotary_emb(x, native_positions)
        semantic_rope = native.rotary_emb(x, semantic_positions)
        mix = self.layer_mix_logits.float().softmax(dim=-1).to(
            device=x.device,
            dtype=x.dtype,
        )

        for index, block in enumerate(native.blocks):
            x = block(
                x,
                native_source,
                target_attention_mask=target_attention_mask,
                source_attention_mask=native_source_mask,
                position_embeddings=query_rope,
                position_embeddings_context=native_rope,
            )
            semantic_source = self._mixed_source(
                semantic_states,
                mix,
                index,
            )
            semantic_source = self.source_norms[index](semantic_source)
            x = x + self.semantic_attentions[index](
                self.query_norms[index](x),
                mask=semantic_mask,
                context=semantic_source,
                position_embeddings=query_rope,
                position_embeddings_context=semantic_rope,
            )

        return native.norm(native.out_proj(x))
