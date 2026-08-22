from __future__ import annotations

import torch
from torch import nn


from .layers import HybridBlock, Qwen35RMSNorm, _precompute_freqs_cis


class Qwen35HybridModel(nn.Module):
    """Qwen3.5 4B hybrid text backbone used by the Anima adapter."""

    SELF_ATTN_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}
    NUM_LAYERS = 32
    HIDDEN_SIZE = 2560
    INTERMEDIATE_SIZE = 9216
    VOCAB_SIZE = 248320
    OUTPUT_DIM = 1024
    ROTARY_DIM = 64
    ROPE_THETA = 10000000.0

    def __init__(self, config_dict=None, dtype=None, device=None, operations=None):
        super().__init__()
        operations = operations or nn
        self.num_layers = self.NUM_LAYERS
        self.dtype = dtype
        self.embed_tokens = operations.Embedding(
            self.VOCAB_SIZE, self.HIDDEN_SIZE, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList([
            HybridBlock(
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                use_ssm=index not in self.SELF_ATTN_LAYERS,
                has_mlp=index != 31,
                device=device,
                dtype=dtype,
                ops=operations,
            )
            for index in range(self.NUM_LAYERS)
        ])
        self.norm = nn.Sequential(
            operations.Linear(
                self.HIDDEN_SIZE, self.OUTPUT_DIM, bias=True,
                device=device, dtype=dtype,
            ),
            Qwen35RMSNorm(self.OUTPUT_DIM, device=device, dtype=dtype),
            nn.SiLU(),
            operations.Linear(
                self.OUTPUT_DIM, self.OUTPUT_DIM, bias=True,
                device=device, dtype=dtype,
            ),
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.embed_tokens = embeddings

    def forward(
        self,
        input_ids,
        attention_mask=None,
        embeds=None,
        num_tokens=None,
        intermediate_output=None,
        final_layer_norm_intermediate=True,
        dtype=None,
        embeds_info=None,
        **kwargs,
    ):
        del num_tokens, final_layer_norm_intermediate, embeds_info, kwargs
        if embeds is not None:
            hidden_states = embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        sequence_length = hidden_states.shape[1]
        rotary = _precompute_freqs_cis(
            self.ROTARY_DIM,
            sequence_length,
            theta=self.ROPE_THETA,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        linear_attention_mask = attention_mask
        attention_bias = None
        if attention_mask is not None:
            mask_fill = torch.finfo(hidden_states.dtype).min / 4
            causal = torch.empty(
                sequence_length,
                sequence_length,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            ).fill_(mask_fill).triu_(1)
            padding = 1.0 - attention_mask.to(hidden_states.dtype).reshape(
                attention_mask.shape[0], 1, -1, attention_mask.shape[-1]
            ).expand(
                attention_mask.shape[0],
                1,
                sequence_length,
                attention_mask.shape[-1],
            )
            padding = padding.masked_fill(padding.to(torch.bool), mask_fill)
            attention_bias = causal + padding
        elif sequence_length > 1:
            mask_fill = torch.finfo(hidden_states.dtype).min / 4
            attention_bias = torch.empty(
                sequence_length,
                sequence_length,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            ).fill_(mask_fill).triu_(1)

        intermediate = None
        for index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_bias,
                freqs_cis=rotary,
                linear_attention_mask=linear_attention_mask,
            )
            if isinstance(intermediate_output, int) and index == intermediate_output:
                intermediate = hidden_states.clone()
            elif isinstance(intermediate_output, list) and index in intermediate_output:
                if intermediate is None:
                    intermediate = {}
                intermediate[index] = hidden_states.clone()

        return self.norm(hidden_states), intermediate
