from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from backend import memory_management
from backend.operations import ForgeOperations, using_forge_operations
from backend.patcher.clip import CLIP

from .adapter import ProgressiveCrossAdapter
from .files import ARCHITECTURE, adapters, qwen35_models
from .qwen35 import Qwen35HybridModel
from .tokenizer import Qwen35Tokenizer

logger = logging.getLogger(__name__)
LAYER_INDICES = (7, 15, 23, 31)


class Anima3BRuntime:
    def __init__(self) -> None:
        self._qwen_path: str | None = None
        self._qwen: Qwen35HybridModel | None = None
        self._qwen_clip: CLIP | None = None
        self._tokenizer: Qwen35Tokenizer | None = None
        self._adapter_key: tuple[str, int] | None = None
        self._adapter: ProgressiveCrossAdapter | None = None

    @staticmethod
    def _require_anima(sd_model):
        engine = getattr(sd_model, "text_processing_engine_anima", None)
        clip = getattr(getattr(sd_model, "forge_objects", None), "clip", None)
        if engine is None or clip is None:
            raise RuntimeError("Anima 3.8B requires a loaded Anima checkpoint.")
        return engine, clip

    def _load_qwen(self) -> tuple[Qwen35HybridModel, Qwen35Tokenizer, CLIP]:
        choices = qwen35_models()
        if not choices:
            raise FileNotFoundError(
                "qwen35_4b.safetensors was not found in models/text_encoder."
            )
        path = choices.get("qwen35_4b.safetensors") or next(iter(choices.values()))
        if self._qwen_path == path and self._qwen is not None:
            return self._qwen, self._tokenizer, self._qwen_clip

        logger.info("Loading Qwen3.5 4B from %s", path)
        state = load_file(path, device="cpu")
        dtype = next(
            (
                state[key].dtype
                for key in (
                    "norm.1.weight",
                    "layers.0.input_layernorm.weight",
                    "model.norm.weight",
                )
                if key in state
            ),
            torch.bfloat16,
        )
        with torch.device("meta"):
            with using_forge_operations(
                device=torch.device("meta"),
                dtype=dtype,
                manual_cast_enabled=True,
            ):
                model = Qwen35HybridModel(
                    dtype=dtype,
                    device=None,
                    operations=ForgeOperations,
                )
        incompatible = model.load_state_dict(state, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Qwen3.5 checkpoint mismatch: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        del state
        model.eval()
        tokenizer = Qwen35Tokenizer()
        clip = CLIP(model_dict={"qwen35_4b": model}, tokenizer_dict={})
        self._qwen_path = path
        self._qwen = model
        self._tokenizer = tokenizer
        self._qwen_clip = clip
        self._adapter_key = None
        self._adapter = None
        return model, tokenizer, clip

    def _load_adapter(self, name: str, native_adapter) -> ProgressiveCrossAdapter:
        choices = adapters()
        path = choices.get(name)
        if path is None:
            raise FileNotFoundError(
                f"Adapter '{name}' is unavailable. Refresh Forge and select it again."
            )
        key = (path, id(native_adapter))
        if key == self._adapter_key and self._adapter is not None:
            return self._adapter

        adapter = ProgressiveCrossAdapter(native_adapter)
        state = load_file(path, device="cpu")
        incompatible = adapter.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Adapter checkpoint mismatch: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        del state
        adapter.eval().requires_grad_(False)
        self._adapter_key = key
        self._adapter = adapter
        return adapter

    @staticmethod
    def _semantic_layers(model, tokenizer, line: str, device: torch.device):
        ids = tokenizer([line])["input_ids"]
        token_ids = torch.tensor(ids, device=device, dtype=torch.long)
        attention_mask = torch.ones_like(token_ids)
        output, intermediate = model(
            token_ids,
            attention_mask=attention_mask,
            intermediate_output=list(LAYER_INDICES),
            dtype=torch.float32,
        )
        del output
        if not isinstance(intermediate, dict):
            raise RuntimeError("Qwen3.5 did not return its semantic layers.")
        return [intermediate[index] for index in LAYER_INDICES], attention_mask

    @staticmethod
    def _native_inputs(native_engine, line: str, device, dtype):
        chunks = native_engine.tokenize_line(line)
        if len(chunks) != 1:
            raise RuntimeError("Anima 3.8B expects one prompt chunk.")
        chunk = chunks[0]
        source = native_engine.process_tokens(
            [chunk.qwen_tokens], [chunk.qwen_multipliers]
        )[0].unsqueeze(0)
        target_ids = torch.tensor(
            chunk.t5_tokens, device=device, dtype=torch.long
        ).unsqueeze(0)
        target_weights = torch.tensor(
            chunk.t5_multipliers, device=device, dtype=dtype
        ).reshape(1, -1, 1)
        return source.to(device=device, dtype=dtype), target_ids, target_weights

    @torch.inference_mode()
    def encode(
        self,
        sd_model,
        prompt,
        adapter_name: str,
        strength: float,
        negative_strength: float | None,
    ):
        native_engine, native_clip = self._require_anima(sd_model)
        original = sd_model._anima3b_original_get_learned_conditioning
        if getattr(prompt, "is_negative_prompt", False):
            strength = negative_strength
        if strength is None or strength == 0.0:
            return original(prompt)

        qwen, tokenizer, qwen_clip = self._load_qwen()
        memory_management.load_model_gpu(native_clip.patcher)
        memory_management.load_model_gpu(qwen_clip.patcher)
        native_text_encoder = native_clip.cond_stage_model.qwen3_06b
        native_adapter = native_text_encoder.llm_adapter
        adapter = self._load_adapter(adapter_name, native_adapter)
        device = native_adapter.embed.weight.device
        dtype = native_adapter.embed.weight.dtype
        adapter.to(device=device, dtype=dtype)

        outputs = []
        for line in prompt:
            source, target_ids, target_weights = self._native_inputs(
                native_engine, str(line), device, dtype
            )
            semantic, semantic_mask = self._semantic_layers(
                qwen, tokenizer, str(line), device
            )
            semantic = [state.to(dtype=dtype) for state in semantic]
            expanded = adapter(
                source,
                target_ids[:, :512],
                semantic,
                semantic_mask=semantic_mask,
            )
            if strength != 1.0:
                native = native_adapter(source, target_ids[:, :512])
                expanded = native + float(strength) * (expanded - native)
            expanded = expanded * target_weights[:, : expanded.shape[1]]
            if expanded.shape[1] < 512:
                expanded = F.pad(expanded, (0, 0, 0, 512 - expanded.shape[1]))
            outputs.append(expanded)
        return outputs

    def install(
        self,
        processing,
        adapter_name: str,
        strength: float,
        negative_strength: float | None,
    ) -> None:
        self._require_anima(processing.sd_model)
        if hasattr(processing.sd_model, "_anima3b_original_get_learned_conditioning"):
            self.restore(processing)
        original = processing.sd_model.get_learned_conditioning
        processing.sd_model._anima3b_original_get_learned_conditioning = original

        def patched(prompt):
            return self.encode(
                processing.sd_model,
                prompt,
                adapter_name,
                strength,
                negative_strength,
            )

        processing.sd_model.get_learned_conditioning = patched
        processing.cached_c = [None, None, None]
        processing.cached_uc = [None, None, None]
        processing.extra_generation_params.update(
            {
                "Anima 3.8B adapter": Path(adapter_name).name,
                "Anima 3.8B strength": float(strength),
                "Anima 3.8B architecture": ARCHITECTURE,
            }
        )
        if negative_strength is not None:
            processing.extra_generation_params[
                "Anima 3.8B negative strength"
            ] = float(negative_strength)

    @staticmethod
    def restore(processing) -> None:
        model = processing.sd_model
        original = getattr(model, "_anima3b_original_get_learned_conditioning", None)
        if original is not None:
            model.get_learned_conditioning = original
            del model._anima3b_original_get_learned_conditioning
            processing.cached_c = [None, None, None]
            processing.cached_uc = [None, None, None]
