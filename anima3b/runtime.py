from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from backend import memory_management
from backend.operations import ForgeOperations, using_forge_operations
from backend.patcher.clip import CLIP

from .adapter import ProgressiveCrossAdapter
from .files import (
    ARCHITECTURE,
    CONNECTOR_PREFIX,
    V2_ARCHITECTURE,
    adapters,
    bundle_metadata,
    qwen35_models,
)
from .qwen35 import Qwen35HybridModel
from .semantic_v2 import BundledV2Models, QualityAnchoredSemanticConnectorV2
from .tokenizer import Qwen35Tokenizer

logger = logging.getLogger(__name__)
LAYER_INDICES = (7, 15, 23, 31)


@dataclass
class _V2Run:
    source: torch.Tensor
    target_ids: torch.Tensor
    target_weights: torch.Tensor
    semantic: list[torch.Tensor]
    semantic_mask: torch.Tensor


class Anima3BRuntime:
    def __init__(self) -> None:
        self._qwen_path: str | None = None
        self._qwen: Qwen35HybridModel | None = None
        self._qwen_clip: CLIP | None = None
        self._tokenizer: Qwen35Tokenizer | None = None
        self._adapter_key: tuple[str, int] | None = None
        self._adapter: ProgressiveCrossAdapter | None = None
        self._active_bundle_path: str | None = None
        self._active_bundle_metadata: dict[str, str] | None = None
        self._v2_key: tuple[str, int] | None = None
        self._v2_models: BundledV2Models | None = None
        self._v2_sampling_patcher = None
        self._v2_diffusion_model = None
        self._v2_original_forward = None
        self._v2_runs: dict[int, _V2Run] = {}
        self._v2_run_counter = 0

    @staticmethod
    def _require_anima(sd_model):
        engine = getattr(sd_model, "text_processing_engine_anima", None)
        clip = getattr(getattr(sd_model, "forge_objects", None), "clip", None)
        if engine is None or clip is None:
            raise RuntimeError("Anima 3.8B requires a loaded Anima checkpoint.")
        return engine, clip

    @staticmethod
    def _checkpoint_path(sd_model) -> str:
        info = getattr(sd_model, "sd_checkpoint_info", None)
        path = getattr(info, "filename", None) or getattr(sd_model, "filename", None)
        if not path:
            raise RuntimeError("Could not determine the selected Anima checkpoint.")
        return str(path)

    def is_v2_bundle(self, sd_model) -> bool:
        try:
            return bundle_metadata(self._checkpoint_path(sd_model)) is not None
        except RuntimeError:
            return False

    @staticmethod
    def _same_patcher(left, right) -> bool:
        if left is right:
            return True
        if left is None or right is None:
            return False
        try:
            return left.is_clone(right) or right.is_clone(left)
        except Exception:
            return False

    @classmethod
    def _unload_patchers(cls, *patchers) -> None:
        targets = [patcher for patcher in patchers if patcher is not None]
        if not targets:
            return
        removed = False
        loaded_models = memory_management.current_loaded_models
        for index in range(len(loaded_models) - 1, -1, -1):
            loaded = loaded_models[index]
            patcher = loaded.model
            if not any(cls._same_patcher(patcher, target) for target in targets):
                continue
            loaded.model_unload()
            loaded_models.pop(index)
            removed = True
        if removed:
            memory_management.soft_empty_cache()

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
        state["parameter_bank.layer_mix_logits"] = state.pop("layer_mix_logits")
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
            [chunk.qwen_tokens],
            [chunk.qwen_multipliers],
        )[0].unsqueeze(0)
        target_ids = torch.tensor(
            chunk.t5_tokens,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        target_weights = torch.tensor(
            chunk.t5_multipliers,
            device=device,
            dtype=dtype,
        ).reshape(1, -1, 1)
        return source.to(device=device, dtype=dtype), target_ids, target_weights

    def _extract_prompt_features(self, native_engine, native_clip, prompt):
        qwen, tokenizer, qwen_clip = self._load_qwen()
        native_adapter = native_clip.cond_stage_model.qwen3_06b.llm_adapter
        dtype = native_adapter.embed.weight.dtype
        offload_device = native_clip.patcher.offload_device

        native_rows = []
        memory_management.load_model_gpu(native_clip.patcher)
        try:
            device = native_clip.patcher.load_device
            for line in prompt:
                source, target_ids, target_weights = self._native_inputs(
                    native_engine,
                    str(line),
                    device,
                    dtype,
                )
                native_rows.append(
                    (
                        source.to(offload_device),
                        target_ids.to(offload_device),
                        target_weights.to(offload_device),
                    )
                )
        finally:
            self._unload_patchers(native_clip.patcher)

        semantic_rows = []
        memory_management.load_model_gpu(qwen_clip.patcher)
        try:
            device = qwen_clip.patcher.load_device
            for line in prompt:
                semantic, semantic_mask = self._semantic_layers(
                    qwen,
                    tokenizer,
                    str(line),
                    device,
                )
                semantic_rows.append(
                    (
                        [state.to(offload_device, dtype=dtype) for state in semantic],
                        semantic_mask.to(offload_device),
                    )
                )
        finally:
            self._unload_patchers(qwen_clip.patcher)
        return native_adapter, native_rows, semantic_rows

    @staticmethod
    def _adapter_metadata(metadata: dict[str, str]) -> dict[str, str]:
        prefix = "anima_v2_adapter_"
        return {
            name[len(prefix) :]: value
            for name, value in metadata.items()
            if name.startswith(prefix)
        }

    def _load_v2_models(
        self,
        sd_model,
        bundle_path: str,
        metadata: dict[str, str],
    ) -> BundledV2Models:
        _, native_clip = self._require_anima(sd_model)
        native_adapter = native_clip.cond_stage_model.qwen3_06b.llm_adapter
        key = (bundle_path, id(native_adapter))
        if key == self._v2_key and self._v2_models is not None:
            return self._v2_models

        adapter_metadata = self._adapter_metadata(metadata)
        if adapter_metadata.get("architecture") != V2_ARCHITECTURE:
            raise RuntimeError("The bundled connector is not Semantic Connector v2.")
        config = {
            "num_queries": int(adapter_metadata.get("semantic_query_tokens", "64")),
            "resampler_blocks": int(
                adapter_metadata.get("semantic_resampler_blocks", "6")
            ),
            "resampler_dim": int(
                adapter_metadata.get("semantic_resampler_dim", "2048")
            ),
            "resampler_heads": int(
                adapter_metadata.get("semantic_resampler_heads", "16")
            ),
            "mlp_hidden_dim": int(
                adapter_metadata.get(
                    "semantic_resampler_mlp_hidden_dim",
                    "5632",
                )
            ),
        }
        prefix = metadata.get("anima_v2_connector_prefix", CONNECTOR_PREFIX)
        connector_state = {}
        with safe_open(bundle_path, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                if name.startswith(prefix):
                    connector_state[name[len(prefix) :]] = checkpoint.get_tensor(name)
        if not connector_state:
            raise RuntimeError("The selected v2 bundle contains no connector tensors.")
        connector_state[
            "quality_anchor.parameter_bank.layer_mix_logits"
        ] = connector_state.pop("quality_anchor.layer_mix_logits")
        connector_state[
            "semantic_resampler.parameter_bank.query_tokens"
        ] = connector_state.pop("semantic_resampler.query_tokens")
        connector_state[
            "semantic_resampler.parameter_bank.layer_embeddings"
        ] = connector_state.pop("semantic_resampler.layer_embeddings")

        dtype = native_adapter.embed.weight.dtype
        with torch.device("meta"):
            with using_forge_operations(
                device=torch.device("meta"),
                dtype=dtype,
                manual_cast_enabled=True,
            ):
                connector = QualityAnchoredSemanticConnectorV2(
                    native_adapter=native_adapter,
                    **config,
                )
        incompatible = connector.load_state_dict(
            connector_state,
            strict=True,
            assign=True,
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Bundled v2 connector mismatch: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        del connector_state
        connector.eval().requires_grad_(False)
        models = BundledV2Models(native_adapter, connector)
        models.eval().requires_grad_(False)
        self._v2_key = key
        self._v2_models = models
        return models

    def _register_v2_run(self, run: _V2Run) -> int:
        self._v2_run_counter += 1
        self._v2_runs[self._v2_run_counter] = run
        return self._v2_run_counter

    def _encode_v2(self, native_engine, native_clip, prompt):
        _, native_rows, semantic_rows = self._extract_prompt_features(
            native_engine,
            native_clip,
            prompt,
        )
        placeholders = []
        run_ids = []
        for native, semantic in zip(native_rows, semantic_rows):
            source, target_ids, target_weights = native
            semantic_states, semantic_mask = semantic
            run_id = self._register_v2_run(
                _V2Run(
                    source=source,
                    target_ids=target_ids[:, :512],
                    target_weights=target_weights[:, :512],
                    semantic=semantic_states,
                    semantic_mask=semantic_mask,
                )
            )
            placeholder = source[:, :512]
            if placeholder.shape[1] < 512:
                placeholder = F.pad(
                    placeholder,
                    (0, 0, 0, 512 - placeholder.shape[1]),
                )
            placeholders.append(placeholder)
            run_ids.append(run_id)
        return {
            "crossattn": torch.cat(placeholders),
            "vector": torch.tensor(run_ids, dtype=torch.long).reshape(-1, 1),
        }

    def _expand_v2_context(
        self,
        context: torch.Tensor,
        timesteps: torch.Tensor,
        run_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self._v2_models is None:
            raise RuntimeError("The anima.3-8B-v2 connector is not loaded.")
        connector = self._v2_models.connector
        flat_ids = run_ids.reshape(-1).to(dtype=torch.long)
        if flat_ids.numel() != context.shape[0]:
            repeats = (context.shape[0] + flat_ids.numel() - 1) // flat_ids.numel()
            flat_ids = flat_ids.repeat(repeats)[: context.shape[0]]
        timestep_rows = timesteps.reshape(-1)
        if timestep_rows.numel() != context.shape[0]:
            repeats = (
                context.shape[0] + timestep_rows.numel() - 1
            ) // timestep_rows.numel()
            timestep_rows = timestep_rows.repeat(repeats)[: context.shape[0]]

        outputs: list[torch.Tensor | None] = [None] * context.shape[0]
        dtype = self._v2_models.native_adapter.embed.weight.dtype
        for run_id in flat_ids.unique().tolist():
            run = self._v2_runs.get(int(run_id))
            if run is None:
                raise RuntimeError(
                    f"Anima v2 conditioning run {run_id} is unavailable; "
                    "re-encode the prompt."
                )
            indices = (flat_ids == run_id).nonzero(as_tuple=False).reshape(-1)
            count = indices.numel()
            source = run.source.to(context.device, dtype=dtype).expand(count, -1, -1)
            target_ids = run.target_ids.to(context.device).expand(count, -1)
            semantic = [
                state.to(context.device, dtype=dtype).expand(count, -1, -1)
                for state in run.semantic
            ]
            semantic_mask = run.semantic_mask.to(context.device).expand(count, -1)
            expanded = connector(
                source,
                target_ids,
                semantic,
                semantic_source_mask=semantic_mask,
                timesteps=timestep_rows[indices].to(context.device),
            )
            weights = run.target_weights.to(
                context.device,
                dtype=expanded.dtype,
            ).expand(count, -1, -1)
            expanded = expanded * weights[:, : expanded.shape[1]]
            if expanded.shape[1] < 512:
                expanded = F.pad(
                    expanded,
                    (0, 0, 0, 512 - expanded.shape[1]),
                )
            for output_index, row_index in enumerate(indices.tolist()):
                outputs[row_index] = expanded[output_index : output_index + 1]
        return torch.cat(outputs).to(dtype=context.dtype)

    @staticmethod
    def _offload_conditioning(value, device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, list):
            return [Anima3BRuntime._offload_conditioning(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(
                Anima3BRuntime._offload_conditioning(item, device) for item in value
            )
        if isinstance(value, dict):
            return {
                key: Anima3BRuntime._offload_conditioning(item, device)
                for key, item in value.items()
            }
        return value

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
            try:
                result = original(prompt)
                return self._offload_conditioning(
                    result,
                    native_clip.patcher.offload_device,
                )
            finally:
                self._unload_patchers(native_clip.patcher)

        if self._active_bundle_metadata is not None:
            return self._encode_v2(native_engine, native_clip, prompt)

        native_adapter, native_rows, semantic_rows = self._extract_prompt_features(
            native_engine,
            native_clip,
            prompt,
        )
        memory_management.load_model_gpu(native_clip.patcher)
        try:
            device = native_clip.patcher.load_device
            dtype = native_adapter.embed.weight.dtype
            adapter = self._load_adapter(adapter_name, native_adapter)
            adapter.to(device=device, dtype=dtype)
            outputs = []
            for native, semantic in zip(native_rows, semantic_rows):
                source, target_ids, target_weights = native
                semantic_states, semantic_mask = semantic
                source = source.to(device=device, dtype=dtype)
                target_ids = target_ids.to(device=device)
                target_weights = target_weights.to(device=device, dtype=dtype)
                semantic_states = [
                    state.to(device=device, dtype=dtype) for state in semantic_states
                ]
                semantic_mask = semantic_mask.to(device=device)
                expanded = adapter(
                    source,
                    target_ids[:, :512],
                    semantic_states,
                    semantic_mask=semantic_mask,
                )
                if strength != 1.0:
                    native_context = native_adapter(source, target_ids[:, :512])
                    expanded = native_context + float(strength) * (
                        expanded - native_context
                    )
                expanded = expanded * target_weights[:, : expanded.shape[1]]
                if expanded.shape[1] < 512:
                    expanded = F.pad(
                        expanded,
                        (0, 0, 0, 512 - expanded.shape[1]),
                    )
                outputs.append(expanded.to(native_clip.patcher.offload_device))
            return outputs
        finally:
            self._unload_patchers(native_clip.patcher)

    def _install_v2(
        self,
        processing,
        path: str,
        metadata: dict[str, str],
    ) -> None:
        models = self._load_v2_models(processing.sd_model, path, metadata)
        unet = processing.sd_model.forge_objects.unet
        self._v2_sampling_patcher = unet.add_extra_torch_module_during_sampling(
            models,
            cast_to_unet_dtype=False,
        )
        diffusion_model = unet.model.diffusion_model
        self._v2_diffusion_model = diffusion_model
        self._v2_original_forward = diffusion_model.forward
        original_forward = self._v2_original_forward

        def patched_forward(x, timesteps, context, **kwargs):
            run_ids = kwargs.pop("y", None)
            if run_ids is not None:
                context = self._expand_v2_context(context, timesteps, run_ids)
            return original_forward(x, timesteps, context, **kwargs)

        diffusion_model.forward = patched_forward

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
        checkpoint_path = self._checkpoint_path(processing.sd_model)
        metadata = bundle_metadata(checkpoint_path)
        self._active_bundle_path = checkpoint_path if metadata is not None else None
        self._active_bundle_metadata = metadata
        self._v2_runs.clear()

        original = processing.sd_model.get_learned_conditioning
        processing.sd_model._anima3b_original_get_learned_conditioning = original
        if metadata is not None:
            self._install_v2(processing, checkpoint_path, metadata)
            strength = 1.0
            if negative_strength is not None:
                negative_strength = 1.0

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
        if metadata is not None:
            adapter_file = metadata.get(
                "anima_v2_adapter_filename",
                Path(checkpoint_path).name,
            )
            processing.extra_generation_params.update(
                {
                    "Anima 3.8B adapter": adapter_file,
                    "Anima 3.8B strength": 1.0,
                    "Anima 3.8B architecture": V2_ARCHITECTURE,
                    "Anima 3.8B bundle": Path(checkpoint_path).name,
                }
            )
        else:
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

    def restore(self, processing) -> None:
        model = processing.sd_model
        original = getattr(model, "_anima3b_original_get_learned_conditioning", None)
        if original is not None:
            model.get_learned_conditioning = original
            del model._anima3b_original_get_learned_conditioning
        if (
            self._v2_diffusion_model is not None
            and self._v2_original_forward is not None
        ):
            self._v2_diffusion_model.forward = self._v2_original_forward
        unet = getattr(getattr(model, "forge_objects", None), "unet", None)
        if unet is not None and self._v2_sampling_patcher is not None:
            unet.extra_model_patchers_during_sampling = [
                patcher
                for patcher in unet.extra_model_patchers_during_sampling
                if patcher is not self._v2_sampling_patcher
            ]
            self._unload_patchers(self._v2_sampling_patcher)
        self._v2_sampling_patcher = None
        self._v2_diffusion_model = None
        self._v2_original_forward = None
        self._active_bundle_path = None
        self._active_bundle_metadata = None
        self._v2_runs.clear()
        processing.cached_c = [None, None, None]
        processing.cached_uc = [None, None, None]
