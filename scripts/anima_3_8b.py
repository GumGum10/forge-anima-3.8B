from __future__ import annotations

import gradio as gr

from modules import scripts
from modules.processing import StableDiffusionProcessing
from modules.ui_components import InputAccordion

from anima3b.files import adapters
from anima3b.runtime import Anima3BRuntime


class Anima3BScript(scripts.Script):
    sorting_priority = 260209301

    def __init__(self):
        super().__init__()
        self.runtime = Anima3BRuntime()

    def title(self):
        return "Anima 3.8B"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        choices = list(adapters())
        if not choices:
            choices = ["Anima-3.8B-expanded_adapter.safetensors"]
        with InputAccordion(False, label="Anima 3.8B (Qwen3.5)") as enabled:
            adapter = gr.Dropdown(
                label="Adapter",
                choices=choices,
                value=choices[0],
                info="Detected by checkpoint metadata in models/text_encoder.",
            )
            strength = gr.Slider(
                label="Adapter strength",
                minimum=0.0,
                maximum=2.0,
                value=1.0,
                step=0.05,
                info="1.0 is trained strength; 0.0 is native Anima.",
            )
            negative = gr.Checkbox(
                label="Use adapter on negative prompt",
                value=False,
                info="Off keeps negatives on the native Anima encoder.",
            )
            negative_strength = gr.Slider(
                label="Negative adapter strength",
                minimum=0.0,
                maximum=2.0,
                value=1.0,
                step=0.05,
                visible=False,
                info="Adapter strength applied to the negative prompt.",
            )
            negative.change(
                fn=lambda value: gr.update(visible=bool(value)),
                inputs=[negative],
                outputs=[negative_strength],
                show_progress=False,
                queue=False,
            )
        return [enabled, adapter, strength, negative, negative_strength]

    def process_batch(
        self,
        p: StableDiffusionProcessing,
        enabled: bool,
        adapter: str,
        strength: float,
        negative: bool = False,
        negative_strength: float = 1.0,
        **kwargs,
    ):
        if not enabled:
            return
        self.runtime.install(
            p,
            adapter,
            strength,
            float(negative_strength) if negative else None,
        )

    def process_before_every_sampling(
        self,
        p: StableDiffusionProcessing,
        enabled: bool,
        adapter: str,
        strength: float,
        negative: bool = False,
        negative_strength: float = 1.0,
        **kwargs,
    ):
        if not enabled or self.runtime._qwen_clip is None:
            return
        unet = p.sd_model.forge_objects.unet
        required = self.runtime._qwen_clip.patcher.model_size()
        current = unet.model_options.get(
            "extra_preserved_memory_during_sampling", 0
        )
        unet.model_options["extra_preserved_memory_during_sampling"] = max(
            current, required
        )

    def postprocess(self, p, processed, *args):
        self.runtime.restore(p)
