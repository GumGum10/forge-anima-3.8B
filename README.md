# Anima 3.8B for Forge Neo

A Forge Neo extension that runs the [Anima 3.8B](https://huggingface.co/lylogummy/Anima-3.8B)
diffusion model with its **Qwen3.5 4B progressive-cross adapter**, the same
pairing as the reference ComfyUI workflow.

The UI is deliberately small: an adapter dropdown, an adapter-strength slider,
and a switch for putting the negative prompt through the adapter too.

## Requirements

- Forge Neo `neo 2.28` or newer.
- The Anima checkpoint must already work natively in your Forge install
  (that is where `qwen_3_06b_base.safetensors` and `qwen_image_vae.safetensors`
  come from — this extension does not replace them, it adds a second encoder).

## Install

### 1. Install the extension

```bash
git clone https://github.com/GumGum10/forge-anima-3.8B extensions/sd-forge-anima-3-8b
```

Run that from your `sd-webui-forge-neo` folder, so you end up with:

```text
sd-webui-forge-neo/extensions/sd-forge-anima-3-8b/
```

There is nothing to pip-install; the Qwen3.5 tokenizer is bundled in
`qwen35_tokenizer/`, so the extension never touches the Hugging Face cache or
the network.

### 2. Download the models

From [lylogummy/Anima-3.8B](https://huggingface.co/lylogummy/Anima-3.8B/tree/main):

| Download this file | Put it here |
| --- | --- |
| `difussion_models/Anima-3.8B.safetensors` (7.5 GB) | `models/Stable-diffusion/Anima-3.8B.safetensors` |
| `text_encoders/qwen35_4b.safetensors` (4.8 GB) | `models/text_encoder/qwen35_4b.safetensors` |
| `text_encoders/Anima-3.8B-expanded_adapter.safetensors` (88 MB) | `models/text_encoder/Anima-3.8B-expanded_adapter.safetensors` |

These two you already have from native Anima — leave them where they are:

```text
models/text_encoder/qwen_3_06b_base.safetensors
models/VAE/qwen_image_vae.safetensors
```

All paths are relative to your `sd-webui-forge-neo` folder. If you also run
ComfyUI, hard-link the big files instead of copying them:

```bash
mklink /H "models\Stable-diffusion\Anima-3.8B.safetensors" "C:\path\to\ComfyUI\models\diffusion_models\Anima-3.8B.safetensors"
```

Restart Forge after copying the files.

## Use

1. Select the `Anima-3.8B` checkpoint.
2. In Forge's VAE / Text Encoder control, select `qwen_image_vae` and
   `qwen_3_06b_base`, exactly as you would for native Anima.
3. Expand **Anima 3.8B (Qwen3.5)** and enable it.
4. Pick the adapter and set **Adapter strength**. `1.0` is the trained
   strength; `0.0` is plain native Anima.
5. Optionally tick **Use adapter on negative prompt** to route the negative
   through Qwen3.5 as well. A separate **Negative adapter strength** slider
   appears; `0.0` there keeps the negative on native Anima.

The positive prompt is encoded by both text encoders and fused by the selected
adapter. The negative prompt stays on native Anima — the reference workflow's
behaviour — unless you enable the negative switch, in which case it takes the
same path at its own strength.

Everything is restored after each generation, so disabling the accordion gives
you stock Anima again without a restart.

## Notes

- Adapters are discovered by safetensors metadata, not by filename. Supported
  architecture: `anima_progressive_qwen35_cross_adapter_v1`.
- The Qwen3.5 model is found automatically; its filename must contain
  `qwen35_4b`, `qwen3.5-4b`, or `qwen3_5_4b`.
- Qwen3.5 4B is loaded in addition to Anima's own encoder, so expect the extra
  VRAM/RAM cost of a 4B model while conditioning is computed.
- Adapter name, strength, and negative strength are written into the generation
  parameters of every image.
