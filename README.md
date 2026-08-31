# Anima 3.8B for Forge Neo

A Forge Neo extension that runs the [Anima 3.8B](https://huggingface.co/lylogummy/Anima-3.8B)
diffusion model with its **Qwen3.5 4B conditioning path**, including the
fixed-strength timestep-aware Semantic Connector v2 release.

Bundled v2 checkpoints activate automatically. The adapter dropdown and
strength controls remain available only as a backup for legacy v1 checkpoints.

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

For v2, place the combined checkpoint in Forge's checkpoint directory:

| Download this file | Put it here |
| --- | --- |
| `Anima-3.8B-v2.safetensors` | `models/Stable-diffusion/Anima-3.8B-v2.safetensors` |
| `text_encoders/qwen35_4b.safetensors` (4.8 GB) | `models/text_encoder/qwen35_4b.safetensors` |

The older separate v1 checkpoint and adapter remain supported:

| Download this file | Put it here |
| --- | --- |
| `difussion_models/Anima-3.8B.safetensors` | `models/Stable-diffusion/Anima-3.8B.safetensors` |
| `text_encoders/Anima-3.8B-expanded_adapter.safetensors` | `models/text_encoder/Anima-3.8B-expanded_adapter.safetensors` |

These two you already have from native Anima — leave them where they are:

```text
models/text_encoder/qwen_3_06b_base.safetensors
models/VAE/qwen_image_vae.safetensors
```

All paths are relative to your `sd-webui-forge-neo` folder. If you also run
ComfyUI, hard-link the big files instead of copying them:

```bash
mklink /H "models\Stable-diffusion\Anima-3.8B-v2.safetensors" "C:\path\to\ComfyUI\models\diffusion_models\Anima-3.8B-v2.safetensors"
```

Restart Forge after copying the files.

## Use

1. Select the `Anima-3.8B-v2` checkpoint.
2. In Forge's VAE / Text Encoder control, select `qwen_image_vae` and
   `qwen_3_06b_base`, exactly as you would for native Anima.
3. Generate normally. Forge detects the bundle metadata and activates v2 even
   when the extension accordion is collapsed.
4. Optionally expand **Anima 3.8B (Qwen3.5 / v2)** and enable
   **Use adapter on negative prompt**. V2 always uses its trained strength of
   `1.0`.

For a legacy v1 checkpoint, enable the accordion, select its separate adapter,
and choose the desired strength.

The positive prompt is encoded by both text encoders and fused by the selected
adapter. The negative prompt stays on native Anima — the reference workflow's
behaviour — unless you enable the negative switch, in which case it takes the
same path at its own strength.

Everything is restored after each generation, so disabling the accordion gives
you stock Anima again without a restart.

## Notes

- V2 bundles are discovered by safetensors metadata, not by filename.
- Legacy separate adapters use architecture
  `anima_progressive_qwen35_cross_adapter_v1`.
- The Qwen3.5 model is found automatically; its filename must contain
  `qwen35_4b`, `qwen3.5-4b`, or `qwen3_5_4b`.
- The native encoder and Qwen3.5 are run sequentially and released after their
  outputs are copied to system RAM. On lower-VRAM systems, Forge can offload
  text-encoder and connector weights and reload them when a prompt changes.
- Semantic Connector v2 remains active during every denoising step because its
  extraction is timestep-aware. Forge manages it as part of the sampling model
  and can stream/offload its layers when the full model does not fit.
- Adapter name, strength, and negative strength are written into the generation
  parameters of every image.

## Bundling a newer v2 epoch

The included `bundle_v2.py` validates that the new adapter was trained against
the native adapter inside the selected DiT, then writes a new combined
checkpoint. It deliberately refuses to overwrite an existing bundle:

```powershell
python bundle_v2.py `
  --base "path\to\Anima-base.safetensors" `
  --adapter "path\to\expanded_adapter_v2_epXX.safetensors" `
  --output "models\Stable-diffusion\Anima-3.8B-v2-epXX.safetensors"
```

Use a new output filename for every release epoch, refresh Forge's checkpoint
list, and select the new bundle. No code change is needed while the adapter
architecture and native-adapter hash remain compatible.
