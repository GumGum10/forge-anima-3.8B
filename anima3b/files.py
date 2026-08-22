from __future__ import annotations

import os
from pathlib import Path

from safetensors import SafetensorError, safe_open

ARCHITECTURE = "anima_progressive_qwen35_cross_adapter_v1"


def forge_root() -> Path:
    return Path(__file__).resolve().parents[3]


def text_encoder_roots() -> list[Path]:
    roots = [forge_root() / "models" / "text_encoder"]
    try:
        from modules_forge.main_entry import module_list

        roots.extend(Path(path).parent for path in module_list.values())
    except Exception:
        pass
    return list(dict.fromkeys(path.resolve() for path in roots if path.is_dir()))


def qwen35_models() -> dict[str, str]:
    found: dict[str, str] = {}
    markers = ("qwen35_4b", "qwen3.5-4b", "qwen3_5_4b")
    for root in text_encoder_roots():
        for path in root.rglob("*.safetensors"):
            if any(marker in path.name.lower() for marker in markers):
                found.setdefault(path.name, str(path))
    return dict(sorted(found.items()))


def adapters() -> dict[str, str]:
    found: dict[str, str] = {}
    for root in text_encoder_roots():
        for path in root.rglob("*.safetensors"):
            try:
                with safe_open(path, framework="pt", device="cpu") as checkpoint:
                    metadata = checkpoint.metadata() or {}
                    keys = set(checkpoint.keys())
                if metadata.get("architecture") != ARCHITECTURE:
                    continue
                if any(key.startswith(("timestep_gates.", "anchor_deviation")) for key in keys):
                    continue
            except (OSError, ValueError, SafetensorError):
                continue
            label = os.path.relpath(path, root).replace("\\", "/")
            found.setdefault(label, str(path))
    return dict(sorted(found.items()))


def tokenizer_dir() -> Path:
    bundled = Path(__file__).resolve().parents[1] / "qwen35_tokenizer"
    candidates = [bundled]
    candidates.extend(root / "qwen35_tokenizer" for root in text_encoder_roots())
    for candidate in candidates:
        if (candidate / "tokenizer.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Qwen3.5 tokenizer files are missing from the extension's "
        "qwen35_tokenizer directory. Reinstall the extension."
    )
