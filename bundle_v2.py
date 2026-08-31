from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


BUNDLE_ARCHITECTURE = "anima_3_8b_semantic_connector_v2_bundle"
BUNDLE_FORMAT = "1"
CONNECTOR_PREFIX = "net.anima_v2_connector."
V2_ARCHITECTURE = "anima_qwen35_quality_anchored_semantic_connector_v2"


def checkpoint_metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        return checkpoint.metadata() or {}


def native_adapter_hash(state: dict[str, torch.Tensor]) -> str:
    prefix = "net.llm_adapter."
    native = {
        name[len(prefix) :]: tensor
        for name, tensor in state.items()
        if name.startswith(prefix)
    }
    if not native:
        raise RuntimeError("The DiT checkpoint has no net.llm_adapter tensors.")
    digest = hashlib.sha256()
    for name, tensor in sorted(native.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def bundle(base_path: Path, adapter_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle: {output_path}")
    base_metadata = checkpoint_metadata(base_path)
    adapter_metadata = checkpoint_metadata(adapter_path)
    if adapter_metadata.get("architecture") != V2_ARCHITECTURE:
        raise RuntimeError("The adapter is not a Semantic Connector v2 checkpoint.")

    base = load_file(str(base_path), device="cpu")
    adapter = load_file(str(adapter_path), device="cpu")
    expected_hash = adapter_metadata.get("frozen_native_sha256")
    actual_hash = native_adapter_hash(base)
    if expected_hash and expected_hash != actual_hash:
        raise RuntimeError(
            "The adapter was trained against a different native Anima adapter: "
            f"expected {expected_hash}, got {actual_hash}."
        )

    combined = dict(base)
    for name, tensor in adapter.items():
        bundled_name = f"{CONNECTOR_PREFIX}{name}"
        if bundled_name in combined:
            raise RuntimeError(f"Duplicate bundled tensor: {bundled_name}")
        combined[bundled_name] = tensor

    metadata = dict(base_metadata)
    metadata.update(
        {
            "architecture": BUNDLE_ARCHITECTURE,
            "anima_v2_bundle_format": BUNDLE_FORMAT,
            "anima_v2_connector_prefix": CONNECTOR_PREFIX,
            "anima_v2_strength": "1.0",
            "anima_v2_base_filename": base_path.name,
            "anima_v2_adapter_filename": adapter_path.name,
            "anima_v2_native_sha256": actual_hash,
        }
    )
    for name, value in adapter_metadata.items():
        metadata[f"anima_v2_adapter_{name}"] = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    if temporary_path.exists():
        raise FileExistsError(f"Refusing to overwrite temporary file: {temporary_path}")
    try:
        save_file(combined, str(temporary_path), metadata=metadata)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bundle a matched Anima DiT and Semantic Connector v2 checkpoint."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle(args.base.resolve(), args.adapter.resolve(), args.output.resolve())
    sys.stdout.write(f"{args.output.resolve()}\n")


if __name__ == "__main__":
    main()
