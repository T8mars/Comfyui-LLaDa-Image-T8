#!/usr/bin/env python3
"""Independently verify a completed LLaDA-Image ComfyUI AIO artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from convert_comfyui_aio import (
    canonical_json_sha256,
    read_safetensors_header,
    sha256_file,
)


def read_json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def require_equal(name: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def verify_artifact(checkpoint: Path, plan_path: Path, manifest_path: Path) -> dict:
    checkpoint = checkpoint.resolve()
    plan_path = plan_path.resolve()
    manifest_path = manifest_path.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    plan = read_json_object(plan_path)
    manifest = read_json_object(manifest_path)
    plan_sha256 = sha256_file(plan_path)

    for key in ("format_version", "variant", "source_repo", "source_revision"):
        require_equal(f"manifest.{key}", manifest.get(key), plan.get(key))
    require_equal("plan.output", checkpoint.name, plan.get("output"))
    require_equal("manifest.output", checkpoint.name, manifest.get("output"))
    require_equal("manifest.tensor_count", manifest.get("tensor_count"), plan.get("tensor_count"))

    locked_source = {
        "format_version": manifest.get("format_version"),
        "variant": manifest.get("variant"),
        "source_repo": manifest.get("source_repo"),
        "source_revision": manifest.get("source_revision"),
        "files": manifest.get("sources"),
    }
    source_lock_sha256 = canonical_json_sha256(locked_source)
    require_equal(
        "plan.source_lock_sha256",
        plan.get("source_lock_sha256"),
        source_lock_sha256,
    )

    checkpoint_size = checkpoint.stat().st_size
    require_equal("checkpoint size vs plan", checkpoint_size, plan.get("output_size"))
    require_equal(
        "checkpoint size vs manifest", checkpoint_size, manifest.get("output_size")
    )

    header, data_start = read_safetensors_header(checkpoint)
    require_equal("AIO header size", data_start, plan.get("output_header_size"))
    with checkpoint.open("rb") as handle:
        header_sha256 = hashlib.sha256(handle.read(data_start)).hexdigest()
    require_equal(
        "AIO header SHA-256", header_sha256, plan.get("aio_header_sha256")
    )

    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint __metadata__ must be an object")
    metadata_identity = {
        "llada_image.format_version": plan.get("format_version"),
        "llada_image.variant": plan.get("variant"),
        "llada_image.source_repo": plan.get("source_repo"),
        "llada_image.source_revision": plan.get("source_revision"),
        "llada_image.source_lock_sha256": source_lock_sha256,
    }
    for key, expected in metadata_identity.items():
        require_equal(f"metadata.{key}", metadata.get(key), expected)
    try:
        config = json.loads(metadata["config"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint metadata contains no valid config JSON") from exc
    if not isinstance(config, dict) or not isinstance(config.get("llada_image"), dict):
        raise TypeError("checkpoint config must contain a llada_image object")
    require_equal(
        "config.llada_image.variant",
        config["llada_image"].get("variant"),
        plan.get("variant"),
    )

    plan_tensors = plan.get("tensors")
    if not isinstance(plan_tensors, list):
        raise TypeError(f"{plan_path}: tensors must be a list")
    actual_keys = [key for key in header if key != "__metadata__"]
    expected_keys = [tensor.get("key") for tensor in plan_tensors]
    require_equal("checkpoint tensor key order", actual_keys, expected_keys)
    require_equal("checkpoint tensor count", len(actual_keys), plan.get("tensor_count"))
    for tensor in plan_tensors:
        key = tensor["key"]
        entry = header[key]
        require_equal(f"{key}.dtype", entry.get("dtype"), tensor.get("dtype"))
        require_equal(f"{key}.shape", entry.get("shape"), tensor.get("shape"))
        require_equal(
            f"{key}.data_offsets",
            entry.get("data_offsets"),
            tensor.get("output_data_offsets"),
        )
    tensor_data_size = checkpoint_size - data_start
    require_equal(
        "checkpoint tensor-data size",
        tensor_data_size,
        plan.get("output_tensor_data_size"),
    )

    output_sha256 = sha256_file(checkpoint)
    require_equal(
        "checkpoint SHA-256", output_sha256, manifest.get("output_sha256")
    )
    return {
        "checkpoint": str(checkpoint),
        "plan": str(plan_path),
        "plan_sha256": plan_sha256,
        "manifest": str(manifest_path),
        "variant": plan["variant"],
        "source_repo": plan["source_repo"],
        "source_revision": plan["source_revision"],
        "source_lock_sha256": source_lock_sha256,
        "output_size": checkpoint_size,
        "output_sha256": output_sha256,
        "aio_header_sha256": header_sha256,
        "tensor_count": len(actual_keys),
        "status": "verified",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = args.checkpoint.with_suffix(
            f"{args.checkpoint.suffix}.manifest.json"
        )
    report = verify_artifact(args.checkpoint, args.plan, manifest_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
