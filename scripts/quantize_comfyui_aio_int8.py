#!/usr/bin/env python3
"""Experimental LLaDA AIO: ConvRot INT8 DiT + unrotated INT8 expert banks.

Uses Comfy Kitchen's quantizer/format; does not patch ComfyUI runtime code.
Other tensors are copied byte-for-byte, including VAE, tokenizer and config.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import time
from pathlib import Path

from convert_comfyui_aio import (
    output_lock,
    read_safetensors_header,
    sha256_file,
    write_json_atomic,
)

PROFILE = "llada-int8-convrot-dit-rowwise-moe-v1"
DIT_LINEAR = re.compile(
    r"^model\.diffusion_model\.(?:layers|context_refiner|noise_refiner|sigvq_refiner)\.\d+\."
    r"(?:attention\.(?:to_q|to_k|to_v|to_out\.0)|feed_forward\.w[123])\.weight$"
)
EXPERT_BANK = re.compile(
    r"^text_encoders\.llada2\.model\.language_model\.layers\.\d+\.mlp\.experts\."
    r"(?:gate_proj|up_proj|down_proj)$"
)


def select_tensor(key, info):
    shape = info["shape"]
    if info["dtype"] != "BF16":
        return None
    if DIT_LINEAR.fullmatch(key) and len(shape) == 2 and shape[1] % 256 == 0:
        return "convrot"
    if EXPERT_BANK.fullmatch(key) and len(shape) == 3:
        return "rowwise"
    return None


def marker(key, shape, mode):
    base = key.removesuffix(".weight")
    config = {"format": "int8_tensorwise"}
    if mode == "convrot":
        config.update(convrot=True, convrot_groupsize=256)
    else:
        config["num_experts"] = shape[0]
    return base, json.dumps(config, sort_keys=True, separators=(",", ":")).encode()


def make_plan(header, source_sha256):
    metadata = dict(header.get("__metadata__", {}))
    if "config" not in metadata or not any(EXPERT_BANK.fullmatch(k) for k in header):
        raise ValueError("Expected an original BF16 LLaDA AIO with config and expert banks")
    if any(k.endswith((".comfy_quant", ".weight_scale")) for k in header):
        raise ValueError("Re-quantizing an already quantized checkpoint is not supported")
    metadata["llada_quantization"] = json.dumps({
        "profile": PROFILE, "source_sha256": source_sha256,
        "status": "experimental-not-quality-approved",
        "dit": "int8_convrot_g256", "moe": "int8_rowwise_no_rotation",
        "passthrough": "byte-identical", "compute_dtype": "bfloat16",
    }, sort_keys=True)
    output = {"__metadata__": metadata}
    plan = []
    offset = 0

    def add(key, dtype, shape, size):
        nonlocal offset
        if key in output:
            raise ValueError(f"Duplicate output key: {key}")
        output[key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size

    for key, info in header.items():
        if key == "__metadata__":
            continue
        mode = select_tensor(key, info)
        plan.append((key, mode))
        if mode:
            shape = info["shape"]
            base, conf = marker(key, shape, mode)
            add(key, "I8", shape, math.prod(shape))
            add(base + ".weight_scale", "F32", shape[:-1] + [1], math.prod(shape[:-1]) * 4)
            add(base + ".comfy_quant", "U8", [len(conf)], len(conf))
        else:
            add(key, info["dtype"], info["shape"], info["data_offsets"][1] - info["data_offsets"][0])
    if not any(mode == "convrot" for _, mode in plan):
        raise ValueError("No supported DiT linear weights found")
    encoded = json.dumps(output, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return plan, output, encoded, 8 + len(encoded) + offset


def quantize_rows(weight, mode, device):
    import torch
    from comfy_kitchen.tensor import TensorWiseINT8Layout

    wf = weight.to(device=device, dtype=torch.float32)
    if not torch.isfinite(wf).all():
        raise ValueError("Non-finite input weight")
    q, params = TensorWiseINT8Layout.quantize(
        wf, per_channel=True, convrot=mode == "convrot", convrot_groupsize=256,
    )
    restored = TensorWiseINT8Layout.dequantize(q, params)
    squared_error = (restored - wf).double().square().sum().item()
    squared_weight = wf.double().square().sum().item()
    return q.cpu(), params.scale.cpu(), squared_error, squared_weight


def configure_backend():
    import comfy_kitchen as kitchen
    import torch

    # Match Core's CUDA-version gate; keep Kitchen's GPU eager implementation
    # available on cu128 instead of loading its cu130 extension.
    if torch.version.cuda is None or tuple(map(int, torch.version.cuda.split("."))) < (13,):
        kitchen.registry.disable("cuda")


def convert(source, output, expected_sha256, device="cuda", rows=2048, dry_run=False):
    if rows < 1:
        raise ValueError("rows must be positive")
    if source.resolve() == output.resolve():
        raise ValueError("Source and output must differ")
    header, source_data_offset = read_safetensors_header(source)
    plan, output_header, encoded, output_size = make_plan(header, expected_sha256)
    summary = {
        "profile": PROFILE, "source_bytes": source.stat().st_size,
        "output_bytes": output_size, "quantized_dit_layers": sum(m == "convrot" for _, m in plan),
        "quantized_expert_banks": sum(m == "rowwise" for _, m in plan),
        "source_sha256": expected_sha256, "output": str(output),
    }
    print(json.dumps(summary), flush=True)
    if dry_run:
        return summary

    import importlib.metadata

    import torch
    from safetensors import safe_open

    configure_backend()
    partial = output.with_name(output.name + ".partial")
    manifest = output.with_name(output.name + ".manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        if any(p.exists() for p in (output, partial, manifest)):
            raise FileExistsError("Output/partial/manifest already exists; select a new output path")
        if shutil.disk_usage(output.parent).free < output_size + 4 * 1024**3:
            raise OSError("Insufficient disk space: need output size plus 4 GiB reserve")
        print("Verifying complete source SHA-256 before writing...", flush=True)
        if sha256_file(source) != expected_sha256:
            raise ValueError("Source SHA-256 mismatch")
        started = time.monotonic()
        report = []
        with safe_open(str(source), framework="pt", device="cpu") as tensors, \
                source.open("rb") as raw, partial.open("xb") as dst, torch.inference_mode():
            dst.write(struct.pack("<Q", len(encoded)))
            dst.write(encoded)
            for index, (key, mode) in enumerate(plan):
                shape = header[key]["shape"]
                if mode is None:
                    start, end = header[key]["data_offsets"]
                    raw.seek(source_data_offset + start)
                    remaining = end - start
                    while remaining:
                        chunk = raw.read(min(16 * 1024**2, remaining))
                        if not chunk:
                            raise EOFError(f"Unexpected EOF copying {key}")
                        dst.write(chunk)
                        remaining -= len(chunk)
                else:
                    view = tensors.get_slice(key)
                    scales = []
                    error = magnitude = 0.0
                    experts = range(shape[0]) if mode == "rowwise" else (None,)
                    for expert in experts:
                        count = shape[-2]
                        for row in range(0, count, rows):
                            block = view[row:row + rows] if expert is None else view[expert, row:row + rows, :]
                            q, scale, se, sw = quantize_rows(block, mode, device)
                            dst.write(q.contiguous().numpy().tobytes())
                            scales.append(scale)
                            error += se
                            magnitude += sw
                    for scale in scales:
                        dst.write(scale.contiguous().numpy().tobytes())
                    _, conf = marker(key, shape, mode)
                    dst.write(conf)
                    result = {"key": key, "mode": mode, "relative_l2": math.sqrt(error / max(magnitude, 1e-30))}
                    report.append(result)
                    print(json.dumps({"progress": f"{index + 1}/{len(plan)}", **result}), flush=True)
                expected_end = (output_header[marker(key, shape, mode)[0] + ".comfy_quant"]["data_offsets"][1]
                                if mode else output_header[key]["data_offsets"][1])
                if dst.tell() != 8 + len(encoded) + expected_end:
                    raise RuntimeError(f"Writer offset mismatch after {key}")
            dst.flush()
            os.fsync(dst.fileno())
        if partial.stat().st_size != output_size:
            raise RuntimeError("Final size mismatch")
        # Validate the full safetensors structure with the independent reader.
        with safe_open(str(partial), framework="pt", device="cpu") as result:
            if set(result.keys()) != set(output_header) - {"__metadata__"}:
                raise RuntimeError("Final key mismatch")
            for key in result.keys():  # noqa: SIM118 - safe_open is not an iterable dict
                view = result.get_slice(key)
                if view.get_shape() != output_header[key]["shape"] or view.get_dtype() != output_header[key]["dtype"]:
                    raise RuntimeError(f"Final tensor metadata mismatch: {key}")
        summary.update(output_sha256=sha256_file(partial), layers=report,
                       torch_version=torch.__version__, kitchen_version=importlib.metadata.version("comfy-kitchen"),
                       elapsed_seconds=time.monotonic() - started, status="converted-not-inference-validated")
        # Persist evidence before installing the verified artifact; never overwrite.
        write_json_atomic(manifest, summary)
        os.rename(partial, output)
        print(json.dumps({k: v for k, v in summary.items() if k != "layers"}), flush=True)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_sha256):
        parser.error("--source-sha256 must be 64 lowercase hex characters")
    convert(args.source, args.output, args.source_sha256, args.device, args.rows, args.dry_run)


if __name__ == "__main__":
    main()
