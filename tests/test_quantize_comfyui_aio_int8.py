import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from convert_comfyui_aio import read_safetensors_header, sha256_file
from quantize_comfyui_aio_int8 import (
    configure_backend,
    convert,
    make_plan,
    quantize_rows,
    select_tensor,
)

DIT = "model.diffusion_model.layers.0.attention.to_q.weight"
MOE = "text_encoders.llada2.model.language_model.layers.1.mlp.experts.gate_proj"


@pytest.fixture
def source(tmp_path):
    torch.manual_seed(7)
    tensors = {
        DIT: torch.randn(16, 256).bfloat16(),
        MOE: torch.randn(2, 12, 256).bfloat16(),
        "vae.encoder.conv_in.weight": torch.randn(3, 3, 3, 3).bfloat16(),
        "text_encoders.tokenizer_json": torch.tensor(list(b'{"test":true}'), dtype=torch.uint8),
    }
    path = tmp_path / "source.safetensors"
    save_file(tensors, str(path), metadata={"config": "{}", "source_revision": "test"})
    return path, tensors


def test_plan_preserves_source_config_and_marks_experiment(source):
    path, _ = source
    header, _ = read_safetensors_header(path)
    _, output, _, size = make_plan(header, sha256_file(path))
    assert output["__metadata__"]["config"] == "{}"
    assert "experimental" in output["__metadata__"]["llada_quantization"]
    assert output[DIT]["dtype"] == "I8"
    assert output[MOE]["dtype"] == "I8"
    assert size < path.stat().st_size


@pytest.mark.parametrize("key,shape", [
    ("vae.encoder.conv_in.weight", [8, 256]),
    ("text_encoders.llada2.model.language_model.word_embeddings.weight", [512, 256]),
    ("model.diffusion_model.layers.0.adaLN_modulation.0.weight", [512, 256]),
    ("text_encoders.llada2.model.language_model.layers.1.mlp.gate.weight", [256, 256]),
    (DIT, [16, 250]),
])
def test_sensitive_or_ineligible_tensors_are_not_quantized(key, shape):
    assert select_tensor(key, {"shape": shape, "dtype": "BF16"}) is None


def test_streaming_conversion_and_real_core_ops(source, tmp_path):
    from comfy.cli_args import args
    args.cpu = True
    import comfy.ops
    from comfy_kitchen.tensor import QuantizedTensor

    path, original = source
    output = tmp_path / "int8.safetensors"
    result = convert(path, output, sha256_file(path), device="cpu", rows=5)
    assert result["output_sha256"] == sha256_file(output)
    assert result["quantized_expert_banks"] == result["quantized_dit_layers"] == 1
    assert max(layer["relative_l2"] for layer in result["layers"]) < .02
    assert not Path(str(output) + ".partial").exists()
    with safe_open(str(output), framework="pt") as f:
        for key in original:
            if key not in (DIT, MOE):
                assert torch.equal(f.get_tensor(key), original[key])
        ops = comfy.ops.mixed_precision_ops({}, torch.bfloat16, full_precision_mm=True)
        linear = ops.Linear(256, 16, bias=False, device="cpu")
        base = DIT[:-7]
        linear.load_state_dict({suffix: f.get_tensor(base + "." + suffix)
                                for suffix in ("weight", "weight_scale", "comfy_quant")})
        bank = ops.MoEExperts(2, 256, 12, bias=False, device="cpu")
        bank.load_state_dict({"weight": f.get_tensor(MOE), **{
            suffix: f.get_tensor(MOE + "." + suffix) for suffix in ("weight_scale", "comfy_quant")}})
        assert isinstance(linear.weight, QuantizedTensor)
        assert linear.weight._params.convrot
        assert isinstance(bank.weight, QuantizedTensor)
        assert not bank.weight._params.convrot
        x = torch.randn(4, 256).bfloat16()
        for actual, reference in [
            (linear(x), x @ original[DIT].t()),
            (bank.expert_linear(x, 0), x @ original[MOE][0].t()),
        ]:
            assert torch.isfinite(actual).all()
            assert ((actual.float() - reference.float()).norm() / reference.float().norm()).item() < .025
        assert bank.weight.dequantize().shape == original[MOE].shape
    assert json.loads(Path(str(output) + ".manifest.json").read_text())["output_sha256"] == sha256_file(output)


def test_wrong_source_hash_does_not_write_payload(source, tmp_path):
    path, _ = source
    output = tmp_path / "wrong.safetensors"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        convert(path, output, "0" * 64, device="cpu")
    assert not output.exists()
    assert not Path(str(output) + ".partial").exists()


def test_never_overwrites_source_or_output(source, tmp_path):
    path, _ = source
    digest = sha256_file(path)
    with pytest.raises(ValueError, match="must differ"):
        convert(path, path, digest)
    existing = tmp_path / "existing.safetensors"
    save_file({"sentinel": torch.tensor([17])}, str(existing))
    original_output_hash = sha256_file(existing)
    with pytest.raises(FileExistsError):
        convert(path, existing, digest)
    assert sha256_file(path) == digest
    assert sha256_file(existing) == original_output_hash


@pytest.mark.parametrize("mode", ["convrot", "rowwise"])
def test_zero_rows_are_finite(mode):
    q, scale, error, magnitude = quantize_rows(torch.zeros(2, 256), mode, "cpu")
    assert not q.any()
    assert torch.isfinite(scale).all() and (scale > 0).all()
    assert error == magnitude == 0


def test_nonfinite_weights_rejected():
    with pytest.raises(ValueError, match="Non-finite"):
        quantize_rows(torch.full((2, 256), float("nan")), "convrot", "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["convrot", "rowwise"])
def test_real_gpu_backend_roundtrip(mode):
    configure_backend()
    q, scale, error, magnitude = quantize_rows(torch.randn(16, 256).bfloat16(), mode, "cuda")
    assert q.dtype == torch.int8 and scale.dtype == torch.float32
    assert (error / magnitude) ** .5 < .02
