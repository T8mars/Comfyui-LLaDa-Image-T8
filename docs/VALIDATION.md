# Standalone validation

## INT8 update (2026-09-06)

The current GitHub loader adds Core's native mixed-precision detection. Six
full-weight INT8 frontend workflows completed at 1024 x 1024; see
[INT8 scope and images](INT8.md) and [separate execution evidence](int8-evidence.json).
The INT8 node suite passed 126 tests; 14 streaming quantizer tests passed
separately, including GPU quantization. Both suites are now shipped together;
the combined run passed **140 tests with no skips** in 7.68 seconds.
INT8 is experimental, not pixel-exact to BF16 and not quality-approved.

The BF16 acceptance JSON below is a historical snapshot: its loader hash belongs
to the previous release and is deliberately not rewritten to certify a changed
file. This update does not publish a new Registry version; Registry publication
remains gated on fresh release acceptance. GitHub users can update directly.

## BF16 release validation (historical)

This package is tested against unmodified ComfyUI commit
`250b2e9551a7bc7a8ebb5beb07e0fecd2983e04a` (0.34.0).
The Core working tree remained clean; only this custom-node package was installed.
An archive built by official `comfy-cli 1.20.0` with `comfy node pack` was
also extracted into a separate `llada-image-t8` directory and loaded through
Core's real custom-node loader. All five node IDs registered successfully;
the global Core model registry was unchanged. The archive excluded `.qa/`,
tests, GitHub workflow files and model weights.

## Unit and reference checks

123 tests passed without skips in the pinned reference runtime on 2026-09-06:
Python 3.10.21, PyTorch 2.8.0+cu128 (CPU tests), Transformers 4.57.6,
Diffusers 0.39.0. Tests include tiny complete Base/Turbo AIO loading, text,
VQ and editing execution, exact tensor loading, native sampling formulas,
tokenization and official VAE comparison. Eleven converter regression tests and
two frontend/extension release-contract tests are included in this total.

The test runner must import PyTorch before adding the reference dependency
overlay to its module path. This is test-environment setup, not a model runtime
requirement. Run `pytest tests --confcutdir=tests` with `COMFYUI_PATH` set to
the clean ComfyUI directory. The manual installation directory is now
`custom_nodes/Comfyui-LLaDa-Image-T8`, matching the repository name. Tests create
a test-only `llada_image_t8` import alias pointing at the current checkout;
they do not require an installation folder with that alias. Core's actual
custom-node loader is also tested against the repository-named directory.

Reference tests use a sibling `LLaDA-Image` source checkout pinned to
`b4dfa9a3e50d90d6718975ca1fa1b0edbc90d512` and the following optional environment variables:

- `LLADA_IMAGE_TEXT_ENCODER_CODE`: pinned Base snapshot `text_encoder/` directory.
- `LLADA_IMAGE_TOKENIZER_JSON`: pinned Base snapshot `tokenizer/tokenizer.json`.
- `LLADA_IMAGE_VAE_WEIGHTS`: pinned Base snapshot `vae/diffusion_pytorch_model.safetensors`.

Fixture hashes are checked by tests. Do not treat a run with skipped reference
tests as the complete reference suite.

A different CPU environment (PyTorch 2.14 / Transformers 5.16.1) failed one
strict BF16 SigVQ comparison. The release does not claim numerical identity
across dependency versions or attention backends. No test threshold was changed
to make the fixed reference environment pass.

## Frontend acceptance

Documentation/gallery update on 2026-09-06: 124 tests passed without skips in
the same pinned runtime, including an additional real Core loader regression
for the repository-named installation directory. Production runtime files and
the six shipped workflow JSON files are unchanged from the original acceptance.

All six examples were imported, queued and saved through the actual ComfyUI
frontend. All six completed at 1024 x 1024 and matched the validated native Core
baseline pixel-for-pixel in the pinned environment. The shipped JSON files are
frontend Save results, not API prompts. See [acceptance.json](acceptance.json)
for job IDs, output pixel hashes and shipped workflow/runtime file hashes.

After those runs, loader reload factories gained `disable_dynamic` compatibility.
The final 123-test suite checks those factories, and a fresh-server frontend
reimport and execution of the shipped Turbo text JSON also passed with identical
pixels. Default inference mathematics was unchanged. The release action verifies
the acceptance file and its recorded hashes before publishing.
