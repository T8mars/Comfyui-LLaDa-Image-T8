# Native Core review evidence

Supporting artifacts for [ComfyUI PR #16095](https://github.com/Comfy-Org/ComfyUI/pull/16095).
These workflows require the **native Core PR branch**, not this repository's
standalone T8 nodes. Keeping evidence here does not add example media or conversion
tools to the Core runtime diff. This is not a claim that Core has merged support.

## Run the native workflows

Check out `T8mars/ComfyUI` branch `llada-image-core`, install Core's requirements,
and place the [Base/Turbo AIO checkpoints](https://huggingface.co/t8star/LLaDa-Image-Comfy/tree/main)
in `models/checkpoints/`. Open the UI JSON below and use `CheckpointLoaderSimple`.
No custom nodes are required. These are unchanged frontend exports whose bytes
match the six previously executed native workflows; no API prompt JSON is shipped here.
The BF16 frontend evidence below was recorded at `648a8e6796151b1253072e22f4b5d4b45839b62a`.
The INT8 test-only follow-ups are `6125cc7` and `1f90ed9c588c9f9d470be6d8dad6f3813cac4f4f`; runtime code is unchanged.

| Mode | Base | Turbo |
| --- | --- | --- |
| Text | [UI workflow](workflows/base_text.json) | [UI workflow](workflows/turbo_text.json) |
| VQ | [UI workflow](workflows/base_vq.json) | [UI workflow](workflows/turbo_vq.json) |
| Editing | [UI workflow](workflows/base_editing.json) | [UI workflow](workflows/turbo_editing.json) |

The six workflows use seed 42, 1024 x 1024, empty negative prompt, Base 50 steps
with CFG 5 / Euler, or Turbo 4 steps with CFG 1 / its seeded stochastic sampler.
Text/VQ prompt: `A red fox sleeping beneath a glass tree.` Editing input is
[this PNG](../../example_workflows/inputs/llada_image_edit_source.png); upload it
through Load Image. Read the editing instruction directly in its workflow.

After merging upstream `15eb748`, Base and Turbo text workflows were imported
and run using the actual frontend with custom nodes disabled at `aa28d7a`.
Both outputs were pixel-exact against the prior native baseline:
[Base](images/base-text-current.png), [Turbo](images/turbo-text-current.png).
The later `648a8e6` change fixes test isolation only; production code is unchanged.
The local database was locked by another instance during this temporary server
run; generation succeeded, but this smoke run does not claim database QA.

## Experimental INT8 ConvRot

The converted Base/Turbo mixed INT8 AIO files are approximately 27.65 GB each.
252 DiT linears use Kitchen ConvRot INT8 (group size 256); 57 MoE expert banks
use unrotated rowwise INT8. All other tensor bytes remain unchanged. This does
not claim full-model ConvRot, BF16-equivalent quality, or an inference speedup.

- [Native Turbo text UI export](workflows/turbo_text_int8.json), executed through
  the frontend with `CheckpointLoaderSimple` on native Core `648a8e6`: success,
  1024 x 1024, 29.836 seconds. Pixel SHA-256:
  `f9c478c634bd76ca4615110238be4cc99d5104b759849da71c6daf59459ea89d`.
- This native output matches the standalone INT8 Turbo text output, **not BF16**.
  [Six standalone INT8 workflows and images](../INT8.md) and
  [portable execution evidence](../int8-evidence.json) are separate evidence;
  they do not establish six full-weight native INT8 workflow runs.
- Core commit `6125cc7` adds tiny Base/Turbo mixed-checkpoint tests through the
  standard checkpoint loader: stored INT8 weights/scales/metadata roundtrip
  exactly, and text/VQ/editing produce finite results. The focused pinned native
  suite passes **128 tests, no skips**. No new runtime quantization implementation
  or dependencies are needed.
- The first CI run exposed a test-only grad-mode transition on newer PyTorch
  (`Cannot set version_counter for inference tensor`). `1f90ed9` removes the
  partial inference-mode wrapper, keeping checkpoint loading and sampling in
  the same mode. The focused pinned suite still passes 128 tests; the detection
  file also passes all 16 tests on PyTorch 2.14 CPU. No runtime code or tolerance
  was changed. Check the PR for the current cross-platform CI status.
- The existing Core MoE expert-view path does not preserve ConvRot flags, and
  whole-bank ConvRot dequantization rejects 3D weights in the tested Kitchen
  version. Rotated expert banks are therefore explicitly excluded from these
  artifacts; this PR does not change the shared MoE kernel.

Quality remains experimental. The BF16 comparison and tolerance profile below
do not certify the quantized artifacts.

## Official-reference comparison (BF16)

The paired images below were generated independently from the pinned official
implementation and native Core `1596f90`, before this PR refresh. Their PNG hashes
were rechecked against the original inference reports. They are distinct from
the seed-42 workflow examples above.

- Source code: `inclusionAI/LLaDA-Image@b4dfa9a3e50d90d6718975ca1fa1b0edbc90d512`.
- Base weights: `e4e2703f410f7ddb6ee8d6b09dac6a8ec5093039`.
- Turbo weights: `f4afc52d925bbac4e22a1c947111fc1f127e37e5`.
- Seed 43; 1024 x 1024; empty negative prompt; Base 50 steps / CFG 5; Turbo 4 steps / CFG 1.
- Text prompt: `A ceramic teapot beside two oranges on a wooden table, soft morning light.`
- Editing instruction: `Render the geometric image as a soft pastel illustration.`
- Editing input: [procedural image](images/holdout-edit-input.png).

| Case | Official reference | Native Core | Image MAE / RMSE |
| --- | --- | --- | --- |
| Base text | ![reference](images/base-text-reference.png) | ![native](images/base-text-core.png) | 0.004106 / 0.010428 |
| Base editing | ![reference](images/base-editing-reference.png) | ![native](images/base-editing-core.png) | 0.001663 / 0.002319 |
| Turbo text | ![reference](images/turbo-text-reference.png) | ![native](images/turbo-text-core.png) | 0.015666 / 0.036815 |
| Turbo editing | ![reference](images/turbo-editing-reference.png) | ![native](images/turbo-editing-core.png) | 0.001247 / 0.001779 |

**Not bitwise parity with the official BF16 pipeline.** Core keeps its native
optimized math; complex versus matrix RoPE and floating-point operation ordering
produce differences. [The frozen acceptance profile](acceptance-profile.json)
is a project-specific regression policy, not an official ComfyUI tolerance standard.
Six calibration pipelines and these four independent holdouts met that policy.
Metadata, tensor keys/shapes/dtypes, discrete values, prompt/semantic features,
initial noise and sigmas remain exact; named floating stages have explicit bounds.
The original strict failures were retained rather than overwritten.

Twelve full-weight FP32 DiT probes (identical BF16-quantized inputs promoted to
FP32, positive branch, no CFG) measured max absolute error 0.000375510 and max
relative L2 0.0000124692. These are component checks, not full FP32 pipeline parity.
The Base editing reference itself is pale and has weak structure; reproducing it
does not establish good editing quality for all inputs.

## Tests and limits

- At `648a8e6`, upstream [Unit Tests](https://github.com/Comfy-Org/ComfyUI/actions/runs/34034252784)
  and [Execution Tests](https://github.com/Comfy-Org/ComfyUI/actions/runs/34034252809)
  passed on Linux, macOS and Windows, alongside lint, server-launch and policy checks.
- 126 focused native/reference and existing detection tests passed without skips
  after merging the latest Core, with Python 3.10.21, Torch 2.8.0+cu128,
  Transformers 4.57.6 / Diffusers 0.39.0 reference fixtures, and comfy-kitchen 0.2.33.
- Full CI initially exposed an existing quantization test mutating a class-shared
  weight-patch list. The same ordered tests reproduced the failure locally;
  using an instance-owned list fixes it without touching model math or tolerances.
- Full GPU evidence uses Windows, RTX 5090 Laptop 24 GiB and 64 GiB RAM.
  Six native workflows also passed a dynamic-VRAM +8 GiB headroom profile and
  unload/reload testing; that is not proof of operation on a physically smaller GPU.
- Reference-fixture tests skip in upstream CI when the optional pinned sources
  and weights are absent. Other devices, backend combinations and FP8/GGUF are
  not covered by this full-weight acceptance.
- A separate local Torch 2.14 / Transformers 5.16.1 reference check fails strict
  BF16 SigVQ equality; no cross-version equality claim or relaxed threshold is made.

[evidence.json](evidence.json) records image/workflow hashes, exact holdout settings,
decoded-image metrics and the two current frontend job IDs. Production runtime
remains offline and uses native Core model management; reference dependencies
are test-only.
