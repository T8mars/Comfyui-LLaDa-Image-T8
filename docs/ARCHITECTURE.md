# Native pipeline, standalone ownership

The independent package is an installation boundary, not a second inference
engine. It does not register a model in `comfy.supported_models.models`, replace
Core's detector, patch `CheckpointLoaderSimple`, or use a Diffusers pipeline.

| Responsibility | Package owner | Existing ComfyUI interface |
| --- | --- | --- |
| Identify Base/Turbo AIO and component configs | `runtime/detection.py`, `runtime/config.py` | `supported_models_base.BASE`, `ClipTarget` |
| Load the one local safetensors file | `runtime/loader.py` | `utils.load_torch_file`, `CoreModelPatcher`, `sd.CLIP`, `sd.VAE` |
| DiT and flow conditioning | `runtime/model.py`, `runtime/model_base.py` | `BaseModel`, native ops, selected attention, `CONDRegular` |
| LLaDA2, QueryFormer, projection, SigVQ | `runtime/text_encoder.py`, `runtime/conditioning.py` | `SD1ClipModel`, tokenizer, native ops/model management |
| Base/Turbo schedule and Turbo update | `nodes.py` | `SIGMAS`, `SAMPLER`, `SamplerCustomAdvanced` |
| Text/VQ/edit examples | `example_workflows/*.json` | Native frontend graph serialization and standard image/latent nodes |

Checkpoint tensor namespaces are unchanged from the validated native port:

```text
model.diffusion_model.*
text_encoders.llada2.*
text_encoders.queryformer.*
text_encoders.text_projection.*
text_encoders.sigvq.*
text_encoders.tokenizer_json
vae.*
```

All components and tokenizer bytes are inside the same AIO file. Model load,
offload, caching and device policy belong to Core; the package has no persistent
tensor cache or autonomous download/install path. The package loader registers
reload factories with Core, including the `disable_dynamic` clone argument.

The primary architectural difference from a future native Core integration is
the package-owned AIO loader and `T8` node IDs. The transformer, conditioning,
token layouts and sampling math remain native. A future Core contribution can
move the owners into their native modules and replace the loader registration
without redesigning the inference pipeline. Existing standalone workflows must
retain their IDs or receive an explicit migration; do not silently redirect them.

Reference interface: unmodified ComfyUI commit
`250b2e9551a7bc7a8ebb5beb07e0fecd2983e04a`, particularly
`comfy/sd.py:load_state_dict_guess_config` and
`comfy/model_patcher.py:clone` / `deepclone_multigpu`.
This repository's publication does not update the separate upstream Core PR.
