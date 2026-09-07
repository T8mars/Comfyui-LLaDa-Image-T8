# Comfyui-LLaDa-Image-T8

[简体中文](README.md) | [English](README_EN.md)

> **Experimental mixed INT8 + ConvRot checkpoints are now available.** Base and Turbo are approximately 27.65 GB each, and all six frontend workflows have completed real runs. Update the GitHub custom-node checkout before using them: Registry release `0.1.0` does not include the INT8 loader support. Quantization is not lossless; see the [INT8 guide and output gallery](docs/INT8.md).

Use **LLaDA-Image Base and Turbo** in ComfyUI for text-to-image, VQ semantic generation, and image editing. Each variant uses one self-contained **AIO checkpoint** and does not require a ComfyUI Core modification.

[Download T8 AIO checkpoints](https://huggingface.co/t8star/LLaDa-Image-Comfy/tree/main) · [Original inclusionAI project](https://github.com/inclusionAI/LLaDA-Image) · [Frontend workflows](example_workflows/) · [Registry](https://registry.comfy.org/t8star/llada-image-t8)

**Model provenance:** inclusionAI publishes the original models and weights. This repository provides T8's ComfyUI integration and AIO conversion tools. The checkpoints used for validation were converted and packaged from the official weights; they are not official upstream AIO releases and were not retrained. Native support is under review and has not been merged into ComfyUI Core.

## 1. Install the nodes

Requirements: **ComfyUI 0.34.0+ and Python 3.10+**. The validated host used Windows, an RTX 5090 Laptop GPU with 24 GiB VRAM, 64 GiB RAM, and PyTorch 2.8.0+cu128. This is not a minimum hardware guarantee.

### Standard installation

Open a terminal in the **ComfyUI root directory** and use the Python environment that runs ComfyUI:

```bash
git clone https://github.com/T8mars/Comfyui-LLaDa-Image-T8.git custom_nodes/Comfyui-LLaDa-Image-T8
python -m pip install -r custom_nodes/Comfyui-LLaDa-Image-T8/requirements.txt
```

### Windows portable installation

Run the following in the portable root directory that contains `ComfyUI` and `python_embeded`:

```powershell
git clone https://github.com/T8mars/Comfyui-LLaDa-Image-T8.git ComfyUI/custom_nodes/Comfyui-LLaDa-Image-T8
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\Comfyui-LLaDa-Image-T8\requirements.txt
```

After installation, **restart the ComfyUI backend and refresh the browser**.

**About the directory name:** manual installations now use `custom_nodes/Comfyui-LLaDa-Image-T8`, matching the GitHub repository. The older `llada_image_t8` name was only a manually chosen directory alias, not a different package. Existing installations under that name still work. To standardize the name, stop ComfyUI before renaming it, and **do not keep two copies installed at the same time**. Manager may use the Registry ID `llada-image-t8` as the directory name; that is also expected.

The Manager display name is **LLaDA-Image T8**, and the Publisher ID is `t8star`. Registry version `0.1.0` is currently **Active** and can be installed through Manager; it supports BF16 only. Use the manual GitHub installation above for INT8 support.

## 2. Download a model

Base and Turbo each have a BF16 AIO checkpoint of approximately **49.26 GB**. Every AIO includes the text encoder, VAE, and tokenizer; no separate component download is required.

Experimental mixed INT8 AIO checkpoints are also available at approximately **27.65 GB**, a disk-size reduction of about **43.9%**. The BF16 files remain unchanged. INT8 execution has been validated, but broad prompt-quality acceptance has not been completed. File-size reduction does not imply proportional VRAM savings or speedup.

**T8-converted checkpoints are hosted at [t8star/LLaDa-Image-Comfy](https://huggingface.co/t8star/LLaDa-Image-Comfy). BF16 keeps its original validation record, while INT8 is clearly marked experimental. The downloaded file is ready to use and does not need to be converted again.** GitHub hosts the node code, conversion tools, workflows, and example images; model weights are hosted on Hugging Face.

| Model | Download |
| --- | --- |
| Base BF16 AIO | [LLaDA-Image-Base-BF16-AIO.safetensors](https://huggingface.co/t8star/LLaDa-Image-Comfy/resolve/main/LLaDA-Image-Base-BF16-AIO.safetensors?download=true) |
| Turbo BF16 AIO | [LLaDA-Image-Turbo-BF16-AIO.safetensors](https://huggingface.co/t8star/LLaDa-Image-Comfy/resolve/main/LLaDA-Image-Turbo-BF16-AIO.safetensors?download=true) |
| Base INT8 + ConvRot Mixed AIO (experimental) | [LLaDA-Image-Base-INT8-ConvRot-Mixed-AIO.safetensors](https://huggingface.co/t8star/LLaDa-Image-Comfy/resolve/main/LLaDA-Image-Base-INT8-ConvRot-Mixed-AIO.safetensors?download=true) |
| Turbo INT8 + ConvRot Mixed AIO (experimental) | [LLaDA-Image-Turbo-INT8-ConvRot-Mixed-AIO.safetensors](https://huggingface.co/t8star/LLaDa-Image-Comfy/resolve/main/LLaDA-Image-Turbo-INT8-ConvRot-Mixed-AIO.safetensors?download=true) |

Place the selected AIO file in:

```text
ComfyUI/models/checkpoints/
```

Download only the file you plan to use; BF16 and INT8 are alternatives, not a combined installation. See the [model guide](docs/MODELS.md) for CLI downloads, SHA-256 values, and conversion instructions. If converting the original weights yourself, reserve at least **110 GB** for one variant.

## 3. Open and run a workflow

BF16 and INT8 each have six examples. **Every JSON file below is a ComfyUI frontend workflow, not an API prompt.** Drag it into ComfyUI or press **Ctrl+O**. The original BF16 examples are:

| Function | Base (50 steps) | Turbo (4 steps) |
| --- | --- | --- |
| Text-to-image | [base_text.json](example_workflows/base_text.json) | [turbo_text.json](example_workflows/turbo_text.json) |
| VQ generation: generate semantic tokens, then an image, without an input image | [base_vq.json](example_workflows/base_vq.json) | [turbo_vq.json](example_workflows/turbo_vq.json) |
| Image editing | [base_editing.json](example_workflows/base_editing.json) | [turbo_editing.json](example_workflows/turbo_editing.json) |

Experimental mixed INT8 workflows require the latest GitHub checkout. Run `git pull --ff-only` in the node directory, then restart ComfyUI:

| Function | Base INT8 | Turbo INT8 |
| --- | --- | --- |
| Text-to-image | [Frontend workflow](example_workflows/int8/base_text.json) | [Frontend workflow](example_workflows/int8/turbo_text.json) |
| VQ generation | [Frontend workflow](example_workflows/int8/base_vq.json) | [Frontend workflow](example_workflows/int8/turbo_vq.json) |
| Image editing | [Frontend workflow](example_workflows/int8/base_editing.json) | [Frontend workflow](example_workflows/int8/turbo_editing.json) |

1. Select the matching Base or Turbo checkpoint in **LLaDA-Image AIO Loader (T8)**.
2. Edit the prompt. Editing workflows also require an image in **Load Image**; you can use the [included test image](example_workflows/inputs/llada_image_edit_source.png).
3. Click **Run**. The result appears in **Save Image** and is written to ComfyUI's `output` directory.

Press **`.`** to fit the whole canvas if necessary. Start with the example's default settings, and do not replace the dedicated AIO loader with a regular checkpoint loader.

## Output examples

These are the original, unedited **1024 × 1024 BF16** validation outputs. Click an image to open the full-size file. The six INT8 outputs are kept in the separate [INT8 gallery](docs/INT8.md#实际输出), and do not inherit the BF16 quality conclusion.

| Function | Base | Turbo |
| --- | --- | --- |
| Text-to-image | [![Base text-to-image](docs/images/base_text.png)](docs/images/base_text.png) | [![Turbo text-to-image](docs/images/turbo_text.png)](docs/images/turbo_text.png) |
| VQ generation | [![Base VQ](docs/images/base_vq.png)](docs/images/base_vq.png) | [![Turbo VQ](docs/images/turbo_vq.png)](docs/images/turbo_vq.png) |
| Image editing | [![Base editing](docs/images/base_editing.png)](docs/images/base_editing.png) | [![Turbo editing](docs/images/turbo_editing.png)](docs/images/turbo_editing.png) |

Shared editing input:

<img src="example_workflows/inputs/llada_image_edit_source.png" alt="Shared input for the two editing examples" width="320">

The [final restart verification output](docs/images/turbo_text_restart.png) is pixel-identical to the Turbo text-to-image example. See the [image index](docs/images/README.md) for image-to-workflow mapping and checksums.

## Important notes

- **Turbo's four steps apply only to diffusion.** VQ semantic-token generation is still slow; the entire workflow does not finish in four quick operations.
- The checkpoints are large. Loading and transfers between VRAM and system memory can take time. Other GPUs, operating systems, and low-memory configurations have not received complete full-weight validation.
- VQ conditioning dimensions must match the empty latent dimensions and be divisible by 16. Editing dimensions are aligned to multiples of 32.
- **BF16 AIO** retains its original validation record. INT8 is an experimental mixed-quantization release with six completed workflow runs: the DiT uses ConvRot, MoE experts do not, and other components remain at their original precision. FP8 and GGUF are not validated.
- All six examples completed successfully, but editing quality still depends on the source image and instruction. Success does not guarantee quality for every image.

## Social links and related resources

- [Bilibili](https://space.bilibili.com/385085361)
- [YouTube](https://www.youtube.com/@T8star-Aix/)
- [API](https://api.seedance.nz/sign-up?aff=5f4w)
- [Online AI applications](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [ComfyUI package](https://pan.quark.cn/s/264edb7e36bd)
- [Hugging Face profile](https://huggingface.co/t8star)

Some links contain referral or invitation parameters. These external services are not required to run the nodes.

## Validation and attribution

The combined suite passes **140 tests with no skips**, including node behavior, frontend graph validation, and quantizer regression tests. The six BF16 frontend outputs are pixel-identical to the native Core baseline in the pinned environment, and their historical acceptance record is unchanged. Six INT8 runs are recorded separately in the [INT8 validation guide](docs/INT8.md); INT8 outputs are not pixel-identical to BF16. See the [validation notes](docs/VALIDATION.md) and [BF16 acceptance record](docs/acceptance.json) for the complete environment and historical evidence.

Thanks to [inclusionAI/LLaDA-Image](https://github.com/inclusionAI/LLaDA-Image) for the original models and algorithms, and to [ComfyUI](https://github.com/Comfy-Org/ComfyUI) for native model management and node interfaces. The integration performs no network requests during inference and does not run a Diffusers pipeline. See the [architecture guide](docs/ARCHITECTURE.md) for interface and Core-migration details.

License: [GPL-3.0](LICENSE). Apache-2.0 source attribution is retained in [NOTICE](NOTICE); model weights remain subject to the original publisher's license.
