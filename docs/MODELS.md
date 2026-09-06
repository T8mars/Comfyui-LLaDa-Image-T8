# 模型下载与转换

[返回 README](../README.md)

原权重来自 inclusionAI。我们用本仓库工具转换出的 Base / Turbo BF16 AIO 托管在 [t8star/LLaDa-Image-Comfy](https://huggingface.co/t8star/LLaDa-Image-Comfy/tree/main)，与前端工作流验收使用的文件一致，不涉及重新训练。

## 直接下载（推荐）

在 **ComfyUI 根目录**执行，按需选择 Base 或 Turbo：

```bash
hf download t8star/LLaDa-Image-Comfy LLaDA-Image-Base-BF16-AIO.safetensors --local-dir models/checkpoints
hf download t8star/LLaDa-Image-Comfy LLaDA-Image-Turbo-BF16-AIO.safetensors --local-dir models/checkpoints
```

没有 `hf` 命令时可按 [官方 CLI 文档](https://huggingface.co/docs/huggingface_hub/guides/cli)安装，或直接在模型仓库页面下载。**下载的是转换完成的 AIO，无需再次转换，也无需另下文本编码器或 VAE。**

下载后的文件 SHA-256 应为：

```text
0866b75effdc7598d88f4d5228af90fb2e38e7ed92c08708fba9f1598b8f01a2  LLaDA-Image-Base-BF16-AIO.safetensors
198d9729aaf57df4d785f7c7a88179cb7ea4cbe09fa09ef5830662c82a444d8b  LLaDA-Image-Turbo-BF16-AIO.safetensors
```

Windows 使用 `Get-FileHash 文件路径 -Algorithm SHA256`，Linux 使用 `sha256sum 文件路径`。源文件与转换输出校验清单（`.manifest.json`）也随模型发布。

## INT8 下载与量化（实验）

INT8 混合量化下载（实验版，先更新 GitHub 节点，两个命令按需选一个）：

```bash
hf download t8star/LLaDa-Image-Comfy LLaDA-Image-Base-INT8-ConvRot-Mixed-AIO.safetensors --local-dir models/checkpoints
hf download t8star/LLaDa-Image-Comfy LLaDA-Image-Turbo-INT8-ConvRot-Mixed-AIO.safetensors --local-dir models/checkpoints
```

两个文件各约 27.65 GB；SHA-256：

```text
4766571e1fc6ac8bc16940e46b91083b1a9a7a00f42750165b852e07a95e36fc  LLaDA-Image-Base-INT8-ConvRot-Mixed-AIO.safetensors
57263ccd5e26acb67d50350d2bca7586dcdff4b5206beb17bd6b3cef25758853  LLaDA-Image-Turbo-INT8-ConvRot-Mixed-AIO.safetensors
```

自行量化已有 BF16 AIO 时，使用 ComfyUI 的 Python 环境（需 PyTorch、safetensors、Comfy Kitchen；本次验证 Kitchen 0.2.33）：

```bash
python scripts/quantize_comfyui_aio_int8.py /path/to/LLaDA-Image-Base-BF16-AIO.safetensors /path/to/LLaDA-Image-Base-INT8-ConvRot-Mixed-AIO.safetensors --source-sha256 0866b75effdc7598d88f4d5228af90fb2e38e7ed92c08708fba9f1598b8f01a2
```

默认使用 CUDA；`--dry-run` 仅检查转换计划。输出盘需至少约 32 GB 可用空间，源文件另算。转换拒绝覆盖已有输出或 partial，失败后不会自动续转。先读 [量化范围与限制](INT8.md)。

## 原始权重到 BF16 AIO（可选）

进入 `ComfyUI/custom_nodes/Comfyui-LLaDa-Image-T8` 后执行，并使用 ComfyUI 的 Python 环境。

每个 BF16 AIO 约 **49.26 GB（45.88 GiB）**，Base 和 Turbo 各为一个文件。下方下载的是官方原始分目录模型，不能直接选进 AIO Loader，需要运行转换脚本。

先按 [Hugging Face 官方 CLI 文档](https://huggingface.co/docs/huggingface_hub/guides/cli) 安装 `hf`，下载固定版本。以下命令在本节点仓库根目录执行，路径可以自行调整；下载和转换需要同时容纳原始文件与 AIO，单版本建议预留至少 **110 GB** 可用磁盘。

Base：

```bash
hf download inclusionAI/LLaDA-Image --revision e4e2703f410f7ddb6ee8d6b09dac6a8ec5093039 --local-dir models-source/base
python scripts/convert_comfyui_aio.py models-source/base LLaDA-Image-Base-BF16-AIO.safetensors --variant base --source-repo inclusionAI/LLaDA-Image --source-revision e4e2703f410f7ddb6ee8d6b09dac6a8ec5093039
```

Turbo：

```bash
hf download inclusionAI/LLaDA-Image-Turbo --revision f4afc52d925bbac4e22a1c947111fc1f127e37e5 --local-dir models-source/turbo
python scripts/convert_comfyui_aio.py models-source/turbo LLaDA-Image-Turbo-BF16-AIO.safetensors --variant turbo --source-repo inclusionAI/LLaDA-Image-Turbo --source-revision f4afc52d925bbac4e22a1c947111fc1f127e37e5
```

转换器校验固定源文件 SHA-256，流式打包并写出校验清单，不把全部权重装进内存。请保留生成的 `.manifest.json`。把完成的 `.safetensors` 放到 `ComfyUI/models/checkpoints/`，或通过 ComfyUI 的 `extra_model_paths.yaml` 配置 checkpoints 搜索路径。
