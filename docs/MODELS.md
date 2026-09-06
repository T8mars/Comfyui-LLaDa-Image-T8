# 模型下载与转换

[返回 README](../README.md)

原权重来自 inclusionAI。我们用本仓库工具转换出了验收使用的 Base / Turbo AIO，目前仅保存在本地，尚无公开 AIO 下载地址。以下步骤供其他用户复现转换，不涉及重新训练。

进入 `ComfyUI/custom_nodes/Comfyui-LLaDa-Image-T8` 后执行，并使用 ComfyUI 的 Python 环境。

本仓库目前不分发模型权重。每个 BF16 AIO 约 **49.26 GB（45.88 GiB）**，Base 和 Turbo 各为一个文件。普通 Diffusers 分目录模型不能直接选进 AIO Loader。

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
