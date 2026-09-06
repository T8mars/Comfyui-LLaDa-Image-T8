# LLaDA-Image T8

LLaDA-Image Base / Turbo 的独立 ComfyUI 节点包。使用单体 AIO safetensors，包含 DiT、LLaDA2 文本编码器、QueryFormer、TextProjection、SigVQ、VAE 和 tokenizer。

不需要修改 ComfyUI Core，不运行 Diffusers pipeline，不在推理时联网。MODEL / CLIP / VAE 的设备、卸载和显存管理由 ComfyUI 原生接口负责。本仓库不是已合并到 Core 的官方实现。

接口来源、代码归属和未来 Core 迁移边界见 [架构说明](docs/ARCHITECTURE.md)。

## 安装

要求 ComfyUI **0.34.0 或更新版本**、Python 3.10+。已验证的精确 Core 基线为 `250b2e9551a7bc7a8ebb5beb07e0fecd2983e04a`；更新版本仍可能引入接口变化。

手动安装，在 ComfyUI 根目录执行：

```bash
git clone https://github.com/T8mars/Comfyui-LLaDa-Image-T8 custom_nodes/llada_image_t8
python -m pip install -r custom_nodes/llada_image_t8/requirements.txt
```

使用 **ComfyUI 自己的 Python 环境**，然后重启后端并刷新浏览器。Windows 便携版请替换为其 `python_embeded/python.exe`。

Registry 发布目标：Publisher `t8star`，节点包 ID `llada-image-t8`。发布完成并审核可用后，可在 Manager 搜索 **LLaDA-Image T8**；GitHub 推送成功并不等于 Registry 已上架。

## 准备单体模型

本仓库不分发模型权重，也没有虚构的 AIO 下载链接。每个 BF16 AIO 约 **49.26 GB（45.88 GiB）**，Base 和 Turbo 各为一个文件。普通 Diffusers 分目录模型不能直接选进 AIO Loader。

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

## 前端工作流

[`example_workflows/`](example_workflows/) 中的 **六个 JSON 全部是可导入画布的 UI 工作流，不是 API prompt**。使用 ComfyUI 的“打开”（Ctrl+O）或拖入 JSON。工作流包含节点坐标、组件值、端口和完整连线。

| 工作流 | 功能 | 默认扩散步数 |
| --- | --- | --- |
| [base_text.json](example_workflows/base_text.json) | Base 文生图 | 50 |
| [base_vq.json](example_workflows/base_vq.json) | Base VQ 语义生成 | 50 |
| [base_editing.json](example_workflows/base_editing.json) | Base 图像编辑 | 50 |
| [turbo_text.json](example_workflows/turbo_text.json) | Turbo 文生图 | 4 |
| [turbo_vq.json](example_workflows/turbo_vq.json) | Turbo VQ 语义生成 | 4 |
| [turbo_editing.json](example_workflows/turbo_editing.json) | Turbo 图像编辑 | 4 |

1. 在 **LLaDA-Image AIO Loader (T8)** 选择对应 Base / Turbo 单体文件。
2. 编辑提示词；编辑工作流还需在 **Load Image** 上传自己的图，或上传 [`example_workflows/inputs/llada_image_edit_source.png`](example_workflows/inputs/llada_image_edit_source.png)。
3. 点击“运行”。结果通过标准 Save Image 保存。

首次导入若只看到部分画布，点击“适应视图”或按 `.`。

VQ 节点中的宽高应与空白 Flux2 latent 的宽高一致。VQ 图像宽高必须为 16 的倍数；编辑路径按原始算法对齐到 32 的倍数，尺寸不对齐时会重采样。Base 使用 Euler；Turbo 的随机插值采样器是独立节点，不等同于普通 Euler。Scheduler 的 `steps=0` 根据模型自动选择 Base 50 / Turbo 4。

节点 ID 统一使用 `T8` 前缀，避免与未来 Core 节点冲突。普通 `CheckpointLoaderSimple` 在未支持 LLaDA 的 Core 中无法识别本模型，请保留工作流中的专用 AIO Loader。

## 硬件与已知限制

- 全分辨率验收环境：Windows、RTX 5090 Laptop 24 GiB、64 GiB RAM、PyTorch 2.8.0+cu128。该记录不是其他显卡、操作系统或更低内存机器的性能保证。
- 模型很大，加载和 CPU/GPU 卸载可能耗时。Turbo 的 **4 步只指扩散阶段**，VQ 的 LLaDA2 离散 token 生成仍然很慢。
- 当前面向 BF16 AIO；FP32 的小模型接口测试通过，但不代表全尺寸 FP32 的硬件验收。FP8/GGUF/其他量化格式未验证。
- 部分独立参考验收用例中的 Base 编辑结果较淡、结构保持较弱；推理成功不代表所有输入的编辑质量都理想。
- 本地 BF16 对齐采用事先固定的非零误差范围，不宣称与上游所有浮点计算逐位一致。原始失败记录没有被放宽阈值覆盖。

## 开发与发布

```bash
python scripts/check_release.py
```

离线检查六个前端工作流的格式、节点与连线，并检查包内模型导入边界。单元测试使用 `COMFYUI_PATH` 指向干净 Core，安装目录名为 `custom_nodes/llada_image_t8`，执行 `pytest tests --confcutdir=tests`。部分精确对齐测试需要固定上游参考源码与权重，参见 [验证说明](docs/VALIDATION.md)。

发布遵循 [ComfyUI 官方 Registry 流程](https://docs.comfy.org/registry/publishing)，通过手动触发的 GitHub Action 使用仓库 Secret `REGISTRY_ACCESS_TOKEN`，不会因普通 push 自动上架。当前不向 upstream Core 提交或更新 PR。

## License / attribution

整体集成遵循 [GPL-3.0](LICENSE)；Apache-2.0 来源的模型代码保留各自版权声明，见 [NOTICE](NOTICE) 和 [Apache-2.0 license](licenses/Apache-2.0.txt)。模型权重遵循原发布者自己的许可。

English: Standalone LLaDA-Image Base/Turbo AIO support using native ComfyUI model management. Install into `custom_nodes/llada_image_t8`, prepare the pinned AIO checkpoint, and open one of the six **frontend workflow** JSON files. No Core patch or runtime network access is required. Weights are not included. VQ generation is substantially slower than the Turbo diffusion stage.
