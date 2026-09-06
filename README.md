# Comfyui-LLaDa-Image-T8

在 ComfyUI 中使用 **LLaDA-Image Base / Turbo**，支持文生图、VQ 语义生成和图片编辑。每个版本只需一个 **AIO 单体模型文件**，不需要修改 ComfyUI Core。

[下载 T8 AIO 单体模型](https://huggingface.co/t8star/LLaDa-Image-Comfy/tree/main) · [原项目 inclusionAI/LLaDA-Image](https://github.com/inclusionAI/LLaDA-Image) · [前端工作流](example_workflows/) · [Registry](https://registry.comfy.org/t8star/llada-image-t8)

**模型来源说明：**原模型与原始权重由 inclusionAI 发布；本仓库提供 T8 的 ComfyUI 适配与 AIO 转换工具。验收使用的单体模型是我们从官方权重转换打包的，不是官方直接发布的 AIO，也不是重新训练的模型。本仓库尚未合并到 ComfyUI Core。

## 1. 安装节点

要求 **ComfyUI 0.34.0+、Python 3.10+**。已验证环境：Windows、RTX 5090 Laptop 24 GiB 显存、64 GiB 内存、PyTorch 2.8.0+cu128；这不是最低配置保证。

### 普通安装

在 **ComfyUI 根目录**打开终端，使用运行 ComfyUI 的 Python 环境：

```bash
git clone https://github.com/T8mars/Comfyui-LLaDa-Image-T8.git custom_nodes/Comfyui-LLaDa-Image-T8
python -m pip install -r custom_nodes/Comfyui-LLaDa-Image-T8/requirements.txt
```

### Windows 便携版

在包含 `ComfyUI` 和 `python_embeded` 的**便携版根目录**执行：

```powershell
git clone https://github.com/T8mars/Comfyui-LLaDa-Image-T8.git ComfyUI/custom_nodes/Comfyui-LLaDa-Image-T8
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\Comfyui-LLaDa-Image-T8\requirements.txt
```

安装完成后，**重启 ComfyUI 后端并刷新浏览器**。

**关于名字：**手动安装目录现在统一为 `custom_nodes/Comfyui-LLaDa-Image-T8`，与 GitHub 仓库同名。旧文档的 `llada_image_t8` 只是人为指定的目录别名，并非另一个节点包。旧安装仍可使用；如需统一名字，关闭 ComfyUI 后重命名即可，**不要同时保留两份安装**。Manager 自动安装时可能采用 Registry ID `llada-image-t8` 作为目录名，这是正常的。

Manager 中的显示名为 **LLaDA-Image T8**，Publisher 为 `t8star`。截至 2026-09-06，`0.1.0` 已上传，但版本状态仍为 Pending；审核可用前请用上面的手动安装方式。

## 2. 准备模型

Base 和 Turbo 各使用一个 BF16 AIO 文件，**每个约 49.26 GB**。AIO 已包含文本编码器、VAE 和 tokenizer，无需分别加载。

**我们转换并验收的 AIO 模型托管在 [t8star/LLaDa-Image-Comfy](https://huggingface.co/t8star/LLaDa-Image-Comfy)。下载后直接使用，无需再次转换。** GitHub 只保存节点代码、转换工具、工作流和示例图片，模型权重放在 Hugging Face。

| 模型 | 下载 |
| --- | --- |
| Base BF16 AIO | [LLaDA-Image-Base-BF16-AIO.safetensors](https://huggingface.co/t8star/LLaDa-Image-Comfy/resolve/main/LLaDA-Image-Base-BF16-AIO.safetensors?download=true) |
| Turbo BF16 AIO | [LLaDA-Image-Turbo-BF16-AIO.safetensors](https://huggingface.co/t8star/LLaDa-Image-Comfy/resolve/main/LLaDA-Image-Turbo-BF16-AIO.safetensors?download=true) |

下载后将单体文件放入：

```text
ComfyUI/models/checkpoints/
├── LLaDA-Image-Base-BF16-AIO.safetensors
└── LLaDA-Image-Turbo-BF16-AIO.safetensors
```

只使用其中一个版本时，只需下载对应文件，并预留足够磁盘空间（单个约 49.26 GB）。命令行下载、SHA-256 校验及自行转换方法见 [模型指南](docs/MODELS.md)。只有自行下载原始权重再转换时，才建议单版本预留至少 **110 GB**。

## 3. 打开工作流并运行

以下六个 JSON **全部是前端 UI 工作流，不是 API 格式**。下载后拖进 ComfyUI，或按 **Ctrl+O** 打开。

| 功能 | Base（50 步） | Turbo（4 步） |
| --- | --- | --- |
| 文生图：输入提示词生成图片 | [base_text.json](example_workflows/base_text.json) | [turbo_text.json](example_workflows/turbo_text.json) |
| VQ 生成：先生成语义 token，再生成图片，无需输入图 | [base_vq.json](example_workflows/base_vq.json) | [turbo_vq.json](example_workflows/turbo_vq.json) |
| 图片编辑：输入图片和修改指令 | [base_editing.json](example_workflows/base_editing.json) | [turbo_editing.json](example_workflows/turbo_editing.json) |

1. 在 **LLaDA-Image AIO Loader (T8)** 中选择对应的 Base / Turbo 模型。
2. 修改提示词；编辑工作流还需要在 **Load Image** 中上传图片，可以用 [随附测试输入图](example_workflows/inputs/llada_image_edit_source.png)。
3. 点击 **运行**。结果会显示在 **Save Image** 节点中，并保存到 ComfyUI 的 `output` 目录。

看不到完整画布时，按 **`.`** 适应视图。请先用范例默认参数跑通，不要将专用 AIO Loader 换成普通 Checkpoint Loader。

## 效果示例

以下六张均为本仓库工作流在 **1024 × 1024** 下的真实验收输出，未经后期修改；点击可查看原图。每列使用对应版本模型，参数以链接中的工作流为准。

| 功能 | Base | Turbo |
| --- | --- | --- |
| 文生图 | [![Base 文生图](docs/images/base_text.png)](docs/images/base_text.png) | [![Turbo 文生图](docs/images/turbo_text.png)](docs/images/turbo_text.png) |
| VQ 生成 | [![Base VQ 生成](docs/images/base_vq.png)](docs/images/base_vq.png) | [![Turbo VQ 生成](docs/images/turbo_vq.png)](docs/images/turbo_vq.png) |
| 图片编辑 | [![Base 图片编辑](docs/images/base_editing.png)](docs/images/base_editing.png) | [![Turbo 图片编辑](docs/images/turbo_editing.png)](docs/images/turbo_editing.png) |

编辑前的共同输入图：

<img src="example_workflows/inputs/llada_image_edit_source.png" alt="两个图片编辑范例的原始输入图" width="320">

[最终重启复测原图](docs/images/turbo_text_restart.png)与 Turbo 文生图像素一致。原图与工作流的对应关系、校验说明见 [图片索引](docs/images/README.md)。

## 使用提醒

- **Turbo 的 4 步仅指扩散阶段。** VQ 的语义 token 生成仍然较慢，不代表整个流程只需几秒。
- 模型较大，加载、显存与内存之间的卸载需要时间；其他显卡、系统及低内存配置未做完整验收。
- VQ 节点与空白 latent 的宽高应一致，且为 16 的倍数；编辑尺寸会对齐到 32 的倍数。
- 当前主要验证 **BF16 AIO**，未验证 FP8 / GGUF。原项目提供其他权重格式，不代表本节点已经支持。
- 六套范例均执行成功，但编辑效果仍取决于输入和指令，不保证所有图片的编辑质量。

## 验证与来源

六套前端工作流已实跑并保存，输出与固定环境下的原生 Core 实现基线逐像素一致；当前完整测试 **124 项通过、无跳过**（含安装目录加载回归测试）。这不代表所有依赖版本或后端都逐位一致。完整环境、已知差异及运行记录见 [验证说明](docs/VALIDATION.md) 和 [验收记录](docs/acceptance.json)。

感谢 [inclusionAI/LLaDA-Image](https://github.com/inclusionAI/LLaDA-Image) 提供原模型与算法，以及 [ComfyUI](https://github.com/Comfy-Org/ComfyUI) 提供原生模型管理与节点接口。本适配不在推理时联网，不运行 Diffusers pipeline。代码接口和后续 Core 迁移设计见 [架构说明](docs/ARCHITECTURE.md)。

许可证：[GPL-3.0](LICENSE)。Apache-2.0 来源代码保留原声明，详见 [NOTICE](NOTICE)；模型权重遵循原发布者的许可。

English: Standalone ComfyUI nodes for LLaDA-Image Base/Turbo. Original weights are from inclusionAI; T8 provides the AIO conversion and ComfyUI integration. Install into `custom_nodes/Comfyui-LLaDa-Image-T8`, download a converted checkpoint from [Hugging Face](https://huggingface.co/t8star/LLaDa-Image-Comfy), place it in `models/checkpoints`, and open a frontend workflow above. No further conversion or Core patch is required.
