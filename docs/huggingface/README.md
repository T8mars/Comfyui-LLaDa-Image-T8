---
language:
  - en
  - zh
license: apache-2.0
base_model:
  - inclusionAI/LLaDA-Image
  - inclusionAI/LLaDA-Image-Turbo
pipeline_tag: text-to-image
tags:
  - comfyui
  - safetensors
  - image-generation
  - image-editing
  - aio
  - int8
---

# LLaDA-Image — ComfyUI AIO by T8

在 ComfyUI 中使用 LLaDA-Image Base / Turbo，支持**文生图、VQ 生成和图片编辑**。每个版本只需一个 AIO 单体文件，已内含文本编码器、VAE、tokenizer 等组件，无需再次转换。

[安装节点与说明](https://github.com/T8mars/Comfyui-LLaDa-Image-T8) · [原项目](https://github.com/inclusionAI/LLaDA-Image) · [前端工作流](https://github.com/T8mars/Comfyui-LLaDa-Image-T8/tree/main/example_workflows)

**模型是我们转换的，不是重新训练的。** inclusionAI 提供原模型与原始权重，T8 将其打包为 ComfyUI AIO，并进一步制作 INT8 混合量化版本。本仓库不是官方发布的 AIO，也不是可直接用 `DiffusionPipeline.from_pretrained` 加载的 Diffusers 目录。

## 1. 选择模型

只下载需要的一个文件即可，不必同时下载 BF16 与 INT8。

| 模型 | 大小 | 状态与下载 |
| --- | --- | --- |
| Base BF16 AIO | 49.26 GB | [下载](https://huggingface.co/t8star/LLaDa-Image-Comfy/resolve/main/LLaDA-Image-Base-BF16-AIO.safetensors?download=true)，保留原验收版本 |
| Turbo BF16 AIO | 49.26 GB | [下载](https://huggingface.co/t8star/LLaDa-Image-Comfy/resolve/main/LLaDA-Image-Turbo-BF16-AIO.safetensors?download=true)，保留原验收版本 |
| Base INT8 + ConvRot Mixed AIO | 27.65 GB | [下载](https://huggingface.co/t8star/LLaDa-Image-Comfy/resolve/main/LLaDA-Image-Base-INT8-ConvRot-Mixed-AIO.safetensors?download=true)，实验版 |
| Turbo INT8 + ConvRot Mixed AIO | 27.65 GB | [下载](https://huggingface.co/t8star/LLaDa-Image-Comfy/resolve/main/LLaDA-Image-Turbo-INT8-ConvRot-Mixed-AIO.safetensors?download=true)，实验版 |

INT8 文件约缩小 **43.9%**，但不是无损转换，也不表示显存减半或速度翻倍。它已完成六套前端工作流实跑，**尚未完成多提示词画质验收**。

## 2. 安装与运行

1. 按 [GitHub README](https://github.com/T8mars/Comfyui-LLaDa-Image-T8#readme) 安装节点，目录为 `ComfyUI/custom_nodes/Comfyui-LLaDa-Image-T8`。
2. **INT8 用户必须先更新 GitHub 节点代码**：在节点目录执行 `git pull --ff-only`。旧 Registry `0.1.0` 不含 INT8 Loader 支持，不要重复安装两份节点。
3. 将模型放到 `ComfyUI/models/checkpoints/`，重启 ComfyUI 并刷新浏览器。
4. 打开下方对应前端 JSON，在 **LLaDA-Image AIO Loader (T8)** 中选模型，点击运行。图片编辑还需在 Load Image 上传输入图。

要求 ComfyUI 0.34.0+、Python 3.10+；INT8 本次验证使用 Comfy Kitchen 0.2.33。独立节点不需要修改 Core，不要自行用普通 Checkpoint Loader 替换范例中的 T8 Loader。原生 Core 支持仍在 [PR #16095](https://github.com/Comfy-Org/ComfyUI/pull/16095) 中评审，不代表已经合并。

以下都是 **ComfyUI 前端 UI 工作流，不是 API JSON**：

| 功能 | Base BF16 | Turbo BF16 | Base INT8 | Turbo INT8 |
| --- | --- | --- | --- | --- |
| 文生图 | [打开](https://github.com/T8mars/Comfyui-LLaDa-Image-T8/blob/main/example_workflows/base_text.json) | [打开](https://github.com/T8mars/Comfyui-LLaDa-Image-T8/blob/main/example_workflows/turbo_text.json) | [打开](https://github.com/T8mars/Comfyui-LLaDa-Image-T8/blob/main/example_workflows/int8/base_text.json) | [打开](https://github.com/T8mars/Comfyui-LLaDa-Image-T8/blob/main/example_workflows/int8/turbo_text.json) |
| VQ 生成 | [打开](https://github.com/T8mars/Comfyui-LLaDa-Image-T8/blob/main/example_workflows/base_vq.json) | [打开](https://github.com/T8mars/Comfyui-LLaDa-Image-T8/blob/main/example_workflows/turbo_vq.json) | [打开](https://github.com/T8mars/Comfyui-LLaDa-Image-T8/blob/main/example_workflows/int8/base_vq.json) | [打开](https://github.com/T8mars/Comfyui-LLaDa-Image-T8/blob/main/example_workflows/int8/turbo_vq.json) |
| 图片编辑 | [打开](https://github.com/T8mars/Comfyui-LLaDa-Image-T8/blob/main/example_workflows/base_editing.json) | [打开](https://github.com/T8mars/Comfyui-LLaDa-Image-T8/blob/main/example_workflows/turbo_editing.json) | [打开](https://github.com/T8mars/Comfyui-LLaDa-Image-T8/blob/main/example_workflows/int8/base_editing.json) | [打开](https://github.com/T8mars/Comfyui-LLaDa-Image-T8/blob/main/example_workflows/int8/turbo_editing.json) |

默认 Base 50 步，Turbo 4 步。**4 步仅指扩散阶段**，VQ 的语义 token 生成仍较慢。

## 3. INT8 量化了什么

- DiT attention / FFN（含 refiner），共 252 个线性层：INT8 + ConvRot，分组 256。
- LLaDA2 MoE，共 57 个专家权重组：逐专家、逐输出通道 INT8，**不旋转**。
- VAE、SigVQ、QueryFormer、其他文本层、路由器、归一化等：原始精度，字节不变；配置和 tokenizer 原样保留。

当前测试的 Core / Kitchen MoE ConvRot 路径不兼容三维专家权重，因此本包明确采用混合策略，不是“全模型 ConvRot”。INT8 存储也不等于所有运算都用 INT8，专家权重计算时会反量化。转换使用 [开源流式转换器](https://github.com/T8mars/Comfyui-LLaDa-Image-T8/blob/main/scripts/quantize_comfyui_aio_int8.py)，遵循 Kitchen 原生格式。

## 4. 实际输出

以下均为未经后期修改的 1024 × 1024 前端输出。INT8 与 BF16 不逐像素一致，构图和纹理可能变化。

| 功能 | Base INT8 | Turbo INT8 |
| --- | --- | --- |
| 文生图 | ![Base INT8 text](images/int8/base_text.png) | ![Turbo INT8 text](images/int8/turbo_text.png) |
| VQ 生成 | ![Base INT8 VQ](images/int8/base_vq.png) | ![Turbo INT8 VQ](images/int8/turbo_vq.png) |
| 图片编辑 | ![Base INT8 editing](images/int8/base_editing.png) | ![Turbo INT8 editing](images/int8/turbo_editing.png) |

编辑使用的[输入图](images/edit_source.png)。[BF16 六张原始效果图与历史验收](https://github.com/T8mars/Comfyui-LLaDa-Image-T8#效果示例)保留不变；[INT8 六套实跑记录](https://github.com/T8mars/Comfyui-LLaDa-Image-T8/blob/main/docs/int8-evidence.json)独立记录，不借用 BF16 的画质结论。

本机验证环境：Windows、RTX 5090 Laptop 24 GiB 显存、64 GiB RAM、PyTorch 2.8.0+cu128。这不是最低配置保证；其他设备和后端未全面验证，FP8 / GGUF 不在本包验证范围。

## 5. 来源与校验

BF16 源版本固定为 Base `e4e2703f410f7ddb6ee8d6b09dac6a8ec5093039`、Turbo `f4afc52d925bbac4e22a1c947111fc1f127e37e5`。INT8 从本仓库对应 BF16 AIO 转换；两个模型各 1,130 个未量化张量已逐字节核对一致。

```text
0866b75effdc7598d88f4d5228af90fb2e38e7ed92c08708fba9f1598b8f01a2  LLaDA-Image-Base-BF16-AIO.safetensors
198d9729aaf57df4d785f7c7a88179cb7ea4cbe09fa09ef5830662c82a444d8b  LLaDA-Image-Turbo-BF16-AIO.safetensors
4766571e1fc6ac8bc16940e46b91083b1a9a7a00f42750165b852e07a95e36fc  LLaDA-Image-Base-INT8-ConvRot-Mixed-AIO.safetensors
57263ccd5e26acb67d50350d2bca7586dcdff4b5206beb17bd6b3cef25758853  LLaDA-Image-Turbo-INT8-ConvRot-Mixed-AIO.safetensors
```

使用 PowerShell `Get-FileHash 文件路径 -Algorithm SHA256` 或 Linux `sha256sum 文件路径` 核对；[SHA256SUMS.txt](SHA256SUMS.txt) 和每个模型相邻的 `.manifest.json` 一并提供。INT8 manifest 保留转换时状态，后续推理证据单独发布，不改写原始记录。

## 社媒与相关资源

- [B站](https://space.bilibili.com/385085361)
- [YouTube](https://www.youtube.com/@T8star-Aix/)
- [API](https://api.seedance.nz/sign-up?aff=5f4w)
- [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd)
- [Hugging Face 主页](https://huggingface.co/t8star)

部分链接含推广或邀请参数；这些外部服务不是运行节点所必需的依赖。

## License / attribution

原始模型归属 inclusionAI，模型权重遵循 [Apache-2.0](LICENSE)，详见 [NOTICE](NOTICE)。T8 提供 AIO 转换、量化与 ComfyUI 适配。节点代码的 GPL-3.0 许可证与模型权重许可证分开。

English: Community-converted Base/Turbo AIO checkpoints for ComfyUI. BF16 originals remain available; mixed INT8 ConvRot checkpoints are experimental, not lossless or quality-approved. Update the GitHub custom nodes before using INT8 and import the matching frontend workflow. No further conversion is needed.
