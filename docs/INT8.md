# INT8 + ConvRot 混合量化实验

这是公开提供试用的混合量化实验版，模型在 [Hugging Face](https://huggingface.co/t8star/LLaDa-Image-Comfy/tree/main)，Loader 支持与工作流在本 GitHub 仓库。**旧 Registry 0.1.0 不含 INT8 支持，请先更新 GitHub 代码。** 不是无损转换，不能替代 BF16 的验收结论。

## 转换范围

| 部分 | 存储方式 |
| --- | --- |
| DiT attention / FFN，含三个 refiner，共 252 个线性层 | INT8 + ConvRot，分组 256，逐输出通道 scale |
| LLaDA2 MoE，共 57 个专家权重组 | INT8，逐专家、逐输出通道 scale，不旋转 |
| VAE、SigVQ、QueryFormer、其他文本层、路由器、归一化等 | 原始精度，字节不变 |
| tokenizer、组件配置 | 原样内嵌 |

Base / Turbo 仍各为一个 AIO safetensors 文件。它们约 **27.65 GB**，原 BF16 文件约 49.26 GB，磁盘体积减少约 **43.9%**。

ConvRot 使用 Comfy Kitchen 原生格式，不增加自定义量化算子。当前 Core 的 MoE 旋转标记传递及三维权重反量化不兼容，因此本实验明确使用非旋转 INT8 专家，不能称为“全组件 ConvRot”。

INT8 存储不等于所有运算都使用 INT8。文本编码器保留原生全精度矩阵乘法策略；压缩专家权重在计算时反量化。不要据文件大小推断显存减半或速度翻倍。

## 使用方法

1. 按 [README](../README.md) 安装节点；已安装用户进入 `custom_nodes/Comfyui-LLaDa-Image-T8` 执行 `git pull --ff-only`，不要重复安装两份。
2. 从抱脸下载 Base 或 Turbo 的 `INT8-ConvRot-Mixed-AIO.safetensors`，放入 `ComfyUI/models/checkpoints`，重启 ComfyUI。
3. 导入 [六套 INT8 前端工作流](../example_workflows/int8/)，选择相应模型后点击运行。这些是 UI JSON，不是 API JSON。
4. 编辑范例在 Load Image 中上传 [随附输入图](../example_workflows/inputs/llada_image_edit_source.png) 或自己的图片。

旧 Loader 不会自动支持 INT8；新版增加了与 Core 相同的量化识别和 mixed-precision 路径。原生 Core 使用方式及 PR 状态见 [Core review](core-review/README.md#experimental-int8-convrot)，不要在独立节点范例中替换为普通 Checkpoint Loader。

## 实际输出

以下均为通过 ComfyUI 前端运行得到的原始 1024 × 1024 PNG，未经后期修改。seed 42；Base 为 50 步、CFG 5、Euler，Turbo 为 4 步、CFG 1、专用随机采样器。完整参数以工作流为准。

| 功能 | Base INT8 | Turbo INT8 |
| --- | --- | --- |
| 文生图 | [![Base text](images/int8/base_text.png)](../example_workflows/int8/base_text.json) | [![Turbo text](images/int8/turbo_text.png)](../example_workflows/int8/turbo_text.json) |
| VQ 生成 | [![Base VQ](images/int8/base_vq.png)](../example_workflows/int8/base_vq.json) | [![Turbo VQ](images/int8/turbo_vq.png)](../example_workflows/int8/turbo_vq.json) |
| 图片编辑 | [![Base editing](images/int8/base_editing.png)](../example_workflows/int8/base_editing.json) | [![Turbo editing](images/int8/turbo_editing.png)](../example_workflows/int8/turbo_editing.json) |

## 验证边界

- 转换前校验完整 BF16 源文件 SHA-256；转换后检查完整 safetensors 结构并计算输出 SHA-256。
- 转换器按小块处理权重，不将完整模型展开到 GPU；拒绝覆盖源文件、已有输出和已有 partial。
- 本机：Windows、RTX 5090 Laptop 24 GiB、64 GiB RAM、Torch 2.8.0+cu128、Comfy Kitchen 0.2.33。遵循 Core 的 CUDA 版本检查，使用 Kitchen GPU eager 后端；未安装或升级驱动。
- 转换器 14 项测试通过，包括真实 GPU 量化；实验节点 126 项测试通过，包括 Base/Turbo 小模型 INT8 加载、推理、重加载。两部分合并后重跑 **140 项通过、无跳过**。
- 全模型推理记录、前端工作流与图片见 [INT8 实跑证据](int8-evidence.json)。尚未建立独立的 INT8 图像质量验收标准，不能套用 BF16 的验收结论。
- 首张 Turbo 文生图在原生 Core 路径成功，但与 BF16 不逐像素一致；PSNR 约 17.20 dB，仅为固定样例差异记录，不作为质量通过指标。
- 两个模型各有 1,130 个未量化张量（5.92 GB），已从源文件和量化文件逐字节比较，全部一致；原有 metadata 也保持一致。
- Turbo 和 Base 的文生图、VQ、编辑共六套工作流均已通过 ComfyUI 前端运行按钮完成全模型实跑，输出均为 1024 × 1024；六项成功、队列为空。这不等于图像质量已验收。
- FP8、GGUF、其他设备与后端不在本实验验收范围。

## 来源

- [官方 INT8 + ConvRot 转换工具](https://github.com/Comfy-Org/comfy-model-tools/blob/1846ff1a9c3212e12b0edf8347c493651854b80e/quant_int8_convrot.py)
- [Comfy Kitchen INT8 格式](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/tensor/int8.py)
- [流式转换器](../scripts/quantize_comfyui_aio_int8.py) / [测试](../tests/test_quantize_comfyui_aio_int8.py)

正式画质验收前还需人工检查更多提示词及编辑任务，确认质量损失可接受。此实验不承诺量化后与 BF16 等价。
