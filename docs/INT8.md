# INT8 + ConvRot 混合量化实验

这是独立实验分支，**未发布到 Registry，也未更新 Core PR**。不是无损转换，不能替代已验收的 BF16 版本。

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

## 本机试用

1. 使用实验节点目录 `E:/Comfyui-LLaDa-Image-T8-INT8`，不要与正式节点重复安装。
2. 模型位于 `E:/LLaDA-Image/outputs/int8-convrot/`，可放入或通过 extra model paths 指向 `models/checkpoints`。
3. 对应六套前端工作流位于同目录的 `workflows/standalone_*_int8.json`。导入后使用 T8 AIO Loader，不是 API JSON。
4. 已另开隔离前端 `http://127.0.0.1:8192` 进行测试，原有 `8190` 实例未修改。

正式仓库的旧 Loader 不会自动支持 INT8；本实验 Loader 增加了与 Core 相同的量化识别和 mixed-precision 路径。

## 验证边界

- 转换前校验完整 BF16 源文件 SHA-256；转换后检查完整 safetensors 结构并计算输出 SHA-256。
- 转换器按小块处理权重，不将完整模型展开到 GPU；拒绝覆盖源文件、已有输出和已有 partial。
- 本机：Windows、RTX 5090 Laptop 24 GiB、64 GiB RAM、Torch 2.8.0+cu128、Comfy Kitchen 0.2.33。遵循 Core 的 CUDA 版本检查，使用 Kitchen GPU eager 后端；未安装或升级驱动。
- 转换器 14 项测试通过，包括真实 GPU 量化；实验节点 126 项测试通过，包括 Base/Turbo 小模型 INT8 加载、推理、重加载。
- 全模型推理记录、前端工作流与图片保存在实验目录中。尚未建立独立的 INT8 图像质量验收标准，不能套用 BF16 的验收结论。
- 首张 Turbo 文生图在原生 Core 路径成功，但与 BF16 不逐像素一致；PSNR 约 17.20 dB，仅为固定样例差异记录，不作为质量通过指标。
- FP8、GGUF、其他设备与后端不在本实验验收范围。

## 来源

- [官方 INT8 + ConvRot 转换工具](https://github.com/Comfy-Org/comfy-model-tools/blob/1846ff1a9c3212e12b0edf8347c493651854b80e/quant_int8_convrot.py)
- [Comfy Kitchen INT8 格式](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/tensor/int8.py)
- 本机流式转换器：`E:/LLaDA-Image/scripts/quantize_comfyui_aio_int8.py`

发布前还需人工检查更多提示词及编辑任务，确认质量损失可接受。此实验不承诺量化后与 BF16 等价。
