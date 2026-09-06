# 前端工作流验收原图

[返回项目说明](../../README.md)

以下 PNG 均由真实 ComfyUI 前端工作流执行产生，1024 × 1024，直接复制原文件，未修改像素或元数据。PNG 保留 ComfyUI 生成元数据；推荐导入配套 UI JSON，以获得发布版的前端布局。

| 原图 | 对应前端工作流 |
| --- | --- |
| [base_text.png](base_text.png) | [Base 文生图](../../example_workflows/base_text.json) |
| [base_vq.png](base_vq.png) | [Base VQ 生成](../../example_workflows/base_vq.json) |
| [base_editing.png](base_editing.png) | [Base 图片编辑](../../example_workflows/base_editing.json) |
| [turbo_text.png](turbo_text.png) | [Turbo 文生图](../../example_workflows/turbo_text.json) |
| [turbo_vq.png](turbo_vq.png) | [Turbo VQ 生成](../../example_workflows/turbo_vq.json) |
| [turbo_editing.png](turbo_editing.png) | [Turbo 图片编辑](../../example_workflows/turbo_editing.json) |
| [turbo_text_restart.png](turbo_text_restart.png) | 最终重启后再次导入并运行 Turbo 文生图 |

两个编辑范例使用同一张 [输入图](../../example_workflows/inputs/llada_image_edit_source.png)。

每张输出的 RGB 像素 SHA-256 对应 [acceptance.json](../acceptance.json) 的 `results`；重启复测对应 `fresh_restart_smoke`，与首次 Turbo 文生图像素一致。PNG 文件本身的 SHA-256 不等于 RGB 像素 SHA-256。
