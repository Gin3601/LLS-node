# LLS Save Image MASK Output Design

## Context

当前项目里：

- `LLS Save Image` 已支持 `save` / `preview_only` 两种输出模式
- `LLS Simple Mask Preview` 单独负责把 `image + mask` 做彩色叠加预览

用户希望把遮罩的保存和预览能力收回到 `LLS Save Image`，并删除独立的 `LLS Simple Mask Preview` 节点。新的行为要求是：

- `LLS Save Image` 新增一个可选 `mask` 输入端口
- `image` 与 `mask` 的保存/预览互不干扰
- `mask` 以独立的黑白图形式保存或预览，不再依赖彩色叠加节点

## Goals

- 为 `LLS Save Image` 增加可选 `mask: MASK` 输入
- 保持现有 `image` 保存与预览逻辑兼容
- 在连接 `mask` 时，额外输出一份独立的黑白遮罩结果
- 删除 `LLS Simple Mask Preview` 节点及其注册、文档和测试引用
- 更新 README，使工作流改为直接使用 `LLS Save Image` 或 `Preview Image` 预览黑白遮罩

## Non-Goals

本次不实现：

- `image + mask` 彩色叠加预览
- 新增独立的 `LLS Save Mask` 节点
- 扩展新的遮罩颜色、叠加透明度或前端绘制交互
- 修改 `LLS Simple Mask Draw` 现有 `preview_image` 输出语义

## Chosen Architecture

沿用当前模块边界：

- `image/nodes.py`
  - 扩展 `LLSSaveImage` 的输入 schema
  - 增加 `MASK -> IMAGE` 的内部转换
  - 在保存/预览时分别处理 `image` 和 `mask`
- `mask/mask_utils.py`
  - 复用已有 `mask_to_image()`，避免在 `image` 模块重复实现遮罩转黑白图
- `mask/nodes.py`
  - 删除 `mask_preview` 注册导入
- `mask/mask_preview.py`
  - 删除文件

这样保留了现有职责：

- `image/` 负责输出节点
- `mask/` 负责遮罩数据处理

## Node Contract

### Registration

- class key: `LLSSaveImage`
- display name: `LLS Save Image`
- category: `LLS/Image`
- output node: `True`

### Inputs

Required:

- `image: IMAGE`
- `filename_prefix: STRING`
- `output_mode: save | preview_only`
- `save_metadata: BOOLEAN`

Optional:

- `mask: MASK`
- `prompt_info: STRING`
- `latent_info: STRING`
- `sample_info: STRING`
- `decode_info: STRING`
- `upscale_info: STRING`

Hidden:

- `prompt: PROMPT`
- `extra_pnginfo: EXTRA_PNGINFO`

### Outputs

保持与当前 `LLSSaveImage` 一致，不新增节点返回值。

## Runtime Behavior

### Shared Rules

- `image` 继续作为必填输入
- `mask` 为可选输入；未连接时，节点行为与当前版本完全一致
- `mask` 只会被转换为黑白三通道 `IMAGE` 用于输出，不会修改原始 `mask` 数据
- `mask` 输出文件名前缀固定为 `<filename_prefix>_mask`

### Save Mode

当 `output_mode = save` 时：

1. 先按当前逻辑保存 `image`
2. 如果连接了 `mask`，再将 `mask` 转为黑白 `IMAGE`
3. 使用同一个原生 `SaveImage` 节点再保存一次，前缀使用 `<filename_prefix>_mask`
4. `image` 保存继续携带 `lls_metadata`
5. `mask` 保存不附带 `lls_metadata`，避免把生成信息错误写入纯遮罩资产

### Preview Mode

当 `output_mode = preview_only` 时：

1. 先按当前逻辑预览 `image`
2. 如果连接了 `mask`，再将 `mask` 转为黑白 `IMAGE`
3. 使用原生 `PreviewImage` 再预览一次 `mask`
4. 两者各自生成预览条目，不做叠加，不互相覆盖

### UI Result Merging

`LLSSaveImage.save()` 需要兼容 ComfyUI 原生 `save_images()` 返回的 UI 结构。

实现上应将 `image` 和 `mask` 两次调用返回的 `ui.images` 合并为一个结果，保证：

- 用户能在同一个节点输出中同时看到原图和遮罩图
- 原图仍保持当前排序在前，遮罩图追加在后
- 没有 `mask` 时返回结构保持不变

## Edge Cases

- `mask` 为 `None`：按现有逻辑只处理 `image`
- `mask` 尺寸与 `image` 不一致：允许，直接按 `mask_to_image()` 的原始分辨率输出黑白图
- `mask` 形状非法：沿用 `mask_to_image()` 的报错行为
- `save_metadata = False`：`image` 不写 metadata，`mask` 同样不写 metadata
- 原生 `SaveImage` / `PreviewImage` 不可用：沿用当前错误提示

## README Changes

README 需要同步以下调整：

- 从修复工作流节点列表中删除 `LLS Simple Mask Preview`
- 把相关示例连接从 `Load Image.image + mask -> LLS Simple Mask Preview` 改为：
  - `mask -> LLS Save Image.mask`
  - 或 `LLS Simple Mask Create.mask_image -> Preview Image`
- 删除 `LLS Simple Mask Preview` 专节
- 说明 `LLS Save Image` 现在可同时保存/预览原图与黑白遮罩图

## Testing Strategy

测试按 TDD 顺序补齐：

1. `LLSSaveImage.INPUT_TYPES()` 新增可选 `mask`
2. `preview_only` 模式下，连接 `mask` 时应触发两次 `PreviewImage`
3. `save` 模式下，连接 `mask` 时应触发两次 `SaveImage`
4. `mask` 输出使用 `<filename_prefix>_mask`
5. `mask` 结果是黑白三通道图像
6. 插件注册中不再包含 `LLSSimpleMaskPreview`
7. README 不再引用 `LLS Simple Mask Preview`

## Open Decisions Resolved

以下语义已经固定，不在实现阶段重新讨论：

- 采用方案 1：扩展现有 `LLS Save Image`
- `image` 与 `mask` 同时输出，而不是二选一
- `mask` 以独立黑白图输出，而不是彩色叠加图
- 删除 `LLS Simple Mask Preview`，不保留兼容壳节点
