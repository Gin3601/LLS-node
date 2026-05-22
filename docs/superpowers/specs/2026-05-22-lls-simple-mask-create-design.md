# LLS Simple Mask Create Design

## Context

`LLS-node` 已经提供：

- `LLS Simple Mask Draw`
- `LLS Simple Repair Prepare`
- `LLS Simple KSampler`
- `LLS Simple Repair Finish`

当前缺少一个非交互式的规则几何遮罩节点，用于在局部重绘前快速创建矩形、方形、圆形、椭圆形遮罩，并在需要时把结果继续交给 `LLS Simple Mask Draw` 做手工修订。

这个节点只负责：

- 根据输入 `image` 尺寸创建规则 `mask`
- 输出 `preview_image`
- 输出面积与 bbox 信息

它不负责采样，不负责 VAE，不负责生成图像。

## Goals

- 新增一个节点：`LLS Simple Mask Create`
- 支持 `rectangle` / `square` / `circle` / `ellipse`
- 支持 `pixel` / `percent` 坐标模式
- 支持 `feather`、`blur`、`invert_mask`
- 支持 `input_mask` 与新遮罩的 `replace` / `union` / `subtract` / `intersect`
- 输出：
  - `image`
  - `mask`
  - `preview_image`
  - `area_info`
- `mask` 输出与 `LLS Simple Repair Prepare.mask` 直接兼容

## Non-Goals

本阶段不实现：

- 多边形 / 贝塞尔路径
- 文本到 mask
- 自动分割 / SAM / 智能抠图
- 前端交互编辑器
- 多颜色混合 overlay
- area_info 下游消费节点

## Chosen Architecture

沿用现有目录结构，正式启用 `mask/` 子包：

- `mask/mask_utils.py`
  - 图像尺寸解析
  - 几何 mask 生成
  - mask resize / combine
  - feather / blur
  - preview overlay
  - area_info 统计
- `mask/mask_create.py`
  - `LLSSimpleMaskCreate` 节点定义
- `mask/nodes.py`
  - 仅负责导入和注册，不承载实现逻辑

这样可以把规则 mask 和交互 mask 分开：

- `mask/` 专注规则几何生成
- `mask_draw/` 专注前端交互绘制

## Node Contract

### Registration

- class key: `LLSSimpleMaskCreate`
- display name: `LLS Simple Mask Create`
- category: `LLS/Mask`

### Inputs

Required:

- `image: IMAGE`
- `shape_type: rectangle | square | circle | ellipse`
- `coordinate_mode: pixel | percent`
- `center_x: FLOAT`
- `center_y: FLOAT`
- `width: FLOAT`
- `height: FLOAT`
- `radius: FLOAT`
- `feather: FLOAT`
- `blur: FLOAT`
- `invert_mask: BOOLEAN`
- `combine_mode: replace | union | subtract | intersect`
- `overlay_alpha: FLOAT`
- `overlay_color: red | green | blue`

Optional:

- `input_mask: MASK`

### Outputs

- `image: IMAGE`
- `mask: MASK`
- `preview_image: IMAGE`
- `area_info: LLS_MASK_INFO`

`area_info` 实际返回 Python `dict`，自定义类型名仅用于工作流连线语义。

## Geometry Rules

输入 `image` 形状为 `[batch, H, W, C]`。

### Coordinate Conversion

`pixel` 模式：

- `center_px_x = center_x`
- `center_px_y = center_y`
- `width_px = width`
- `height_px = height`
- `radius_px = radius`

`percent` 模式：

- `center_px_x = center_x * W`
- `center_px_y = center_y * H`
- `width_px = width * W`
- `height_px = height * H`
- `radius_px = radius * min(W, H)`

### Shapes

`rectangle`

- 使用 `center_x / center_y / width / height`
- bbox 由中心点和宽高计算

`square`

- 使用 `center_x / center_y / width`
- 边长 = `width`
- `height` 忽略

`circle`

- 使用 `center_x / center_y / radius`

`ellipse`

- 使用 `center_x / center_y / width / height`
- `rx = width / 2`
- `ry = height / 2`

所有 bbox 都会裁剪到图像边界内。

### Edge Cases

- `image` 为空：报错
- `width / height / radius <= 0`：夹到最小值 `1`
- 中心点允许在图像外，但最终 bbox 会裁剪
- 形状完全在图像外：返回全黑 mask，不崩溃
- `input_mask` 尺寸不匹配：自动 resize
- 没有 `input_mask` 但选择 `union / subtract / intersect`：退化为 `created_mask`

## Mask Processing

### Combine Order

1. 先生成 `created_mask`
2. 如果有 `input_mask`，先做 `combine_mode`
3. 最后应用 `invert_mask`
4. 再做 `feather` 和 `blur` 的软化收尾
5. clamp 到 `[0, 1]`

这个顺序保证：

- `invert_mask` 是对最终语义结果做反转
- `combine_mode` 行为稳定
- 软边作用于最终遮罩边界

### Feather and Blur

- `feather`：用于边缘软化，主要作用于二值几何边界
- `blur`：用于整体模糊和平滑

第一版统一用张量运算实现，限制在合理范围内，避免引入额外依赖。

## Preview Image

`preview_image` = 原图 + 半透明彩色 overlay：

- 默认 `red`
- 可选 `green` / `blue`
- `overlay_alpha` 控制叠加强度

输出尺寸与原图完全一致，不修改透传 `image`。

## Area Info

返回 dict，字段包含：

- `image_size`
- `shape_type`
- `coordinate_mode`
- `center`
- `width`
- `height`
- `radius`
- `bbox`
- `geometric_area_px`
- `binary_area_px`
- `effective_area_px`
- `area_ratio`
- `feather`
- `blur`
- `invert_mask`
- `combine_mode`
- `clipped_by_image`

### Area Semantics

- `geometric_area_px`
  - 裁剪前的理论几何面积
- `binary_area_px`
  - `mask > 0.5` 的像素数量
- `effective_area_px`
  - `sum(mask)`，用于软 mask
- `area_ratio`
  - `binary_area_px / (W * H)`

如果几何形状超出图像边界，`geometric_area_px` 可能大于 `binary_area_px`，这正是需要暴露 `clipped_by_image` 的原因。

## Testing Strategy

测试覆盖：

- 节点注册与 schema
- 四种几何形状
- pixel / percent
- invert / feather / blur
- `input_mask` 的 `union / subtract / intersect`
- `preview_image` overlay
- `area_info` 结构与数值
- 输出接入 `LLS Simple Repair Prepare`

## README Changes

README 新增一节 `LLS Simple Mask Create`，说明：

- 节点作用
- 输入输出
- `shape_type`
- `coordinate_mode`
- `area_info`
- 和 `LLS Simple Repair Prepare` 的接法
- 示例工作流

