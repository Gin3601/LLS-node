# LLS Simple Image Composite Design

## Context

`LLS-node` 目前已经有一批图像相关节点集中在 `image/` 域里，例如：

- `LLS Simple VAE Encode`
- `LLS Simple VAE Decode`
- `LLS Save Image`

同时，项目已经具备一个“后端节点 + ComfyUI 节点内前端交互扩展”的成熟模式：

- 后端：`mask_draw/node.py`
- 前端：`web/js/lls_mask_draw.js`

这说明新节点不仅可以输出真实图像结果，也可以在 ComfyUI 节点内部提供实时预览、拖动操作和 widget 联动。

本次需要新增一个图像合成节点：

- 节点名：`LLS Simple Image Composite`
- 作用：把 `overlay_image` 叠加到 `background_image` 上
- 特点：支持平移、缩放、旋转、透明度、锚点定位和旋转原点切换
- 交互：支持 ComfyUI 节点内实时预览与拖动调参

## Goals

- 新增一个后端节点：`LLSSimpleImageComposite`
- 新增一个节点内前端扩展：实时预览合成结果
- 支持以下后端参数：
  - `background_image`
  - `overlay_image`
  - `x_offset`
  - `y_offset`
  - `anchor_mode`
  - `rotation_origin_mode`
  - `opacity`
  - `blend_mode`
  - `scale`
  - `rotation`
  - `keep_aspect`
- 支持以下行为：
  - 前景图平移
  - 前景图缩放
  - 前景图旋转
  - 前景图超出背景时自动裁剪
  - 前景图带 alpha 时优先按 alpha 合成
  - 输出尺寸始终等于背景图尺寸
- 更新 README 和测试

## Non-Goals

本次不实现：

- 多种混合模式，第一版只支持 `normal`
- 独立浏览器应用或独立前端页面
- 类似 Photoshop 的完整控制框、旋转手柄、九宫格 UI
- 透视变换、自由扭曲、蒙版编辑
- 来自任意中间张量节点的可靠前端源图预览

## Chosen Architecture

采用“后端真实输出 + 前端节点内实时预览”的双层结构。

### Backend

- `image/composite_utils.py`
  - 负责纯图像合成逻辑
  - 处理尺寸、batch、alpha、平移、裁剪、缩放、旋转
- `image/composite.py`
  - 定义 `LLSSimpleImageComposite` 节点
  - 暴露 `INPUT_TYPES()`、`RETURN_TYPES`、`composite()` 方法
  - 依赖 `composite_utils.py`
- `image/__init__.py`
  - 合并 `image/nodes.py` 与 `image/composite.py` 的注册表
  - 把 `LLSSimpleImageComposite` 暴露到现有 `LLS/Image` 分类中

### Frontend

- `web/js/lls_image_composite.js`
  - 注册 ComfyUI 扩展
  - 在节点内添加 DOM 预览画布
  - 支持拖动前景图实时更新 `x_offset / y_offset`
  - 支持 widget 联动：
    - `x_offset`
    - `y_offset`
    - `scale`
    - `rotation`
    - `opacity`
    - `anchor_mode`
    - `rotation_origin_mode`
  - 实时显示本地预览结果

### Tests

- 新增后端节点测试
- 新增前端资产与扩展注册测试
- 更新 README 文档测试

## File Structure

本次新增或修改以下文件：

- Create: `image/composite.py`
- Create: `image/composite_utils.py`
- Create: `web/js/lls_image_composite.js`
- Modify: `image/__init__.py`
- Modify: `README.md`
- Create: `tests/test_image_composite_registration.py`
- Create: `tests/test_image_composite_node.py`
- Create: `tests/test_image_composite_frontend.py`
- Create: `tests/test_image_composite_docs.py`

## Node Contract

### Registration

- class key: `LLSSimpleImageComposite`
- display name: `LLS Simple Image Composite`
- category: `LLS/Image`
- function: `composite`

### Inputs

Required:

- `background_image: IMAGE`
- `overlay_image: IMAGE`
- `x_offset: INT`
- `y_offset: INT`
- `anchor_mode: top_left | center`
- `rotation_origin_mode: top_left | center`
- `opacity: FLOAT`
- `blend_mode: normal`
- `scale: FLOAT`
- `rotation: FLOAT`
- `keep_aspect: BOOLEAN`

### Outputs

- `output_image: IMAGE`

## Parameter Semantics

### `x_offset` / `y_offset`

- 类型：`INT`
- 默认：`0`
- 表示前景图在背景上的平移偏移
- `x_offset > 0` 向右移动
- `x_offset < 0` 向左移动
- `y_offset > 0` 向下移动
- `y_offset < 0` 向上移动

### `anchor_mode`

- `top_left`
  - `x_offset / y_offset` 解释为前景图放到背景上的左上角位置
- `center`
  - `x_offset / y_offset` 解释为前景图放到背景上的中心点位置

### `rotation_origin_mode`

- `top_left`
  - 前景图围绕自身左上角旋转
- `center`
  - 前景图围绕自身中心点旋转

这两个模式与 `anchor_mode` 独立，因此允许以下组合：

- 按左上角定位 + 按左上角旋转
- 按左上角定位 + 按中心旋转
- 按中心定位 + 按左上角旋转
- 按中心定位 + 按中心旋转

### `opacity`

- 类型：`FLOAT`
- 默认：`1.0`
- 范围：`0.0 ~ 1.0`
- 表示前景图叠加透明度

### `blend_mode`

- 第一版仅支持 `normal`

### `scale`

- 类型：`FLOAT`
- 默认：`1.0`
- 最小值：`0.01`
- 表示前景图整体缩放倍率

### `rotation`

- 类型：`FLOAT`
- 默认：`0.0`
- 单位：角度（degrees）
- 正负方向在实现上统一为标准 2D 旋转角

### `keep_aspect`

- 类型：`BOOLEAN`
- 默认：`True`
- 第一版由于只提供统一 `scale` 而不是独立 `scale_x / scale_y`，因此该参数会保留在接口中，但结果上始终保持等比缩放
- 保留它是为了后续兼容非等比缩放扩展，不会在本阶段引入额外行为分支

## Backend Compositing Pipeline

后端合成顺序固定为：

1. 校验输入图像 shape
2. 解析 batch 关系
3. 对前景图进行缩放
4. 对缩放后的前景图执行旋转
5. 按 `anchor_mode + x_offset + y_offset` 计算落点
6. 把前景图裁剪到背景边界内
7. 按 alpha 与 opacity 做 normal blend
8. 输出与背景同尺寸的图像

## Image Shape Rules

### Background

- 期望 shape：`[batch, height, width, channels]`
- 输出尺寸始终使用背景图尺寸

### Overlay

- 期望 shape：`[batch, height, width, channels]`
- 允许 3 通道 RGB
- 允许 4 通道 RGBA

### Batch Rules

- 若 `background_image.batch == overlay_image.batch`，逐样本一一合成
- 若 `overlay_image.batch == 1` 且背景 batch > 1，则 overlay 自动广播到每一张背景图
- 其他 batch 组合报错

## Alpha Handling

### RGBA Overlay

如果前景图是 4 通道：

- RGB 部分作为颜色
- A 通道作为局部透明度
- 实际混合权重 = `alpha * opacity`

### RGB Overlay

如果前景图是 3 通道：

- 整张前景图视为 alpha = 1
- 实际混合权重 = `opacity`

## Cropping And Bounds

### Fully Outside

如果前景图完全落在背景边界外：

- 不报错
- 直接返回背景图

### Partially Outside

如果前景图只部分在背景内：

- 自动计算交集区域
- 只把交集区域参与合成

## Transformation Details

### Scale

- 使用统一缩放倍率 `scale`
- 不支持单独 `scale_x / scale_y`
- 输出前景尺寸按 `round(original * scale)` 取整，并保证至少为 `1`

### Rotation

- 第一版使用 Pillow 进行旋转
- 旋转后保留 alpha 信息
- 根据 `rotation_origin_mode` 先平移到目标旋转原点，再做旋转，最后再回贴

采用 Pillow 的原因：

- 当前仓库已经在 `mask_draw/utils.py` 使用 Pillow
- 在现有测试环境中比依赖 ComfyUI 前端运行时更容易稳定验证
- 有利于准确处理 RGBA 和旋转后的包围盒

## Frontend Preview Behavior

### What The Frontend Does

节点内前端扩展负责：

- 读取背景图和前景图源
- 本地渲染预览画布
- 处理拖动移动
- 把拖动结果同步回 widgets
- 响应 widget 改动并重绘预览

### Source Resolution Scope

第一版节点内实时预览优先支持以下可文件化上游：

- `Load Image`
- `Load Image Output`

如果上游不是可直接解析文件 URL 的节点：

- 后端真实输出仍然正常
- 前端预览显示提示信息，而不是崩溃

这是第一版的明确限制，不属于 bug。

## Error Handling

### Required Runtime Errors

- `background_image is None`：报错
- `overlay_image is None`：报错
- 图像 shape 非法：报错
- batch 不兼容：报错
- `blend_mode != normal`：报错
- `scale <= 0`：报错或夹紧到最小有效值，本阶段选择夹紧到 `0.01`

### Non-Error Cases

- `opacity = 0`：直接返回背景图
- 前景完全移出背景：直接返回背景图
- 前景部分越界：自动裁剪并继续

## Testing Strategy

### Registration Tests

- 节点能注册到插件
- display name 正确
- `CATEGORY = "LLS/Image"`
- schema 包含全部参数

### Backend Node Tests

- 两张图像能正常输入并输出
- 输出尺寸与背景图一致
- `x_offset / y_offset` 能移动前景图
- `anchor_mode = top_left` 生效
- `anchor_mode = center` 生效
- `rotation_origin_mode = top_left` 生效
- `rotation_origin_mode = center` 生效
- `opacity = 0` 返回背景图
- `opacity = 1` 正常覆盖
- 部分越界能正确裁剪
- 完全越界返回背景图
- RGBA overlay 会按 alpha 合成
- RGB overlay 会按整图 opacity 合成
- `scale` 生效
- `rotation` 生效

### Frontend Tests

- `web/js/lls_image_composite.js` 文件存在
- 扩展注册到 `LLSSimpleImageComposite`
- 包含 `beforeRegisterNodeDef`
- 包含 `addDOMWidget`
- 包含拖动逻辑
- 包含 widget 同步逻辑

### Docs Tests

- README 提到 `LLS Simple Image Composite`
- README 提到输入输出
- README 提到：
  - `x_offset`
  - `y_offset`
  - `anchor_mode`
  - `rotation_origin_mode`
  - `opacity`
  - `scale`
  - `rotation`

## README Changes

README 新增一个 `LLS Simple Image Composite` 小节，包含：

- 节点用途说明
- 输入输出说明
- 参数说明
- 典型用法：
  - `Load Image(background) + Load Image(overlay) -> LLS Simple Image Composite -> Preview Image`
- 节点内实时预览限制说明

## Open Decisions Resolved

以下事项已经固定：

- 采用方案 2：后端真实输出 + ComfyUI 节点内前端实时预览
- 第一版就实现完整参数接口，不只留空接口
- `rotation_origin_mode` 独立于 `anchor_mode`
- 第一版只支持 `blend_mode = normal`
- 第一版允许 `keep_aspect` 出现在接口中，但实际缩放仍保持等比
- 节点位置放在 `image` 域中
