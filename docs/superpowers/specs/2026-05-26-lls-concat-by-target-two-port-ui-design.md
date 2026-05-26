# LLS Concat By Target Two-Port UI Design

## Context

`LLS Concat By Target` 当前已经支持自动识别 `IMAGE` 和 `MASK` 输入，并且运行时可以只靠两路数据完成拼接逻辑。

当前问题不在运行逻辑，而在 ComfyUI 节点 UI：

- 后端把两个输入声明成 `IMAGE,MASK`
- ComfyUI 前端会把这种联合类型拆成 4 个可见输入口
- 用户实际看到的是 4 个端口，而不是预期的 2 个端口

用户要的最终效果很明确：

- 节点左侧只保留 2 个可见输入口
- 这两个输入口显示为 `image/mask_A` 和 `image/mask_B`
- 两个口都继续支持 `IMAGE` 和 `MASK`
- 运行时继续自动判断实际输入类型

## Goals

- 把 `LLS Concat By Target` 的 UI 从 4 个可见输入口收敛成 2 个
- 保持两个输入都能接 `IMAGE` 或 `MASK`
- 保持现有自动识别逻辑和混接报错逻辑
- 让节点界面显示名使用 `image/mask_A` 和 `image/mask_B`
- 保持现有输出和拼接行为不变

## Non-Goals

本次不做以下事情：

- 不新增独立的 `image_A` / `image_B` / `mask_A` / `mask_B` 四路 schema
- 不修改拼接算法、尺寸匹配、gap、batch broadcast 等运行时行为
- 不引入额外 widget 来切换输入模式
- 不尝试修改 ComfyUI 核心联合类型渲染逻辑

## Chosen Approach

采用“后端 2 个通配输入 + 前端改显示标签”的组合方案。

### Backend

- 把 `LLSConcatByTarget.INPUT_TYPES()` 的可选输入从：
  - `image/mask_A: IMAGE,MASK`
  - `image/mask_B: IMAGE,MASK`
- 调整为：
  - `a: *`
  - `b: *`

这里的 `*` 沿用项目里已经用过的 `AnyType("*")` 模式，使 ComfyUI 在节点 UI 中只生成 2 个可见输入口，而不是按联合类型拆成 4 个。

运行时继续保留现有自动识别逻辑：

- `a` 和 `b` 都是图像张量时，走 `IMAGE` 拼接
- `a` 和 `b` 都是遮罩张量时，走 `MASK` 拼接
- 一边图像一边遮罩时，抛出清晰错误

### Frontend

- 新增一个很小的前端扩展文件，例如 `web/js/lls_concat_by_target.js`
- 在 `beforeRegisterNodeDef` 中挂接 `LLSConcatByTarget`
- 在节点创建后查找输入名为 `a` 和 `b` 的两个输入
- 只修改它们的显示标签：
  - `a -> image/mask_A`
  - `b -> image/mask_B`

这个扩展只负责 UI 呈现，不改变连接和运行逻辑。内部 schema 名仍然是 `a` / `b`，所以不会依赖 ComfyUI 对带斜杠名字的输入做特殊处理。

## Why This Approach

### Option A: 直接使用 `IMAGE,MASK` 联合类型并把输入名写成 `image/mask_A` / `image/mask_B`

不采用。

原因：

- 这正是当前 4 个可见输入口的来源
- 即使名字符合预期，UI 结构仍然不符合用户目标

### Option B: 后端使用 `a` / `b` 通配输入，但不加前端重命名

不采用。

原因：

- 可以得到 2 个可见输入口
- 但界面显示会是 `a` 和 `b`，不符合用户要求

### Option C: 后端通配输入 + 前端改标签

采用。

原因：

- 可以稳定得到 2 个可见输入口
- 可以把显示名改成用户想要的 `image/mask_A` / `image/mask_B`
- 不需要修改现有拼接逻辑
- 风险和改动范围最小

## File Changes

本次会修改或新增以下文件：

- Modify: `utils/concat_by_target.py`
- Create: `web/js/lls_concat_by_target.js`
- Modify: `tests/test_concat_by_target_registration.py`
- Modify: `tests/test_concat_by_target_node.py`
- Create: `tests/test_concat_by_target_frontend.py`

如有需要，也可以补一条节点文档说明到 `LLS-Nodes-Reference.md`，但这不是本次行为变更的阻塞项。

## Backend Contract

### Node Identity

- class key: `LLSConcatByTarget`
- display name: `LLS Concat By Target`
- category: `LLS/Utils`
- function: `concat`

### Inputs

Required inputs 保持不变：

- `data_type`
- `target`
- `position`
- `match_target_size`
- `resize_mode`
- `align`
- `gap`
- `background_color`
- `background_value`
- `multiple_of`
- `allow_batch_broadcast`

Optional inputs 调整为：

- `a: *`
- `b: *`

### UI Labels

节点界面中：

- `a` 显示为 `image/mask_A`
- `b` 显示为 `image/mask_B`

这里的“显示为”仅指前端 label，不改变后端实际字段名。

### Outputs

保持不变：

- `image: IMAGE`
- `mask: MASK`
- `width: INT`
- `height: INT`

## Runtime Behavior

运行时行为保持现状：

1. 从 `a` / `b` 读取输入
2. 根据张量形状自动推断是 `IMAGE` 还是 `MASK`
3. 两边同类型时继续对应拼接流程
4. 两边类型不一致时抛出明确错误
5. `data_type` 仅在无法从输入推断时作为回退值

这意味着本次改动是 UI/schema 形态调整，不是算法行为调整。

## Error Handling

保留或继续强化以下报错语义：

- `IMAGE` 路径缺少 `a` 或 `b` 时，报缺少两个输入
- `MASK` 路径缺少 `a` 或 `b` 时，报缺少两个输入
- `a` / `b` 混接 `IMAGE` 和 `MASK` 时，报两边类型不一致
- 输入 shape 非法时，继续沿用现有 shape 校验错误

错误消息可以在文案中提及 `image/mask_A` / `image/mask_B`，但内部实现仍可使用 `a` / `b` 来接收参数。

## Testing

### Registration Tests

更新注册测试，验证：

- `optional["a"]` 和 `optional["b"]` 存在
- 可选输入不再是 `image/mask_A` / `image/mask_B`
- 节点元数据和输出保持不变

### Runtime Tests

保留现有节点运行测试，确保：

- 图像输入仍然可以拼接
- 遮罩输入仍然可以拼接
- 自动识别逻辑不变
- 混接仍然报错

这些测试大多只需要继续走 `a` / `b` 输入，不需要覆盖 UI 标签。

### Frontend Tests

新增前端资产测试，验证：

- `web/js/lls_concat_by_target.js` 存在
- 文件注册了 ComfyUI 扩展
- 目标节点是 `LLSConcatByTarget` / `LLS Concat By Target`
- 文件中包含 `image/mask_A` 和 `image/mask_B`
- 文件中包含对 `a` 和 `b` 的标签重写逻辑

## Compatibility

### New Nodes

新建节点时，用户会看到 2 个输入口，符合目标 UI。

### Existing Workflows

这是本次唯一需要注意的兼容点。

如果历史工作流已经把 schema 输入名保存成 `image/mask_A` / `image/mask_B`，单纯把后端改回 `a` / `b` 可能导致旧工作流无法自动连回对应输入。

因此实现阶段需要一并确认其中一种处理方式：

- 优先方案：在前端节点创建/图配置完成时，把旧工作流中的这两个历史输入名映射到当前节点的 `a` / `b`
- 如果 ComfyUI 当前序列化结构不便做安全映射，则至少要在实现中验证：
  - 新建节点满足目标
  - 并明确记录旧工作流是否需要手动重连

如果兼容映射实现足够简单且可靠，应当一并纳入本次修改；如果映射风险偏高，则优先保证新节点 UI 和运行正确，不在本次扩展更大范围的工作流迁移逻辑。

## Acceptance Criteria

- `LLS Concat By Target` 在 ComfyUI 界面中只显示 2 个输入口
- 两个输入口显示名分别为 `image/mask_A` 和 `image/mask_B`
- 两个输入口都能连接 `IMAGE` 或 `MASK`
- 连接两路 `IMAGE` 时，图像拼接行为保持正确
- 连接两路 `MASK` 时，遮罩拼接行为保持正确
- 混接 `IMAGE` 与 `MASK` 时，继续报错
- 自动化测试覆盖后端注册、运行逻辑和前端扩展存在性
