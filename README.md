# LLS ComfyUI Custom Nodes

按 ComfyUI 功能域封装的自定义节点插件包。所有节点统一使用 `LLS/` 命名空间，
顶层 `__init__.py` 自动扫描并汇总各子包的注册表，ComfyUI 只需识别一个插件入口。

## 安装方法

1. 将整个 `node/` 目录复制到 `ComfyUI/custom_nodes/`，并重命名为 `LLS_Nodes/`
2. 重启 ComfyUI
3. 在节点搜索框中搜索 `LLS` 即可看到所有已实现节点

```
ComfyUI/
└── custom_nodes/
    └── LLS_Nodes/        ← 将本目录内容放在这里
        ├── __init__.py
        ├── nodes.py
        ├── upscale/
        ├── utils/
        └── ...
```

## 目录结构

```
node/
├── __init__.py           ← 插件入口：自动汇总所有子包的 NODE_CLASS_MAPPINGS
├── nodes.py              ← 共享工具函数（不含节点类）
│
├── model_loader/         ← 第 1 节：模型加载与管理
├── conditioning/         ← 第 2 节：文本编码与条件生成
├── sampling/             ← 第 3 节：图像采样与去噪
├── latent/               ← 第 4 节：Latent 空间操作
├── image/                ← 第 5 节：图像处理与后处理
├── upscale/              ← 第 5 节（超分子域）：图像超分辨率放大
├── mask/                 ← 第 6 节：遮罩操作
├── controlnet/           ← 第 7 节：ControlNet / 控制引导
├── lora/                 ← 第 8 节：LoRA / Hypernetwork / Adapter
├── video/                ← 第 10 节：视频生成与帧处理
├── audio/                ← 第 11 节：音频处理
└── utils/                ← 第 14 节：数据类型与逻辑工具
```

## 已实现节点

### LLS/Upscale

| 节点名 | 说明 |
|--------|------|
| `LLS Upscale Switcher` | 在 upscale_model 与 PyTorch 插值之间切换的超分节点 |

**输入参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | IMAGE | — | 输入图像 |
| `mode` | 选项 | `upscale_model` | `upscale_model`（模型超分）或 `pytorch`（插值放大） |
| `scale` | FLOAT | 2.0 | 放大倍数（仅 pytorch 模式生效） |
| `interpolation` | 选项 | `bilinear` | 插值算法（仅 pytorch 模式生效） |
| `model_name` | 选项 | — | 超分模型名（从 `models/upscale_models/` 读取） |
| `tile` | INT | 512 | 分块大小，越小显存越省（仅 upscale_model 模式生效） |
| `overlap` | INT | 32 | 分块重叠像素（仅 upscale_model 模式生效） |

**输出：** `IMAGE`

---

### LLS/Utils

| 节点名 | 说明 |
|--------|------|
| `LLS String Literal` | 输出一个字符串常量，可复用于多个下游节点 |
| `LLS Int Literal` | 输出一个整数常量 |
| `LLS Float Literal` | 输出一个浮点数常量 |
| `LLS Resolution Selector` | 从常用分辨率预设中选择，或输入自定义宽高，返回 width 和 height |

---

## Image Repair

`LLS-node` 现在提供一套面向本地工作流的修复链路，由以下节点组成：

- `LLS Simple Mask Create`
- `LLS Simple Mask Draw`
- `LLS Save Image`
- `LLS Simple Repair Prepare`
- `LLS Simple KSampler`
- `LLS Simple Repair Finish`

修复工作流不是从 `LLS Simple Empty Latent` 开始，而是从 `image + mask + vae` 开始。`LLS Simple Repair Prepare` 会把这些输入转换为带有 `repair_info` 的修复态 latent，再交给同一个 `LLS Simple KSampler` 继续采样。

### `LLS Simple Mask Create`

`LLS Simple Mask Create` 是一个基础几何遮罩生成节点。它不依赖原图内容，只根据输出尺寸和几何参数直接创建 `rectangle`、`square`、`circle`、`ellipse` 四种基础 `mask`。

它适合放在修复准备前或手工绘制前：

- `LLS Simple Mask Create.mask_image -> Preview Image`
- `LLS Simple Mask Create.mask -> LLS Save Image.mask`
- `Load Image.image + LLS Simple Mask Create.mask -> LLS Simple Repair Prepare`
- `LLS Simple Mask Create.mask -> LLS Simple Mask Draw.input_mask`

**推荐连接方式：**

- `LLS Simple Mask Create.mask_image -> Preview Image`
- `Load Image.image -> LLS Save Image.image`
- `LLS Simple Mask Create.mask -> LLS Save Image.mask`
- `Load Image.image -> LLS Simple Repair Prepare.image`
- `LLS Simple Mask Create.mask -> LLS Simple Repair Prepare.mask`
- `LLS Simple Mask Create.mask -> LLS Simple Mask Draw.input_mask`
- `Load Image.image -> LLS Simple Mask Draw.image`

**输入：**

| 名称 | 类型 | 说明 |
|------|------|------|
| `image_width` | `INT` | 必填，输出遮罩宽度 |
| `image_height` | `INT` | 必填，输出遮罩高度 |
| `input_mask` | `MASK` | 可选，已有遮罩；可与当前几何 mask 做 `replace` / `union` / `subtract` / `intersect` |

**核心参数：**

| 参数 | 说明 |
|------|------|
| `shape_type` | `rectangle` / `square` / `circle` / `ellipse` |
| `coordinate_mode` | `pixel` / `percent` |
| `center_x`, `center_y` | 中心点坐标；`percent` 模式下范围通常为 `0.0 ~ 1.0` |
| `width`, `height` | `rectangle` / `square` / `ellipse` 使用 |
| `radius` | `circle` 使用 |
| `feather` | 边缘羽化 |
| `blur` | 整体边缘模糊 |
| `invert_mask` | 反转最终 mask |
| `combine_mode` | `replace` / `union` / `subtract` / `intersect` |

**输出：**

| 名称 | 类型 | 说明 |
|------|------|------|
| `mask` | `MASK` | 最终遮罩，白色 / `1` 表示要重绘，黑色 / `0` 表示不重绘 |
| `mask_image` | `IMAGE` | 将 `mask` 直接可视化为黑白图像，方便单独预览 |
| `area_info` | `LLS_MASK_INFO` | 包含面积、占比、bbox、坐标模式等信息的 dict |

`area_info` 主要包含：

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

**默认行为：**

- `shape_type = rectangle`
- `coordinate_mode = percent`
- `center_x = 0.5`
- `center_y = 0.5`
- `image_width = 1024`
- `image_height = 1024`
- `width = 0.3`
- `height = 0.3`
- `radius = 0.15`
- `combine_mode = replace`

这会在图像中心创建一个大约占宽高 `30%` 的矩形 mask。

**示例：**

- 中心矩形：`shape_type=rectangle, coordinate_mode=percent, center_x=0.5, center_y=0.5, width=0.3, height=0.3`
- 中心圆形：`shape_type=circle, coordinate_mode=percent, center_x=0.5, center_y=0.5, radius=0.15`
- 左上角方形：`shape_type=square, coordinate_mode=percent, center_x=0.2, center_y=0.2, width=0.2`
- 与已有 mask 合并：连接 `input_mask` 并将 `combine_mode` 设为 `union`

**典型用途：**

- 快速创建规则删除区域
- 快速创建规则修复区域
- 先创建基础几何区域，再交给 `LLS Simple Mask Draw` 手动修边
- 为 `LLS Simple Repair Prepare` 提供稳定的初始局部重绘 mask

### `LLS Save Image`

`LLS Save Image` 现在除了保存或预览 `image`，也支持接收一个可选的 `mask` 输入。连上 `mask` 之后，节点会额外输出一份独立的黑白遮罩图，不会覆盖原图结果，也不会做彩色叠加。

**推荐工作流：**

- `Load Image.image -> LLS Save Image.image`
- `LLS Simple Mask Create.mask -> LLS Save Image.mask`
- `LLS Simple Mask Draw.mask -> LLS Save Image.mask`
- `LLS Simple Mask Create.mask_image -> Preview Image`

**行为说明：**

- `output_mode = save` 时：保存原图，同时额外保存一份 `<filename_prefix>_mask`
- `output_mode = preview_only` 时：预览原图，同时额外预览一份黑白 mask
- `image` 与 `mask` 各自独立输出，互不影响

### `LLS Simple Mask Draw`

`LLS Simple Mask Draw` 是一个面向局部重绘的交互式遮罩输入节点，用来直接在输入图像上手动画 `mask`，并把结果输出给后续修复链路。

**推荐工作流：**

- `Load Image -> LLS Simple Mask Draw -> Preview Image`
- `Load Image -> LLS Simple Mask Draw -> LLS Simple Repair Prepare`
- `Load Image -> LLS Simple Mask Create -> LLS Simple Mask Draw`
- `Load Image -> LLS Simple Mask Create -> LLS Simple Mask Draw -> LLS Simple Repair Prepare`

**推荐连接方式：**

- `Load Image.image -> LLS Simple Mask Draw.image`
- `LLS Simple Mask Draw.image -> LLS Simple Repair Prepare.image`
- `LLS Simple Mask Draw.mask -> LLS Simple Repair Prepare.mask`
- `LLS Simple Mask Draw.mask -> LLS Save Image.mask`

**输入：**

| 名称 | 类型 | 说明 |
|------|------|------|
| `image` | `IMAGE` | 必填，原图输入 |
| `input_mask` | `MASK` | 可选，已有遮罩；节点会在这个遮罩基础上继续编辑 |

**输出：**

| 名称 | 类型 | 说明 |
|------|------|------|
| `image` | `IMAGE` | 原图透传 |
| `mask` | `MASK` | 最终遮罩，白色 / `1` 表示要重绘，黑色 / `0` 表示不重绘 |
| `preview_image` | `IMAGE` | 原图与半透明红色遮罩叠加后的预览图 |

**当前支持的交互功能：**

- 直接在图像上绘制白色遮罩
- `brush` / `erase` 两种模式
- 调整 `brush_size`
- 调整 `brush_softness`
- 调整 `overlay_alpha`
- 预览半透明红色遮罩叠加
- `Clear`
- `Undo`
- `Redo`
- `Invert`
- 在已有 `input_mask` 基础上继续补画或擦除

**如何手动画 mask：**

1. 连接 `Load Image` 到 `LLS Simple Mask Draw.image`
2. 如果有已有遮罩，可再连接到 `input_mask`
3. 在节点预览区域直接涂抹需要修复的区域
4. 用 `erase` 擦除不需要重绘的部分
5. 用 `Clear / Undo / Redo / Invert` 调整最终结果
6. 将 `mask` 输出接到 `LLS Simple Repair Prepare.mask`

**典型用途：**

- 手动指定删除区域
- 手动指定修复区域
- 手动指定去阴影区域
- 手动指定局部增强区域

**当前版本限制：**

- 第一版只支持 `brush` 和 `erase`
- 还没有实现 `polygon`、`rectangle`、`ellipse`
- 还没有实现魔棒、自动分割、智能抠图
- 如果需要规则几何初始化区域，请先使用 `LLS Simple Mask Create`

### Minimal Repair Workflow

`Load Image -> Load Mask -> LLS Simple Checkpoint Loader -> LLS Simple Prompt Encode -> LLS Simple Repair Prepare -> LLS Simple KSampler -> VAE Decode -> LLS Simple Repair Finish -> Preview Image`

### Minimal Manual Mask Workflow

`Load Image -> LLS Simple Mask Draw -> LLS Simple Repair Prepare -> LLS Simple KSampler -> VAE Decode -> LLS Simple Repair Finish -> Preview Image`

### `repair_scope`

- `region`：在原图边界内直接修复
- `crop`：围绕 mask 裁剪局部工作区，提高局部细节修复精度
- `canvas`：扩展画布后修复新增区域或缺失区域

### `repair_kernel`

- `latent_mask`：对工作图做 VAE encode，并附带 latent noise mask
- `vae_inpaint`：使用兼容 VAE inpaint 的 latent 准备方式
- `native_fill`：当后端原生支持 fill/inpaint 时优先请求原生能力，否则回退并发出 warning

### `denoise_mode`

- `manual`：直接使用 sampler 上的 `denoise`
- `auto_from_repair`：使用 `repair_info["recommended_denoise"]`

### Compatibility

- 未连接 `repair_info` 时，现有 txt2img 工作流保持不变。
- 未连接 `repair_info` 时，现有 img2img 工作流保持不变。
- 同一个 `LLS Simple KSampler` 同时处理 txt2img、img2img 和 repair。

---

## Pro Image Edit / Inpaint

`LLS-node` 也提供一套更严格的专业局部编辑链路，针对真正支持原生 image edit / inpaint 语义的模型：

- `LLS Pro Image Edit Prepare`
- `LLS Pro KSampler Bridge`
- `LLS Pro Image Edit Finish`

### Simple vs Pro

- `Simple = lightweight masked latent resampling`
- `Pro = true image edit / inpaint pipeline`

当模型本身具备真实的局部编辑能力，而且你更在意遮罩区域内的提示词跟随和原生 inpaint / image-edit 行为时，用 `Pro` 链；如果你只是需要兼容性更高、约束更少的局部重绘流程，继续使用 `Simple` 链。

### Professional Workflow

`Load Image -> Load Mask or LLS Simple Mask Draw -> LLS Simple Checkpoint Loader -> LLS Simple Prompt Encode -> LLS Pro Image Edit Prepare -> LLS Pro KSampler Bridge -> VAE Decode -> LLS Pro Image Edit Finish -> Preview Image`

### Backend Selection

- `backend_mode = auto | sdxl | flux`
- `auto` 现在使用 profile-driven routing：先解析模型 profile，再决定是走原生编辑路径还是 fallback 局部重绘路径
- 原生 edit/inpaint profile 走 native path；base/generation profile 不再直接报错，而是进入 fallback local repaint
- `sdxl` 和 `flux` 会按家族强制指定后端，但仍然会验证当前模型是否兼容

### Profile-Driven Routing

LLS Simple Checkpoint Loader writes the resolved model profile into runtime metadata.

- `LLS Pro Image Edit Prepare` routes by `backend_type`
- `LLS Pro KSampler Bridge` routes by `sampler_strategy`
- `execution_path` records whether this run used `native_edit` or `fallback_repair`

重要 profile 字段：

- `profile_id`
- `backend_type`
- `sampler_strategy`

示例：

- `sdxl_inpaint` -> `backend_type = sdxl_native`
- `flux_edit` -> `backend_type = flux_edit`

Base profile 不会再模糊地进入 Pro 链：

- `SDXL` / `FLUX` / `SD1.5` base profile 仍然保持官方 base profile
- base profiles automatically fall back to a generic local repaint path
- 原生 edit/inpaint 模型才会走 native path

### Compatibility Metadata

- `model_role`
- `supports_inpaint_native`
- `supports_image_edit_native`
- `preferred_edit_backend`

这些字段可以来自：

- `model_info`
- `LLS Simple Checkpoint Loader` 写入到对象上的 `_lls_*` 元信息
- `utils/model_info.py` 里的 family / name 推断逻辑
- profile resolver 对旧 capability tags 的兼容升级

### Adding new professional edit models

1. 在 `model_profiles/` 里补充或修正 profile rule
2. 确认 `LLS Simple Checkpoint Loader` 会把正确的 `_lls_*` profile tags 写到 `model` / `clip` / `vae`
3. 优先用 `backend_mode=auto` 验证 resolved profile 是否足以自动识别
4. 调试阶段可通过 `model_info` 显式覆盖 `profile_id` / `backend_type` / `sampler_strategy`

### Notes

- `LLSSimple*` 修复链保持不变，不会被这套新链覆盖
- `LLS Pro Image Edit Finish` 负责真实的区域混合、裁剪回贴和扩图 canvas 合成
- `LLS Pro KSampler Bridge` 支持 `denoise_mode = manual | auto_from_edit`

---

## 扩展节点

在对应子包的 `nodes.py` 中添加节点类并注册到 `NODE_CLASS_MAPPINGS`，重启 ComfyUI 自动生效。

```python
# 示例：在 sampling/nodes.py 中添加节点
class LLSKSampler:
    CATEGORY = "LLS/Sampling"
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT",)

    @classmethod
    def INPUT_TYPES(cls):
        return { "required": { ... } }

    def sample(self, ...):
        return (latent,)

NODE_CLASS_MAPPINGS = { "LLSKSampler": LLSKSampler }
NODE_DISPLAY_NAME_MAPPINGS = { "LLSKSampler": "LLS KSampler" }
```

新增功能域只需新建子目录 + `__init__.py`，并将目录名追加到顶层 `__init__.py` 的 `_SUBPACKAGES` 列表。

## 语法检查

```bash
python -m py_compile __init__.py nodes.py upscale/nodes.py utils/nodes.py
```

## 节点命名规范

| 范围 | 格式 | 示例 |
|------|------|------|
| CATEGORY（UI 分组） | `LLS/<域名>` | `LLS/Sampling` |
| 类名 | `LLS` + PascalCase | `LLSKSampler` |
| 显示名称（搜索） | `LLS ` + 描述 | `LLS KSampler` |
| 注册 Key | 与类名相同 | `"LLSKSampler"` |
