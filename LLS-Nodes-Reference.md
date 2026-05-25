# LLS-Node 节点参考手册

> LLS ComfyUI 插件全部节点及参数说明文档。  
> 核心设计：移除 `task_context` 端口，从 MODEL/CLIP/VAE 等标准对象自动推导模型家族与任务模式。

---

## 目录

1. [模型加载 (Model Loader)](#1-模型加载-model-loader)
2. [文本编码 (Conditioning)](#2-文本编码-conditioning)
3. [采样与去噪 (Sampling)](#3-采样与去噪-sampling)
4. [Qwen 一体化节点](#4-qwen-一体化节点)
5. [Latent 空间操作](#5-latent-空间操作)
6. [图像处理 (Image)](#6-图像处理-image)
7. [图像修复 (Repair)](#7-图像修复-repair)
8. [交互式遮罩绘制 (Mask Draw)](#8-交互式遮罩绘制-mask-draw)
9. [专业图像编辑 (Pro Edit)](#9-专业图像编辑-pro-edit)
10. [图像超分 (Upscale)](#10-图像超分-upscale)
11. [遮罩操作 (Mask)](#11-遮罩操作-mask)
12. [工具节点 (Utils)](#12-工具节点-utils)
13. [根级统一节点](#13-根级统一节点)
14. [待实现功能域](#14-待实现功能域)

---

## 1. 模型加载 (Model Loader)

分类：`LLS/Model Loader`

### LLS Simple Checkpoint Loader

统一封装的基础模型加载器，支持 SD1.5 / SDXL / SDXL Turbo / FLUX 全家族。输出同时保留旧工作流兼容的 `clip` 端口和新的 `text_encoder` 别名。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| model | MODEL | 扩散模型 |
| clip | CLIP | 原生 CLIP（旧工作流兼容） |
| vae | VAE | VAE 解码器 |
| text_encoder | CLIP | 同 clip，新命名别名 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| ckpt_name | 下拉列表 | — | 模型文件名，从 ComfyUI 的 checkpoints 和 diffusion_models 目录自动扫描 |
| model_family | 下拉列表 | Auto | 模型家族：`Auto` / `SD1.5` / `SDXL` / `SDXL_TURBO` / `FLUX_DEV` / `FLUX_SDXL`。Auto 时从模型文件名自动推断 |
| load_mode | 下拉列表 | simple | 加载模式：`simple`（简化）/ `advanced`（高级，暴露更多选项） |
| vae_source | 下拉列表 | auto | VAE 来源：`auto`（优先内嵌）/ `embedded`（仅用内嵌）/ `external`（仅用外部）/ `none`（不加载 VAE） |
| text_encoder_source | 下拉列表 | auto | 文本编码器来源：`auto` / `embedded` / `external` / `manual` |
| external_vae_name | 下拉列表 | (auto) | 外部 VAE 文件名，auto 时自动匹配 |
| external_text_encoder_1 | 下拉列表 | (auto) | 外部文本编码器 1（SD1.5: CLIP-L，SDXL: CLIP-L，FLUX: CLIP-L） |
| external_text_encoder_2 | 下拉列表 | (auto) | 外部文本编码器 2（SDXL: CLIP-G，FLUX: T5XXL） |

---

### LLS Universal Model Loader

统一模型加载器，始终输出单一 `text_encoder`，额外输出 `model_info` JSON 串供下游节点使用。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| model | MODEL | 扩散模型 |
| text_encoder | CLIP | 统一文本编码器（内部处理单/双编码器差异） |
| vae | VAE | VAE 解码器 |
| model_info | STRING | JSON 格式模型元信息（家族、profile、能力标签等） |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| model_name | 下拉列表 | — | 模型文件名 |
| model_family | 下拉列表 | Auto | 模型家族选择 |
| load_mode | 下拉列表 | simple | 加载模式 |
| vae_source | 下拉列表 | auto | VAE 来源：`auto` / `embedded` / `external` |
| text_encoder_source | 下拉列表 | auto | 文本编码器来源：`auto` / `embedded` / `external` |
| text_encoder_1 | 下拉列表 | (auto) | 外部文本编码器 1 |
| text_encoder_2 | 下拉列表 | (auto) | 外部文本编码器 2 |
| vae_name | 下拉列表 | (auto) | 外部 VAE 文件名 |

---

## 2. 文本编码 (Conditioning)

分类：`LLS/Conditioning`

### LLS Simple Prompt Encode

家族感知的提示词编码节点。自动根据模型家族选择编码路径（SD1.5 CLIP / SDXL 双编码器 / FLUX CLIP-L+T5）。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| positive | CONDITIONING | 正向条件 |
| negative | CONDITIONING | 反向条件 |
| prompt_info | STRING | JSON 格式编码元信息 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| positive_prompt | STRING（多行） | "" | 正向提示词 |
| negative_prompt | STRING（多行） | "" | 反向提示词 |
| clip_skip | 下拉列表 | -1 | CLIP 跳层：-1 表示不跳层，-2 跳最后一层，依此类推。FLUX 家族强制 -1 |
| model_family | 下拉列表 | Auto | 模型家族选择 |

**可选输入**

| 参数名 | 类型 | 说明 |
|--------|------|------|
| text_encoder | CLIP | 文本编码器（新端口） |
| clip | CLIP | CLIP（旧端口兼容） |

> text_encoder 和 clip 至少连接一个。

---

### LLS Universal Prompt Encode

统一文本编码节点，只公开一个 `text_encoder` 输入，可选接收 `model_info` 来明确家族信息。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| positive | CONDITIONING | 正向条件 |
| negative | CONDITIONING | 反向条件 |
| prompt_info | STRING | JSON 格式编码元信息 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| text_encoder | CLIP | — | 文本编码器（必接） |
| positive_prompt | STRING（多行） | "" | 正向提示词 |
| negative_prompt | STRING（多行） | "" | 反向提示词 |
| clip_skip | 下拉列表 | -1 | CLIP 跳层数 |

**可选输入**

| 参数名 | 类型 | 说明 |
|--------|------|------|
| model_info | STRING | 模型元信息 JSON（来自 Universal Model Loader） |

---

## 3. 采样与去噪 (Sampling)

分类：`LLS/Sampling`

### LLS Simple KSampler

简化版 KSampler。内部复用 ComfyUI 原生采样能力，支持 `quality_preset` 一键切换采样参数，并输出 `sample_info` 元数据。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| latent | LATENT | 采样后的潜空间 |
| sample_info | STRING | JSON 格式采样元信息（seed/steps/cfg/denoise/家族等） |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| model | MODEL | — | 扩散模型 |
| positive | CONDITIONING | — | 正向条件 |
| negative | CONDITIONING | — | 反向条件 |
| latent_image | LATENT | — | 输入潜空间 |
| quality_preset | 下拉列表 | Family Default | 质量预设：`Family Default`（家族默认）/ `Manual`（手动）/ `Fast` / `Balanced` / `High Quality`。非 Manual 时覆盖 steps/cfg/sampler/scheduler/denoise |
| seed | INT | -1 | 随机种子。-1 时自动随机生成 |
| steps | INT | 20 | 采样步数（1–10000） |
| cfg | FLOAT | 7.0 | CFG 引导强度（0.0–100.0） |
| sampler_name | 下拉列表 | euler_ancestral | 采样器名称 |
| scheduler | 下拉列表 | karras | 调度器名称 |
| denoise | FLOAT | 1.0 | 去噪强度（0.0–1.0） |
| denoise_mode | 下拉列表 | manual | 去噪模式：`manual`（手动）/ `auto_from_repair`（从修复信息自动推荐） |
| adapter_mode | 下拉列表 | auto | 适配器模式：`auto` / `sd_classic` / `flux` / `sd3` / `qwen` / `zimage` |
| flux_guidance | FLOAT | 3.5 | FLUX 家族 guidance 值（0.0–100.0），仅 FLUX 时生效 |
| model_family | 下拉列表 | Auto | 模型家族选择 |

**可选输入**

| 参数名 | 类型 | 说明 |
|--------|------|------|
| repair_info | LLS_REPAIR_INFO | 修复信息（来自 Repair Prepare） |
| guidance_stack | LLS_GUIDANCE_STACK | 引导栈 |
| model_info | STRING | 模型元信息 JSON |

---

## 4. Qwen 一体化节点

分类：`LLS/Qwen`

### LLS Qwen Text To Image

Qwen 文生图一体化节点。内部自动加载 Qwen 模型资源，支持高级采样控制和可选 Turbo LoRA。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| image | IMAGE | 生成的图像 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| model_name | 下拉列表 | — | Qwen 文生图模型名称 |
| prompt | STRING（多行） | "" | 提示词 |
| width | INT | 1024 | 图像宽度（16–8192，步长16） |
| height | INT | 1024 | 图像高度（16–8192，步长16） |
| steps | INT | 20 | 采样步数 |
| seed | INT | 0 | 随机种子（0–最大值） |
| batch_size | INT | 1 | 批量大小（1–64） |
| negative_prompt | STRING（多行） | "" | 反向提示词（高级参数） |
| cfg | FLOAT | 4.0 | CFG 引导强度（高级参数） |
| sampler_name | 下拉列表 | euler | 采样器（高级参数） |
| scheduler | 下拉列表 | simple | 调度器（高级参数） |
| shift | FLOAT | 3.1 | 采样偏移量（高级参数，0.0–100.0） |
| enable_turbo_mode | BOOLEAN | False | 启用 Turbo 加速模式（高级参数） |
| turbo_lora_name | 下拉列表 | (auto) | Turbo LoRA 模型名称（高级参数） |
| turbo_strength | FLOAT | 1.0 | Turbo LoRA 强度（高级参数，-100.0–100.0） |

**可选输入**

| 参数名 | 类型 | 说明 |
|--------|------|------|
| model | MODEL | 外部模型（可选链式连接） |

---

### LLS Qwen Image Edit

Qwen 图像编辑节点。支持多图编辑条件、可选 Turbo LoRA。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| image | IMAGE | 编辑后的图像 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| model_name | 下拉列表 | — | Qwen 编辑模型名称 |
| image | IMAGE | — | 输入图像 |
| prompt | STRING（多行） | "" | 编辑提示词 |
| steps | INT | 20 | 采样步数 |
| seed | INT | 0 | 随机种子 |
| negative_prompt | STRING（多行） | "" | 反向提示词（高级参数） |
| cfg | FLOAT | 4.0 | CFG 引导强度（高级参数） |
| sampler_name | 下拉列表 | euler | 采样器（高级参数） |
| scheduler | 下拉列表 | simple | 调度器（高级参数） |
| shift | FLOAT | 3.1 | 采样偏移量（高级参数） |
| cfg_norm_strength | FLOAT | 1.0 | CFG 归一化强度（高级参数，0.0–100.0） |
| reference_latents_method | 下拉列表 | index_timestep_zero | 参考潜空间方法（高级参数） |
| enable_turbo_mode | BOOLEAN | False | 启用 Turbo 加速（高级参数） |
| turbo_lora_name | 下拉列表 | (auto) | Turbo LoRA（高级参数） |
| turbo_strength | FLOAT | 1.0 | Turbo LoRA 强度（高级参数） |

**可选输入**

| 参数名 | 类型 | 说明 |
|--------|------|------|
| image2 | IMAGE | 第二张参考图（多图编辑） |
| image3 | IMAGE | 第三张参考图（多图编辑） |
| model | MODEL | 外部模型 |

---

## 5. Latent 空间操作

分类：`LLS/Latent`

### LLS Simple Empty Latent

统一的 Latent 入口节点。未连接 `image` 时生成空白 Latent（txt2img），连接 `image` 时通过 VAE Encode 生成 img2img Latent。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| latent | LATENT | 潜空间张量 |
| width | INT | 实际宽度 |
| height | INT | 实际高度 |
| latent_info | STRING | JSON 格式元信息 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| size_preset | 下拉列表 | Family Default | 尺寸预设：`Family Default`（家族默认）/ `Custom` / `512x512` / `768x768` / `512x768` / `768x512` / `1024x1024` / `832x1216` / `1216x832` / `896x1152` / `1152x896` / `1024x576` / `576x1024` |
| width | INT | 512 | 宽度（Custom 时生效，64–8192，步长8） |
| height | INT | 512 | 高度（Custom 时生效，64–8192，步长8） |
| batch_size | INT | 1 | 批量大小（1–64） |
| model_family | 下拉列表 | Auto | 模型家族选择 |
| resize_mode | 下拉列表 | keep_aspect | 缩放模式：`keep_aspect`（保持比例）/ `crop_center`（居中裁剪）/ `stretch`（拉伸）/ `none`（不缩放） |

**可选输入**

| 参数名 | 类型 | 说明 |
|--------|------|------|
| model | MODEL | 扩散模型（用于推断家族） |
| image | IMAGE | 输入图像（连接后切换为 img2img 模式） |
| vae | VAE | VAE（img2img 时必需） |

---

## 6. 图像处理 (Image)

分类：`LLS/Image`

### LLS Simple VAE Encode

将 IMAGE 编码为 LATENT，供 img2img 使用。内部复用 ComfyUI 原生 `vae.encode`，外层补尺寸处理与元信息。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| latent | LATENT | 编码后的潜空间 |
| width | INT | 实际宽度 |
| height | INT | 实际高度 |
| latent_info | STRING | JSON 格式元信息 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| image | IMAGE | — | 输入图像 |
| vae | VAE | — | VAE 解码器 |
| resize_mode | 下拉列表 | keep_aspect | 缩放模式：`keep_aspect` / `crop_center` / `stretch` / `none` |
| size_source | 下拉列表 | input_image | 尺寸来源：`input_image`（原图尺寸）/ `custom`（自定义）/ `model_recommended`（模型推荐） |
| width | INT | 512 | 自定义宽度 |
| height | INT | 512 | 自定义高度 |
| model_family | 下拉列表 | Auto | 模型家族 |

**可选输入**

| 参数名 | 类型 | 说明 |
|--------|------|------|
| model | MODEL | 扩散模型 |
| clip | CLIP | CLIP 编码器 |

---

### LLS Simple VAE Decode

将 LATENT 解码为 IMAGE。内部复用 ComfyUI 原生 `vae.decode`。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| image | IMAGE | 解码后的图像 |
| decode_info | STRING | JSON 格式解码元信息 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| samples | LATENT | — | 潜空间张量 |
| vae | VAE | — | VAE 解码器 |

---

### LLS Save Image

保存图像，并将 LLS 生成链路信息合并进 PNG metadata。支持同时保存图像和遮罩。

**输出端口**：无（终端节点）

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| filename_prefix | STRING | LLS | 文件名前缀 |
| output_mode | 下拉列表 | save | 输出模式：`save`（保存到磁盘）/ `preview_only`（仅预览） |
| save_metadata | BOOLEAN | True | 是否保存 LLS 元数据到 PNG |

**可选输入**

| 参数名 | 类型 | 说明 |
|--------|------|------|
| image | IMAGE | 要保存的图像（与 mask 至少接一个） |
| mask | MASK | 要保存的遮罩 |
| prompt_info | STRING | 提示词元信息 |
| latent_info | STRING | 潜空间元信息 |
| sample_info | STRING | 采样元信息 |
| decode_info | STRING | 解码元信息 |
| upscale_info | STRING | 超分元信息 |

---

### LLS Simple Image Composite

将叠加图像合成到背景图像上，支持平移、缩放、旋转和透明度控制。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| output_image | IMAGE | 合成后的图像 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| background_image | IMAGE | — | 背景图像 |
| overlay_image | IMAGE | — | 叠加图像 |
| x_offset | INT | 0 | X 轴偏移（-8192–8192） |
| y_offset | INT | 0 | Y 轴偏移（-8192–8192） |
| anchor_mode | 下拉列表 | top_left | 锚点模式：`top_left`（左上角为基准）/ `center`（中心点为基准） |
| rotation_origin_mode | 下拉列表 | center | 旋转原点模式：`top_left` / `center` |
| opacity | FLOAT | 1.0 | 不透明度（0.0–1.0） |
| blend_mode | 下拉列表 | normal | 混合模式：`normal` |
| scale | FLOAT | 1.0 | 缩放比例（0.01–32.0） |
| rotation | FLOAT | 0.0 | 旋转角度（-360.0–360.0 度） |
| keep_aspect | BOOLEAN | True | 是否保持宽高比 |

---

## 7. 图像修复 (Repair)

分类：`LLS/Image Repair`

### LLS Simple Repair Prepare

修复流程的"准备"节点。处理输入图像和遮罩，确定修复范围（region/crop/canvas）、修复核心策略，生成工作区 Latent 和元数据。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| latent | LATENT | 工作区潜空间 |
| work_image | IMAGE | 工作区图像 |
| work_mask | MASK | 工作区遮罩 |
| repair_info | LLS_REPAIR_INFO | 修复元数据 |
| recommended_denoise | FLOAT | 推荐去噪强度 |
| positive | CONDITIONING | 正向条件 |
| negative | CONDITIONING | 反向条件 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| image | IMAGE | — | 原始图像 |
| mask | MASK | — | 修复遮罩 |
| vae | VAE | — | VAE 解码器 |
| repair_scope | 下拉列表 | auto | 修复范围：`auto` / `region`（全图区域）/ `crop`（裁剪局部）/ `canvas`（画布扩展） |
| repair_kernel | 下拉列表 | auto | 修复核心：`auto` / `latent_mask` / `vae_inpaint` / `native_fill` |
| task_hint | 下拉列表 | auto | 任务提示：`auto` / `repair` / `remove` / `replace` / `fill` / `appearance` / `content` / `structure` / `dehaze` / `deshadow` / `recolor` |
| mask_grow | INT | 24 | 遮罩扩展像素数（0–2048） |
| mask_blur | FLOAT | 8.0 | 遮罩模糊半径（0.0–256.0） |
| mask_threshold | FLOAT | 0.5 | 遮罩二值化阈值（0.0–1.0） |
| invert_mask | BOOLEAN | False | 是否反转遮罩 |
| crop_context | INT | 64 | 裁剪额外上下文边距（0–512 像素） |
| crop_context_factor | FLOAT | 1.5 | 裁剪上下文缩放因子（1.0–8.0） |
| min_size | INT | 256 | 最小工作区尺寸（64–8192，步长8） |
| max_size | INT | 1024 | 最大工作区尺寸（64–8192，步长8） |
| resize_mode | 下拉列表 | fit | 缩放模式：`fit` / `pad` / `stretch` |
| expand_left | INT | 0 | 画布左扩像素（0–4096） |
| expand_right | INT | 0 | 画布右扩像素 |
| expand_top | INT | 0 | 画布上扩像素 |
| expand_bottom | INT | 0 | 画布下扩像素 |
| canvas_fill | 下拉列表 | edge | 画布填充模式：`edge`（边缘延展）/ `blur` / `black` / `white` / `neutral` |
| auto_recommend | 下拉列表 | enabled | 自动推荐去噪值：`enabled` / `disabled` |

**可选输入**

| 参数名 | 类型 | 说明 |
|--------|------|------|
| model | MODEL | 扩散模型 |
| model_info | STRING | 模型元信息 |
| positive | CONDITIONING | 正向条件 |
| negative | CONDITIONING | 反向条件 |

---

### LLS Simple Repair Finish

修复流程的"合成"节点。将修复结果合成回原始图像，支持羽化、颜色匹配、亮度匹配和边缘修复。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| final_image | IMAGE | 最终合成图像 |
| preview_image | IMAGE | 预览图像 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| original_image | IMAGE | — | 原始图像 |
| generated_image | IMAGE | — | 修复后的图像 |
| repair_info | LLS_REPAIR_INFO | — | 修复元数据（来自 Repair Prepare） |
| feather | FLOAT | 8.0 | 羽化半径（0.0–256.0） |
| color_match | 下拉列表 | disabled | 颜色匹配：`disabled` / `mean_std`（均值标准差匹配）/ `histogram_simple`（直方图匹配） |
| brightness_match | 下拉列表 | enabled | 亮度匹配：`disabled` / `enabled` |
| blend_strength | FLOAT | 1.0 | 混合强度（0.0–1.0） |
| restore_unmasked_area | BOOLEAN | True | 是否恢复未遮罩区域 |
| edge_fix | 下拉列表 | soft | 边缘修复：`none` / `soft` / `strong` |
| preview_mode | 下拉列表 | final | 预览模式：`final` / `compare` / `mask` / `before_after` |

**可选输入**

| 参数名 | 类型 | 说明 |
|--------|------|------|
| work_mask | MASK | 工作区遮罩 |
| sample_info | STRING | 采样元信息 |

---

### LLS Native Inpaint Conditioning

ComfyUI 原生 InpaintModelConditioning 的薄封装。可自动为 FLUX 家族应用 DifferentialDiffusion 模型补丁。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| model | MODEL | 可能经过补丁的模型 |
| positive | CONDITIONING | 正向条件 |
| negative | CONDITIONING | 反向条件 |
| latent | LATENT | 编码后的潜空间 |
| inpaint_info | STRING | JSON 格式修复信息 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| model | MODEL | — | 扩散模型 |
| positive | CONDITIONING | — | 正向条件 |
| negative | CONDITIONING | — | 反向条件 |
| vae | VAE | — | VAE 解码器 |
| image | IMAGE | — | 原始图像 |
| mask | MASK | — | 修复遮罩 |
| patch_mode | 下拉列表 | auto | 模型补丁模式：`auto`（FLUX 自动应用）/ `disabled` / `differential_diffusion` |
| patch_strength | FLOAT | 1.0 | 补丁强度（0.0–1.0） |
| noise_mask | BOOLEAN | True | 是否使用噪声遮罩 |

**可选输入**

| 参数名 | 类型 | 说明 |
|--------|------|------|
| model_info | STRING | 模型元信息 |

---

## 8. 交互式遮罩绘制 (Mask Draw)

分类：`LLS/Image Repair`

### LLS Simple Mask Draw

交互式遮罩绘制节点。前端 JS 配合实现在画布上绘制/擦除遮罩，后端解析绘制状态并输出遮罩和预览。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| image | IMAGE | 原始图像（透传） |
| mask | MASK | 最终遮罩 |
| preview_image | IMAGE | 遮罩叠加预览图 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| image | IMAGE | — | 原始图像（同时作为绘制画布背景） |
| draw_mode | 下拉列表 | brush | 绘制模式：`brush`（画笔）/ `erase`（擦除） |
| brush_size | INT | 32 | 画笔大小（1–512） |
| brush_softness | FLOAT | 0.5 | 画笔柔和度（0.0–1.0） |
| overlay_alpha | FLOAT | 0.4 | 预览叠加透明度（0.0–1.0） |
| invert_mask | BOOLEAN | False | 是否反转遮罩 |
| mask_state_json | STRING | {} | 遮罩绘制状态 JSON（前端自动维护，高级参数） |

**可选输入**

| 参数名 | 类型 | 说明 |
|--------|------|------|
| input_mask | MASK | 初始遮罩（可在此基础继续编辑） |

---

## 9. 专业图像编辑 (Pro Edit)

分类：`LLS/Image Edit`

### LLS Pro Image Edit Prepare

专业图像编辑的"准备"节点。与 Repair Prepare 类似，但面向更通用的图像编辑场景（如局部替换、外观修改），支持自动后端路由。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| latent | LATENT | 工作区潜空间 |
| work_image | IMAGE | 工作区图像 |
| work_mask | MASK | 工作区遮罩 |
| edit_info | LLS_EDIT_INFO | 编辑元数据 |
| recommended_denoise | FLOAT | 推荐去噪强度 |
| positive | CONDITIONING | 正向条件 |
| negative | CONDITIONING | 反向条件 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| image | IMAGE | — | 原始图像 |
| mask | MASK | — | 编辑遮罩 |
| vae | VAE | — | VAE 解码器 |
| positive | CONDITIONING | — | 正向条件 |
| negative | CONDITIONING | — | 反向条件 |
| backend_mode | 下拉列表 | auto | 后端模式：`auto` / `sdxl` / `flux` |
| edit_scope | 下拉列表 | auto | 编辑范围：`auto` / `region` / `crop` / `canvas` |
| mask_grow | INT | 24 | 遮罩扩展（0–2048） |
| mask_blur | FLOAT | 8.0 | 遮罩模糊（0.0–256.0） |
| mask_threshold | FLOAT | 0.5 | 遮罩阈值 |
| invert_mask | BOOLEAN | False | 反转遮罩 |
| crop_context | INT | 64 | 裁剪上下文边距 |
| crop_context_factor | FLOAT | 1.5 | 裁剪上下文因子 |
| min_size | INT | 256 | 最小尺寸 |
| max_size | INT | 1024 | 最大尺寸 |
| resize_mode | 下拉列表 | fit | 缩放模式：`fit` / `pad` / `stretch` |
| expand_left | INT | 0 | 左扩像素 |
| expand_right | INT | 0 | 右扩像素 |
| expand_top | INT | 0 | 上扩像素 |
| expand_bottom | INT | 0 | 下扩像素 |
| canvas_fill | 下拉列表 | edge | 画布填充：`edge` / `blur` / `black` / `white` / `neutral` |
| auto_recommend | 下拉列表 | enabled | 自动推荐：`enabled` / `disabled` |

**可选输入**

| 参数名 | 类型 | 说明 |
|--------|------|------|
| model | MODEL | 扩散模型 |
| model_info | STRING | 模型元信息 |

---

### LLS Pro KSampler Bridge

专业编辑流程的高级采样器桥接节点，支持分步采样控制（start_at_step / end_at_step）。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| latent | LATENT | 采样结果 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| model | MODEL | — | 扩散模型 |
| add_noise | 下拉列表 | enable | 是否添加噪声：`enable` / `disable`（高级参数） |
| noise_seed | INT | 0 | 噪声种子 |
| steps | INT | 20 | 采样步数 |
| cfg | FLOAT | 8.0 | CFG 引导强度 |
| sampler_name | 下拉列表 | — | 采样器 |
| scheduler | 下拉列表 | — | 调度器 |
| positive | CONDITIONING | — | 正向条件 |
| negative | CONDITIONING | — | 反向条件 |
| latent_image | LATENT | — | 输入潜空间 |
| start_at_step | INT | 0 | 起始步（高级参数，0–10000） |
| end_at_step | INT | 10000 | 结束步（高级参数，0–10000） |
| return_with_leftover_noise | 下拉列表 | disable | 返回残余噪声：`disable` / `enable`（高级参数） |

---

### LLS Pro Image Edit Finish

专业图像编辑的"合成"节点。与 Repair Finish 类似，处理编辑结果的合成回贴。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| final_image | IMAGE | 最终合成图像 |
| preview_image | IMAGE | 预览图像 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| original_image | IMAGE | — | 原始图像 |
| generated_image | IMAGE | — | 编辑后图像 |
| edit_info | LLS_EDIT_INFO | — | 编辑元数据（来自 Pro Edit Prepare） |
| feather | FLOAT | 8.0 | 羽化半径 |
| color_match | 下拉列表 | disabled | 颜色匹配：`disabled` / `mean_std` / `histogram_simple` |
| brightness_match | 下拉列表 | enabled | 亮度匹配：`disabled` / `enabled` |
| blend_strength | FLOAT | 1.0 | 混合强度 |
| restore_unmasked_area | BOOLEAN | True | 恢复未遮罩区域 |
| edge_fix | 下拉列表 | soft | 边缘修复：`none` / `soft` / `strong` |
| preview_mode | 下拉列表 | final | 预览模式：`final` / `compare` / `mask` / `before_after` |

**可选输入**

| 参数名 | 类型 | 说明 |
|--------|------|------|
| work_mask | MASK | 工作区遮罩 |
| sample_info | STRING | 采样元信息 |

---

## 10. 图像超分 (Upscale)

分类：`LLS/Upscale`

### LLS Upscale Switcher

超分辨率切换节点，可在 ESRGAN 等超分模型和 PyTorch 插值之间切换。超分模型模式支持分块推理防止 OOM。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| image | IMAGE | 超分后的图像 |
| upscale_info | STRING | JSON 格式超分元信息 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| image | IMAGE | — | 输入图像 |
| mode | 下拉列表 | upscale_model | 超分模式：`none`（不放大）/ `interpolation`（PyTorch 插值）/ `upscale_model`（超分模型）/ `latent_upscale`（潜空间超分，回退到插值）/ `tile_upscale`（分块超分）/ `pytorch`（同 interpolation） |
| scale | FLOAT | 2.0 | 放大倍率（0.1–8.0） |
| interpolation | 下拉列表 | bilinear | 插值方法：`nearest` / `bilinear` / `bicubic` / `area` |
| model_name | 下拉列表 | — | 超分模型文件名 |
| tile | INT | 512 | 分块大小（128–2048，步长64） |
| overlap | INT | 32 | 分块重叠（0–256，步长8） |

---

## 11. 遮罩操作 (Mask)

分类：`LLS/Mask`

### LLS Simple Mask Create

几何遮罩创建节点。通过指定宽高、形状、位置等参数创建矩形/圆形/椭圆遮罩。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| mask_image | IMAGE | 遮罩可视化图像 |
| mask | MASK | 遮罩张量 |
| area_info | STRING | JSON 格式遮罩区域信息 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| image_width | INT | 1024 | 图像宽度（1–8192） |
| image_height | INT | 1024 | 图像高度（1–8192） |
| shape_type | 下拉列表 | rectangle | 形状类型：`rectangle` / `ellipse` / `circle` 等 |
| coordinate_mode | 下拉列表 | percent | 坐标模式：`percent`（百分比）/ `pixel`（像素） |
| center_x | FLOAT | 0.5 | 中心 X 坐标 |
| center_y | FLOAT | 0.5 | 中心 Y 坐标 |
| width | FLOAT | 0.3 | 形状宽度 |
| height | FLOAT | 0.3 | 形状高度 |
| radius | FLOAT | 0.15 | 圆形半径 |
| feather | FLOAT | 0.0 | 羽化强度（0.0–128.0） |
| blur | FLOAT | 0.0 | 模糊强度（0.0–128.0） |
| invert_mask | BOOLEAN | False | 是否反转遮罩 |
| combine_mode | 下拉列表 | replace | 与输入遮罩的合并模式：`replace` / `add` / `subtract` / `intersect` 等 |

**可选输入**

| 参数名 | 类型 | 说明 |
|--------|------|------|
| input_mask | MASK | 输入遮罩（在 replace 以外的模式下与此合并） |

---

### LLS Mask Process

单遮罩处理节点，支持多种常见操作。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| mask | MASK | 处理后的遮罩 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| mask | MASK | — | 输入遮罩 |
| operation | 下拉列表 | passthrough | 处理操作：`passthrough`（透传）/ `threshold`（二值化）/ `invert`（反转）/ `grow`（扩展）/ `shrink`（收缩）/ `blur`（模糊）/ `feather`（羽化）/ `fill_holes`（填充孔洞）/ `remove_small_regions`（去除小区域）/ `smooth`（平滑）/ `clamp`（裁剪）/ `resize_to_image`（缩放至图像尺寸） |
| value_float | FLOAT | 0.5 | 浮点参数（threshold 时为阈值，0.0–1.0） |
| value_int | INT | 8 | 整数参数（grow/shrink/blur 的像素值，-512–512） |

**可选输入**

| 参数名 | 类型 | 说明 |
|--------|------|------|
| image | IMAGE | 目标图像（resize_to_image 操作时使用） |

---

### LLS Mask Combine

双遮罩合并节点，支持逻辑运算风格的合并。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| mask | MASK | 合并后的遮罩 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| mask_a | MASK | — | 遮罩 A |
| mask_b | MASK | — | 遮罩 B |
| mode | 下拉列表 | add | 合并模式：`add`/`max`（取大）/ `subtract`（A减B）/ `intersect`/`min`（取小）/ `xor`（异或） |

---

## 12. 工具节点 (Utils)

分类：`LLS/Utils`

### LLS String Literal

输出一个字符串常量，可复用于多个下游节点。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| string | STRING | 字符串值 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| value | STRING（多行） | "" | 字符串内容 |

---

### LLS Int Literal

输出一个整数常量。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| int | INT | 整数值 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| value | INT | 0 | 整数值（-2^31 – 2^31-1） |

---

### LLS Float Literal

输出一个浮点数常量。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| float | FLOAT | 浮点数值 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| value | FLOAT | 0.0 | 浮点数值 |

---

### LLS Resolution Selector

常用分辨率选择器，返回宽度和高度。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| width | INT | 宽度 |
| height | INT | 高度 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| preset | 下拉列表 | 1024x1024 | 预设分辨率：`Custom` / `512x512` / `768x768` / `1024x1024` / `512x768` / `768x512` / `768x1024` / `1024x768` / `1024x576` / `576x1024` / `1280x720` / `720x1280` / `1920x1080` / `1080x1920` / `2048x2048` |
| custom_width | INT | 1024 | 自定义宽度（64–8192，步长8） |
| custom_height | INT | 1024 | 自定义高度（64–8192，步长8） |

---

### LLS Generation Config

按模型家族输出推荐分辨率与采样配置。根据 quality_preset 和 size_preset 自动生成最佳参数。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| width | INT | 推荐宽度 |
| height | INT | 推荐高度 |
| steps | INT | 推荐步数 |
| cfg | FLOAT | 推荐 CFG |
| guidance | FLOAT | 推荐 guidance（FLUX 用） |
| sampler_name | STRING | 推荐采样器 |
| scheduler | STRING | 推荐调度器 |
| denoise | FLOAT | 推荐去噪强度 |
| config_info | STRING | JSON 格式完整配置信息 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| quality_preset | 下拉列表 | Family Default | 质量预设：`Family Default` / `Manual` / `Fast` / `Balanced` / `High Quality` |
| size_preset | 下拉列表 | Family Default | 尺寸预设：`Family Default` / `Custom` / `512x512` / `768x768` / `1024x1024` |
| model_family | 下拉列表 | Auto | 模型家族 |

**可选输入**

| 参数名 | 类型 | 说明 |
|--------|------|------|
| model | MODEL | 扩散模型 |
| clip | CLIP | CLIP 编码器 |

---

## 13. 根级统一节点

### LLS Universal Image Generator

根级统一 txt2img 入口节点。收集请求并交给 dispatcher 选择对应家族后端适配器（SD1.5 / SDXL / FLUX）。

**输出端口**

| 端口名 | 类型 | 说明 |
|--------|------|------|
| image | IMAGE | 生成的图像 |

**输入参数**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| model_family | 下拉列表 | SD1.5 | 模型家族：`SD1.5` / `SDXL` / `FLUX` |
| task_mode | 下拉列表 | txt2img | 任务模式 |
| model_name | 下拉列表 | — | 模型文件名 |
| positive_prompt | STRING（多行） | "" | 正向提示词 |
| negative_prompt | STRING（多行） | "" | 反向提示词 |
| width | INT | 1024 | 图像宽度（64–8192，步长8） |
| height | INT | 1024 | 图像高度 |
| steps | INT | 20 | 采样步数 |
| cfg | FLOAT | 7.0 | CFG 引导强度 |
| seed | INT | -1 | 随机种子（-1 自动生成） |
| sampler_name | 下拉列表 | euler | 采样器 |
| scheduler | 下拉列表 | normal | 调度器 |
| denoise | FLOAT | 1.0 | 去噪强度 |

---

## 14. 待实现功能域

以下子包已创建框架，但节点尚未实现（`NODE_CLASS_MAPPINGS` 为空）：

### ControlNet (`LLS/ControlNet`)

计划节点：
- LLSControlNetApply / LLSControlNetApplyAdv
- LLST2IAdapterApply / LLSControlNetStack
- LLSCannyPreprocess / LLSDepthPreprocess / LLSHEDPreprocess
- LLSLotusDepth / LLSLotusNormal

### LoRA (`LLS/LoRA`)

计划节点：
- LLSLoRALoader / LLSLoRAApply / LLSLoRAStack / LLSLoRAStackApply
- LLSHypernetworkApply / LLSLoRAExtract
- LLSIPAdapterApply / LLSHookLoRA / LLSWeightAdapter

### Video (`LLS/Video`)

计划节点：
- LLSLoadVideoFrames / LLSSaveVideo / LLSFrameInterpolation
- LLSSVDSampler / LLSCogVideoXSampler / LLSHunyuanVideoSampler
- LLSWanVideoSampler / LLSLTXVideoSampler / LLSVideoUpscale / LLSCameraControl

### Audio (`LLS/Audio`)

计划节点：
- LLSLoadAudio / LLSSaveAudio / LLSAudioCrop / LLSAudioConcat
- LLSAudioResample / LLSAudioVolumeAdjust / LLSAudioBatch
- LLSAudioEncode / LLSCosyVoiceTTS / LLSHunyuanAudio / LLSSpectrogram

---

## 节点总览表

| # | 节点类名 | 显示名 | 分类 | 状态 |
|---|----------|--------|------|------|
| 1 | LLSSimpleCheckpointLoader | LLS Simple Checkpoint Loader | LLS/Model Loader | 已实现 |
| 2 | LLSUniversalModelLoader | LLS Universal Model Loader | LLS/Model Loader | 已实现 |
| 3 | LLSSimplePromptEncode | LLS Simple Prompt Encode | LLS/Conditioning | 已实现 |
| 4 | LLSUniversalPromptEncode | LLS Universal Prompt Encode | LLS/Conditioning | 已实现 |
| 5 | LLSSimpleKSampler | LLS Simple KSampler | LLS/Sampling | 已实现 |
| 6 | LLSQwenTextToImage | LLS Qwen Text To Image | LLS/Qwen | 已实现 |
| 7 | LLSQwenImageEdit | LLS Qwen Image Edit | LLS/Qwen | 已实现 |
| 8 | LLSSimpleEmptyLatent | LLS Simple Empty Latent | LLS/Latent | 已实现 |
| 9 | LLSSimpleVAEEncode | LLS Simple VAE Encode | LLS/Image | 已实现 |
| 10 | LLSSimpleVAEDecode | LLS Simple VAE Decode | LLS/Image | 已实现 |
| 11 | LLSSaveImage | LLS Save Image | LLS/Image | 已实现 |
| 12 | LLSSimpleImageComposite | LLS Simple Image Composite | LLS/Image | 已实现 |
| 13 | LLSSimpleRepairPrepare | LLS Simple Repair Prepare | LLS/Image Repair | 已实现 |
| 14 | LLSSimpleRepairFinish | LLS Simple Repair Finish | LLS/Image Repair | 已实现 |
| 15 | LLSNativeInpaintConditioning | LLS Native Inpaint Conditioning | LLS/Image Repair | 已实现 |
| 16 | LLSSimpleMaskDraw | LLS Simple Mask Draw | LLS/Image Repair | 已实现 |
| 17 | LLSProImageEditPrepare | LLS Pro Image Edit Prepare | LLS/Image Edit | 已实现 |
| 18 | LLSProKSamplerBridge | LLS Pro KSampler Bridge | LLS/Image Edit | 已实现 |
| 19 | LLSProImageEditFinish | LLS Pro Image Edit Finish | LLS/Image Edit | 已实现 |
| 20 | LLSUpscaleSwitcher | LLS Upscale Switcher | LLS/Upscale | 已实现 |
| 21 | LLSSimpleMaskCreate | LLS Simple Mask Create | LLS/Mask | 已实现 |
| 22 | LLSMaskProcess | LLS Mask Process | LLS/Mask | 已实现 |
| 23 | LLSMaskCombine | LLS Mask Combine | LLS/Mask | 已实现 |
| 24 | LLSStringLiteral | LLS String Literal | LLS/Utils | 已实现 |
| 25 | LLSIntLiteral | LLS Int Literal | LLS/Utils | 已实现 |
| 26 | LLSFloatLiteral | LLS Float Literal | LLS/Utils | 已实现 |
| 27 | LLSResolutionSelector | LLS Resolution Selector | LLS/Utils | 已实现 |
| 28 | LLSGenerationConfig | LLS Generation Config | LLS/Utils | 已实现 |
| 29 | LLSUniversalImageGenerator | LLS Universal Image Generator | LLS/Image | 已实现 |
