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
