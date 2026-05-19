"""
LLS / ModelLoader
=================
功能域：模型加载与管理（对应功能分类总览第 1 节）

CATEGORY = "LLS/Model Loader"
"""
from __future__ import annotations

# ---------- 防御性导入 ----------

try:
    import folder_paths
except Exception as exc:
    folder_paths = None
    _FOLDER_PATHS_ERR = exc
else:
    _FOLDER_PATHS_ERR = None

try:
    import comfy.sd as comfy_sd
except Exception as exc:
    comfy_sd = None
    _COMFY_SD_ERR = exc
else:
    _COMFY_SD_ERR = None

# 注：comfy.utils 在本模块未使用，无需导入


# ---------- 工具函数 ----------

def _get_checkpoint_names() -> list[str]:
    if folder_paths is None:
        return ["(ComfyUI not available)"]
    try:
        names = folder_paths.get_filename_list("checkpoints")
        return names if names else ["(no checkpoints found)"]
    except Exception:
        return ["(no checkpoints found)"]


def _detect_family(ckpt_name: str, model_family: str) -> str:
    """根据 model_family 参数或文件名推断模型家族。"""
    if model_family in ("SD1.5", "SDXL"):
        return model_family
    # Auto：通过文件名关键词简单判断
    name_lower = ckpt_name.lower()
    if any(kw in name_lower for kw in ("sdxl", "_xl", "-xl", "xl_")):
        return "SDXL"
    return "SD1.5"


# ---------- 节点类 ----------

class LLSSimpleCheckpointLoader:
    """
    简化版 Checkpoint 加载器，支持 SD1.5 / SDXL 基础流程。
    自动识别模型家族，输出 MODEL / CLIP / VAE 以及模型信息字符串。
    内部复用 ComfyUI 原生 load_checkpoint_guess_config，已自动适配两种架构。
    """

    CATEGORY = "LLS/Model Loader"
    FUNCTION = "load_checkpoint"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "model_info")
    DESCRIPTION = "Load a SD1.5 or SDXL checkpoint. Outputs MODEL, CLIP, VAE and a model_info string."

    _FAMILY_CHOICES = ["Auto", "SD1.5", "SDXL"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (_get_checkpoint_names(),),
                "model_family": (cls._FAMILY_CHOICES, {"default": "Auto"}),
            }
        }

    def load_checkpoint(self, ckpt_name: str, model_family: str):
        if folder_paths is None:
            raise RuntimeError(
                "[LLS] folder_paths is not available. "
                "Make sure this node runs inside a ComfyUI environment."
            ) from _FOLDER_PATHS_ERR
        if comfy_sd is None:
            raise RuntimeError(
                "[LLS] comfy.sd is not available. "
                "Make sure this node runs inside a ComfyUI environment."
            ) from _COMFY_SD_ERR

        # 解析模型路径（使用 ComfyUI 原生方法，自动抛出文件不存在异常）
        try:
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        except (AttributeError, TypeError):
            # get_full_path_or_raise 在旧版 ComfyUI 中可能不存在，回退到 get_full_path
            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
            if not ckpt_path:
                raise RuntimeError(
                    f"[LLS] Checkpoint '{ckpt_name}' not found in ComfyUI checkpoints directories. "
                    f"Please add the file to models/checkpoints/ and restart ComfyUI."
                )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"[LLS] Checkpoint '{ckpt_name}' not found: {exc}"
            ) from exc

        # 加载 checkpoint（复用 ComfyUI 原生加载逻辑，自动适配 SD1.5 / SDXL）
        try:
            out = comfy_sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )
        except Exception as exc:
            raise RuntimeError(
                f"[LLS] Failed to load checkpoint '{ckpt_name}': {exc}"
            ) from exc

        model, clip, vae = out[0], out[1], out[2]

        if model is None:
            raise RuntimeError(f"[LLS] Checkpoint '{ckpt_name}' did not produce a valid MODEL.")
        if clip is None:
            raise RuntimeError(f"[LLS] Checkpoint '{ckpt_name}' did not produce a valid CLIP.")
        if vae is None:
            raise RuntimeError(f"[LLS] Checkpoint '{ckpt_name}' did not produce a valid VAE.")

        detected_family = _detect_family(ckpt_name, model_family)

        model_info = (
            f"ckpt={ckpt_name} | family={detected_family}"
        )

        return (model, clip, vae, model_info)


# ---------- 注册表 ----------

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "LLSSimpleCheckpointLoader": LLSSimpleCheckpointLoader,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LLSSimpleCheckpointLoader": "LLS Simple Checkpoint Loader",
}
