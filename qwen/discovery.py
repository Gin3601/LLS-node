"""
Qwen model/resource discovery helpers.
"""
from __future__ import annotations

try:
    import folder_paths
except Exception:
    folder_paths = None


TEXT_MODEL_PLACEHOLDER = "(no qwen text-to-image models found)"
EDIT_MODEL_PLACEHOLDER = "(no qwen image edit models found)"

_QWEN_TEXT_ENCODER_PATTERNS = (
    "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "qwen_2.5_vl_7b",
    "qwen25_7b",
)
_QWEN_VAE_PATTERNS = (
    "qwen_image_vae.safetensors",
    "qwen_image_vae",
)


def _get_filename_list(category: str) -> list[str]:
    if folder_paths is None:
        return []
    try:
        return list(folder_paths.get_filename_list(category))
    except Exception:
        return []


def _sorted_unique(items: list[str]) -> list[str]:
    return sorted(dict.fromkeys(item for item in items if item))


def _match_by_patterns(names: list[str], patterns: tuple[str, ...]) -> str | None:
    lowered = {name.lower(): name for name in names}
    for pattern in patterns:
        pattern_lower = pattern.lower()
        for candidate_lower, candidate in lowered.items():
            if candidate_lower == pattern_lower or pattern_lower in candidate_lower:
                return candidate
    return None


def is_qwen_text_to_image_model(name: str | None) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return (
        "qwen" in lowered
        and "image" in lowered
        and "edit" not in lowered
        and "layered" not in lowered
    )


def is_qwen_image_edit_model(name: str | None) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return "qwen" in lowered and "image" in lowered and "edit" in lowered and "layered" not in lowered


def get_qwen_text_model_choices() -> list[str]:
    names = [name for name in _get_filename_list("diffusion_models") if is_qwen_text_to_image_model(name)]
    return _sorted_unique(names) or [TEXT_MODEL_PLACEHOLDER]


def get_qwen_edit_model_choices() -> list[str]:
    names = [name for name in _get_filename_list("diffusion_models") if is_qwen_image_edit_model(name)]
    return _sorted_unique(names) or [EDIT_MODEL_PLACEHOLDER]


def validate_qwen_text_model_name(model_name: str) -> str:
    if not is_qwen_text_to_image_model(model_name):
        raise RuntimeError(
            f"[LLS] Model '{model_name}' is not compatible with LLSQwenTextToImage."
        )
    return model_name


def validate_qwen_edit_model_name(model_name: str) -> str:
    if not is_qwen_image_edit_model(model_name):
        raise RuntimeError(
            f"[LLS] Model '{model_name}' is not compatible with LLSQwenImageEdit."
        )
    return model_name


def resolve_qwen_text_encoder_name() -> str | None:
    names = _get_filename_list("text_encoders")
    matched = _match_by_patterns(names, _QWEN_TEXT_ENCODER_PATTERNS)
    if matched is not None:
        return matched
    for name in names:
        lowered = name.lower()
        if "qwen" in lowered and ("vl" in lowered or "qwen25_7b" in lowered):
            return name
    return None


def resolve_qwen_vae_name() -> str | None:
    names = _get_filename_list("vae")
    matched = _match_by_patterns(names, _QWEN_VAE_PATTERNS)
    if matched is not None:
        return matched
    for name in names:
        lowered = name.lower()
        if "qwen" in lowered and "vae" in lowered and "layered" not in lowered:
            return name
    return None
