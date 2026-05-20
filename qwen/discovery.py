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
NO_LORA_PLACEHOLDER = "(no loras found)"
AUTO_TURBO_LORA_CHOICE = "(auto)"
NO_TEXT_TURBO_LORA_PLACEHOLDER = "(no qwen text turbo loras found)"
NO_EDIT_TURBO_LORA_PLACEHOLDER = "(no qwen edit turbo loras found)"
REFERENCE_LATENTS_METHOD_CHOICES = ["offset", "index", "uxo/uno", "index_timestep_zero"]

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


def _lower(value: str | None) -> str:
    return (value or "").lower()


def is_qwen_text_to_image_model(name: str | None) -> bool:
    lowered = _lower(name)
    return (
        "qwen" in lowered
        and "image" in lowered
        and "edit" not in lowered
        and "layered" not in lowered
    )


def is_qwen_image_edit_model(name: str | None) -> bool:
    lowered = _lower(name)
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


def get_qwen_lora_choices() -> list[str]:
    names = _sorted_unique(_get_filename_list("loras"))
    return names or [NO_LORA_PLACEHOLDER]


def validate_qwen_lora_name(lora_name: str) -> str:
    if not lora_name or lora_name == NO_LORA_PLACEHOLDER:
        raise RuntimeError("[LLS] Missing Qwen LoRA selection.")
    available = _get_filename_list("loras")
    if lora_name not in available:
        raise RuntimeError(
            f"[LLS] LoRA '{lora_name}' was not found in ComfyUI/models/loras/."
        )
    return lora_name


def is_qwen_text_turbo_lora(name: str | None) -> bool:
    lowered = _lower(name)
    return "qwen-image" in lowered and "lightning" in lowered and "edit" not in lowered


def is_qwen_edit_turbo_lora(name: str | None) -> bool:
    lowered = _lower(name)
    return "qwen-image-edit" in lowered and "lightning" in lowered


def get_qwen_text_turbo_lora_choices() -> list[str]:
    names = _sorted_unique([name for name in _get_filename_list("loras") if is_qwen_text_turbo_lora(name)])
    if not names:
        return [AUTO_TURBO_LORA_CHOICE, NO_TEXT_TURBO_LORA_PLACEHOLDER]
    return [AUTO_TURBO_LORA_CHOICE] + names


def get_qwen_edit_turbo_lora_choices() -> list[str]:
    names = _sorted_unique([name for name in _get_filename_list("loras") if is_qwen_edit_turbo_lora(name)])
    if not names:
        return [AUTO_TURBO_LORA_CHOICE, NO_EDIT_TURBO_LORA_PLACEHOLDER]
    return [AUTO_TURBO_LORA_CHOICE] + names


def _resolve_candidate_turbo_loras(model_name: str, available: list[str]) -> list[str]:
    lowered = _lower(model_name)
    if is_qwen_text_to_image_model(model_name):
        if "2512" in lowered:
            return _sorted_unique([name for name in available if "2512" in _lower(name)])
        return _sorted_unique(
            [
                name for name in available
                if "2512" not in _lower(name)
                and "2509" not in _lower(name)
                and "2511" not in _lower(name)
            ]
        )

    if is_qwen_image_edit_model(model_name):
        if "2511" in lowered:
            return _sorted_unique([name for name in available if "2511" in _lower(name)])
        if "2509" in lowered:
            return _sorted_unique([name for name in available if "2509" in _lower(name)])
    return []


def _resolve_turbo_lora(
    *,
    model_name: str,
    requested_name: str,
    all_loras: list[str],
    available: list[str],
    model_label: str,
) -> str:
    candidates = _resolve_candidate_turbo_loras(model_name, available)

    if requested_name == AUTO_TURBO_LORA_CHOICE:
        if candidates:
            return candidates[0]
        raise RuntimeError(f"[LLS] No compatible turbo/lightning LoRA exists for '{model_name}'.")

    if requested_name not in all_loras:
        raise RuntimeError(
            f"[LLS] Turbo LoRA '{requested_name}' was not found in ComfyUI/models/loras/."
        )

    if requested_name not in available or requested_name not in candidates:
        raise RuntimeError(
            f"[LLS] Turbo LoRA '{requested_name}' is not compatible with {model_label} model '{model_name}'."
        )

    return requested_name


def resolve_qwen_text_turbo_lora(model_name: str, requested_name: str) -> str:
    all_loras = _sorted_unique(_get_filename_list("loras"))
    available = [name for name in all_loras if is_qwen_text_turbo_lora(name)]
    return _resolve_turbo_lora(
        model_name=model_name,
        requested_name=requested_name,
        all_loras=all_loras,
        available=available,
        model_label="text-to-image",
    )


def resolve_qwen_edit_turbo_lora(model_name: str, requested_name: str) -> str:
    all_loras = _sorted_unique(_get_filename_list("loras"))
    available = [name for name in all_loras if is_qwen_edit_turbo_lora(name)]
    return _resolve_turbo_lora(
        model_name=model_name,
        requested_name=requested_name,
        all_loras=all_loras,
        available=available,
        model_label="image-edit",
    )


def get_qwen_turbo_profile(model_name: str) -> dict[str, float] | None:
    lowered = _lower(model_name)
    if is_qwen_image_edit_model(model_name):
        if "2509" in lowered or "2511" in lowered:
            return {"steps": 4, "cfg": 1.0}
        return None

    if is_qwen_text_to_image_model(model_name):
        if "2512" in lowered:
            return {"steps": 4, "cfg": 1.0}
        return {"steps": 8, "cfg": 1.0}

    return None


def resolve_qwen_text_encoder_name() -> str | None:
    names = _get_filename_list("text_encoders")
    matched = _match_by_patterns(names, _QWEN_TEXT_ENCODER_PATTERNS)
    if matched is not None:
        return matched
    for name in names:
        lowered = _lower(name)
        if "qwen" in lowered and ("vl" in lowered or "qwen25_7b" in lowered):
            return name
    return None


def resolve_qwen_vae_name() -> str | None:
    names = _get_filename_list("vae")
    matched = _match_by_patterns(names, _QWEN_VAE_PATTERNS)
    if matched is not None:
        return matched
    for name in names:
        lowered = _lower(name)
        if "qwen" in lowered and "vae" in lowered and "layered" not in lowered:
            return name
    return None
