from __future__ import annotations

import random
from dataclasses import dataclass, replace


MODEL_FAMILY_CHOICES = ("SD1.5", "SDXL", "FLUX")
TASK_MODE_CHOICES = ("txt2img",)
MAX_SEED = 0xFFFFFFFFFFFFFFFF


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, round(value / multiple) * multiple)


@dataclass(frozen=True, slots=True)
class LLSUniversalGenerationRequest:
    """统一图像生成请求。"""

    model_family: str
    task_mode: str
    model_name: str
    positive_prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    cfg: float
    seed: int
    sampler_name: str
    scheduler: str
    denoise: float

    def validate(self) -> "LLSUniversalGenerationRequest":
        if self.model_family not in MODEL_FAMILY_CHOICES:
            raise ValueError(
                f"Unsupported model_family '{self.model_family}'. "
                f"Expected one of {MODEL_FAMILY_CHOICES}."
            )
        if self.task_mode not in TASK_MODE_CHOICES:
            raise ValueError(
                f"Unsupported task_mode '{self.task_mode}'. "
                f"Only {TASK_MODE_CHOICES} are supported in this node."
            )
        if not self.model_name:
            raise ValueError("model_name must not be empty.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive integers.")
        if self.steps < 1:
            raise ValueError("steps must be at least 1.")
        if not (0.0 <= self.denoise <= 1.0):
            raise ValueError("denoise must be within [0.0, 1.0].")
        if self.seed < -1 or self.seed > MAX_SEED:
            raise ValueError(f"seed must be in [-1, {MAX_SEED}].")
        return self

    def resolve_seed(self) -> int:
        if self.seed == -1:
            return random.randint(0, MAX_SEED)
        return int(self.seed)

    def align_to(self, multiple: int) -> tuple["LLSUniversalGenerationRequest", bool]:
        aligned_width = _round_to_multiple(int(self.width), multiple)
        aligned_height = _round_to_multiple(int(self.height), multiple)
        if aligned_width == self.width and aligned_height == self.height:
            return self, False
        return replace(self, width=aligned_width, height=aligned_height), True
