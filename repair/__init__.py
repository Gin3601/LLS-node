"""
node.repair
===========
功能域：图像修复（Repair）节点
"""
from .repair_finish import (
    NODE_CLASS_MAPPINGS as FINISH_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as FINISH_NODE_DISPLAY_NAME_MAPPINGS,
)
from .native_inpaint import (
    NODE_CLASS_MAPPINGS as NATIVE_INPAINT_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as NATIVE_INPAINT_NODE_DISPLAY_NAME_MAPPINGS,
)
from .repair_prepare import (
    NODE_CLASS_MAPPINGS as PREPARE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as PREPARE_NODE_DISPLAY_NAME_MAPPINGS,
)


def _merge_registry_maps(*maps):
    merged = {}
    for mapping in maps:
        overlap = set(merged).intersection(mapping)
        if overlap:
            duplicate_keys = ", ".join(sorted(overlap))
            raise RuntimeError(f"[LLS] Duplicate repair node registration keys: {duplicate_keys}")
        merged.update(mapping)
    return merged


NODE_CLASS_MAPPINGS = _merge_registry_maps(
    NATIVE_INPAINT_NODE_CLASS_MAPPINGS,
    PREPARE_NODE_CLASS_MAPPINGS,
    FINISH_NODE_CLASS_MAPPINGS,
)

NODE_DISPLAY_NAME_MAPPINGS = _merge_registry_maps(
    NATIVE_INPAINT_NODE_DISPLAY_NAME_MAPPINGS,
    PREPARE_NODE_DISPLAY_NAME_MAPPINGS,
    FINISH_NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
