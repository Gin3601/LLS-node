from .pro_edit_bridge import (
    NODE_CLASS_MAPPINGS as BRIDGE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as BRIDGE_NODE_DISPLAY_NAME_MAPPINGS,
)
from .pro_edit_finish import (
    NODE_CLASS_MAPPINGS as FINISH_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as FINISH_NODE_DISPLAY_NAME_MAPPINGS,
)
from .pro_edit_prepare import (
    NODE_CLASS_MAPPINGS as PREPARE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as PREPARE_NODE_DISPLAY_NAME_MAPPINGS,
)


def _merge_registry_maps(*maps):
    merged = {}
    for mapping in maps:
        overlap = set(merged).intersection(mapping)
        if overlap:
            duplicate_keys = ", ".join(sorted(overlap))
            raise RuntimeError(f"[LLS] Duplicate pro_edit node registration keys: {duplicate_keys}")
        merged.update(mapping)
    return merged


NODE_CLASS_MAPPINGS = _merge_registry_maps(
    PREPARE_NODE_CLASS_MAPPINGS,
    BRIDGE_NODE_CLASS_MAPPINGS,
    FINISH_NODE_CLASS_MAPPINGS,
)

NODE_DISPLAY_NAME_MAPPINGS = _merge_registry_maps(
    PREPARE_NODE_DISPLAY_NAME_MAPPINGS,
    BRIDGE_NODE_DISPLAY_NAME_MAPPINGS,
    FINISH_NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
