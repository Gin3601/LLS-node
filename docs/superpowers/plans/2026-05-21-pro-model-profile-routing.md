# LLS Pro Model Profile Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the current Pro image edit chain so loader, backend routing, and sampler routing all use a single explicit model-profile contract while keeping the existing Pro node names and workflow wiring stable.

**Architecture:** Add a new internal `model_profiles/` package that resolves a normalized `ModelProfile` from loader inputs, overrides, and legacy tags. Update `utils/model_info.py`, `model_loader/nodes.py`, `pro_edit/backends/registry.py`, `pro_edit/pro_edit_prepare.py`, and `pro_edit/pro_edit_bridge.py` so `backend_type` and `sampler_strategy` come from that profile instead of repeated filename heuristics.

**Tech Stack:** Python, unittest, existing ComfyUI node classes, JSON-ish metadata parsing, fake model/image/mask helpers, `python3 -m compileall`

---

## File Structure

- Create: `model_profiles/__init__.py`
  - Export the profile resolution helpers.
- Create: `model_profiles/base.py`
  - Define the normalized `ModelProfile` record and helper conversions.
- Create: `model_profiles/rules.py`
  - Define built-in profile rules for `sd15_base`, `sdxl_base`, `sdxl_inpaint`, `sdxl_edit`, `flux_base`, `flux_edit`.
- Create: `model_profiles/registry.py`
  - Resolve explicit overrides, legacy tags, and fallback rule matches into a final profile.
- Create: `tests/test_model_profiles.py`
  - Verify profile resolution from family, name patterns, overrides, and legacy tags.
- Create: `tests/test_model_profile_loader.py`
  - Verify loader writes `_lls_profile_*` metadata and preserves compatibility tags.
- Create: `tests/test_pro_edit_profile_prepare.py`
  - Verify Pro prepare fails cleanly for `backend_type = none` and writes profile metadata into `edit_info`.
- Modify: `tests/test_pro_edit_helpers.py`
  - Extend `FakeModel` to carry explicit profile fields.
- Modify: `tests/test_pro_edit_capabilities.py`
  - Add profile-aware expectations to existing model-info compatibility tests.
- Modify: `tests/test_pro_edit_registry.py`
  - Route by `backend_type` and reject unsupported base profiles.
- Modify: `tests/test_pro_edit_prepare_sdxl.py`
  - Assert profile metadata appears in `edit_info`.
- Modify: `tests/test_pro_edit_prepare_flux.py`
  - Assert profile metadata appears in `edit_info`.
- Modify: `tests/test_pro_edit_bridge.py`
  - Assert `sample_info` uses `profile_id` and `sampler_strategy`.
- Modify: `tests/test_pro_edit_docs.py`
  - Assert README documents the profile-driven routing model.
- Modify: `utils/model_info.py`
  - Wrap legacy helpers around the new profile registry.
- Modify: `model_loader/nodes.py`
  - Resolve and tag complete profiles during checkpoint loading.
- Modify: `pro_edit/backends/base.py`
  - Carry profile-aware routing records and backend validation helpers.
- Modify: `pro_edit/backends/registry.py`
  - Route by `backend_type` using resolved profiles.
- Modify: `pro_edit/backends/sdxl.py`
  - Validate and report `sdxl_native`.
- Modify: `pro_edit/backends/flux.py`
  - Validate and report `flux_edit`.
- Modify: `pro_edit/pro_edit_prepare.py`
  - Resolve the effective profile and reject unsupported base profiles.
- Modify: `pro_edit/pro_edit_bridge.py`
  - Resolve the effective profile and dispatch by `sampler_strategy`.
- Modify: `pro_edit/pro_edit_finish.py`
  - Preserve `profile_id`, `backend_type`, and `sampler_strategy` from `edit_info`.
- Modify: `pro_edit/pro_edit_utils.py`
  - Normalize and serialize the new profile fields inside `edit_info`.
- Modify: `README.md`
  - Document the profile-driven routing model and override strategy.

### Task 1: Add The Model Profile Core

**Files:**
- Create: `model_profiles/__init__.py`
- Create: `model_profiles/base.py`
- Create: `model_profiles/rules.py`
- Create: `model_profiles/registry.py`
- Create: `tests/test_model_profiles.py`
- Modify: `tests/test_pro_edit_helpers.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_profiles.py
import unittest

try:
    from .test_pro_edit_helpers import FakeModel, import_plugin_submodule, load_plugin_package
except ImportError:
    from test_pro_edit_helpers import FakeModel, import_plugin_submodule, load_plugin_package


class TestModelProfiles(unittest.TestCase):
    def setUp(self):
        self.plugin = load_plugin_package()
        self.registry = import_plugin_submodule(self.plugin, "model_profiles.registry")

    def test_resolves_sdxl_base_profile_from_family(self):
        profile = self.registry.resolve_model_profile(
            model=None,
            model_info={"checkpoint_name": "plain-sdxl-base.safetensors", "family": "SDXL"},
        )

        self.assertEqual(profile["profile_id"], "sdxl_base")
        self.assertEqual(profile["backend_type"], "none")
        self.assertEqual(profile["sampler_strategy"], "standard_k")

    def test_resolves_flux_kontext_profile_from_name(self):
        profile = self.registry.resolve_model_profile(
            model=None,
            model_info={"checkpoint_name": "demo-flux-kontext-dev.safetensors", "family": "FLUX_DEV"},
        )

        self.assertEqual(profile["profile_id"], "flux_edit")
        self.assertEqual(profile["role"], "edit")
        self.assertEqual(profile["backend_type"], "flux_edit")
        self.assertEqual(profile["sampler_strategy"], "flux_guided")

    def test_model_info_override_has_highest_priority(self):
        profile = self.registry.resolve_model_profile(
            model=None,
            model_info={
                "checkpoint_name": "plain-sdxl-base.safetensors",
                "family": "SDXL",
                "profile_id": "sdxl_edit",
                "role": "edit",
                "backend_type": "sdxl_native",
                "sampler_strategy": "standard_k",
                "supports_inpaint_native": True,
                "supports_image_edit_native": True,
                "preferred_edit_backend": "sdxl",
            },
        )

        self.assertEqual(profile["profile_id"], "sdxl_edit")
        self.assertEqual(profile["backend_type"], "sdxl_native")
        self.assertTrue(profile["supports_image_edit_native"])

    def test_legacy_tags_upgrade_to_matching_profile(self):
        model = FakeModel(
            family="FLUX_DEV",
            model_role="edit",
            supports_inpaint_native=False,
            supports_image_edit_native=True,
            preferred_edit_backend="flux",
            profile_id="",
            backend_type="",
            sampler_strategy="",
            loader_strategy="",
            model_name="legacy-flux-edit.safetensors",
        )

        profile = self.registry.resolve_model_profile(model=model, model_info=None)

        self.assertEqual(profile["profile_id"], "flux_edit")
        self.assertEqual(profile["backend_type"], "flux_edit")
        self.assertEqual(profile["sampler_strategy"], "flux_guided")


if __name__ == "__main__":
    unittest.main()
```

```python
# tests/test_pro_edit_helpers.py
class FakeModel:
    def __init__(
        self,
        family="SDXL",
        model_role="base",
        supports_inpaint_native=False,
        supports_image_edit_native=False,
        preferred_edit_backend=None,
        model_name="demo-model.safetensors",
        profile_id="",
        backend_type="",
        sampler_strategy="",
        loader_strategy="",
    ):
        self._lls_family = family
        self._lls_model_role = model_role
        self._lls_supports_inpaint_native = supports_inpaint_native
        self._lls_supports_image_edit_native = supports_image_edit_native
        self._lls_preferred_edit_backend = preferred_edit_backend
        self._lls_model_name = model_name
        self._lls_profile_id = profile_id
        self._lls_backend_type = backend_type
        self._lls_sampler_strategy = sampler_strategy
        self._lls_loader_strategy = loader_strategy
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_model_profiles.py' -v`

Expected: FAIL because `model_profiles` does not exist yet and `FakeModel` does not carry profile fields.

- [ ] **Step 3: Implement the model profile core**

```python
# model_profiles/base.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    family: str
    role: str
    backend_type: str
    sampler_strategy: str
    loader_strategy: str
    supports_inpaint_native: bool
    supports_image_edit_native: bool
    preferred_edit_backend: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_profile_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile_id": str(value.get("profile_id") or ""),
        "family": str(value.get("family") or value.get("model_family") or "SD1.5"),
        "role": str(value.get("role") or value.get("model_role") or "base"),
        "backend_type": str(value.get("backend_type") or "none"),
        "sampler_strategy": str(value.get("sampler_strategy") or "standard_k"),
        "loader_strategy": str(value.get("loader_strategy") or ""),
        "supports_inpaint_native": bool(value.get("supports_inpaint_native", False)),
        "supports_image_edit_native": bool(value.get("supports_image_edit_native", False)),
        "preferred_edit_backend": value.get("preferred_edit_backend"),
    }


def build_model_profile(**kwargs: Any) -> ModelProfile:
    normalized = normalize_profile_dict(kwargs)
    return ModelProfile(**normalized)
```

```python
# model_profiles/rules.py
from __future__ import annotations

from ..utils.model_info import canonicalize_family, infer_family_from_name, infer_model_role_from_name


_ROLE_TO_BACKEND = {
    ("SDXL", "inpaint"): ("sdxl_inpaint", "sdxl_native", "standard_k", "sdxl_checkpoint", True, False, "sdxl"),
    ("SDXL", "edit"): ("sdxl_edit", "sdxl_native", "standard_k", "sdxl_checkpoint", True, True, "sdxl"),
    ("SDXL", "fill"): ("sdxl_edit", "sdxl_native", "standard_k", "sdxl_checkpoint", True, True, "sdxl"),
    ("FLUX_DEV", "edit"): ("flux_edit", "flux_edit", "flux_guided", "flux_split_or_bundle", False, True, "flux"),
    ("FLUX_DEV", "fill"): ("flux_edit", "flux_edit", "flux_guided", "flux_split_or_bundle", False, True, "flux"),
    ("FLUX_DEV", "inpaint"): ("flux_edit", "flux_edit", "flux_guided", "flux_split_or_bundle", True, True, "flux"),
    ("FLUX_SCHNELL", "edit"): ("flux_edit", "flux_edit", "flux_guided", "flux_split_or_bundle", False, True, "flux"),
    ("FLUX_SCHNELL", "fill"): ("flux_edit", "flux_edit", "flux_guided", "flux_split_or_bundle", False, True, "flux"),
    ("FLUX_SCHNELL", "inpaint"): ("flux_edit", "flux_edit", "flux_guided", "flux_split_or_bundle", True, True, "flux"),
}


def _base_profile_for_family(family: str) -> dict[str, object]:
    if family == "SDXL":
        return {
            "profile_id": "sdxl_base",
            "family": family,
            "role": "base",
            "backend_type": "none",
            "sampler_strategy": "standard_k",
            "loader_strategy": "sdxl_checkpoint",
            "supports_inpaint_native": False,
            "supports_image_edit_native": False,
            "preferred_edit_backend": None,
        }
    if family in {"FLUX_DEV", "FLUX_SCHNELL"}:
        return {
            "profile_id": "flux_base",
            "family": family,
            "role": "base",
            "backend_type": "none",
            "sampler_strategy": "flux_guided",
            "loader_strategy": "flux_split_or_bundle",
            "supports_inpaint_native": False,
            "supports_image_edit_native": False,
            "preferred_edit_backend": None,
        }
    return {
        "profile_id": "sd15_base",
        "family": "SD1.5",
        "role": "base",
        "backend_type": "none",
        "sampler_strategy": "standard_k",
        "loader_strategy": "sd15_checkpoint",
        "supports_inpaint_native": False,
        "supports_image_edit_native": False,
        "preferred_edit_backend": None,
    }


def match_builtin_profile(model_name: str | None, family: str | None, role: str | None = None) -> dict[str, object]:
    resolved_family = canonicalize_family(family or infer_family_from_name(model_name, "SD1.5"))
    resolved_role = str(role or infer_model_role_from_name(model_name, resolved_family))
    key = (resolved_family, resolved_role)
    matched = _ROLE_TO_BACKEND.get(key)
    if matched is None:
        return _base_profile_for_family(resolved_family)
    profile_id, backend_type, sampler_strategy, loader_strategy, supports_inpaint_native, supports_image_edit_native, preferred_backend = matched
    return {
        "profile_id": profile_id,
        "family": resolved_family,
        "role": resolved_role,
        "backend_type": backend_type,
        "sampler_strategy": sampler_strategy,
        "loader_strategy": loader_strategy,
        "supports_inpaint_native": supports_inpaint_native,
        "supports_image_edit_native": supports_image_edit_native,
        "preferred_edit_backend": preferred_backend,
    }
```

```python
# model_profiles/registry.py
from __future__ import annotations

from .base import build_model_profile
from .rules import match_builtin_profile
from ..utils.model_info import get_lls_attr, parse_jsonish_info


def _extract_profile_overrides(model=None, model_info=None, extra_info=None) -> dict[str, object]:
    raw = parse_jsonish_info(model_info)
    extra = parse_jsonish_info(extra_info)

    def _pick(*values):
        for value in values:
            if value not in (None, ""):
                return value
        return None

    return {
        "profile_id": _pick(extra.get("profile_id"), raw.get("profile_id"), get_lls_attr(model, "profile_id", None)),
        "family": _pick(extra.get("family"), extra.get("model_family"), raw.get("family"), raw.get("model_family"), get_lls_attr(model, "family", None)),
        "role": _pick(extra.get("role"), extra.get("model_role"), raw.get("role"), raw.get("model_role"), get_lls_attr(model, "model_role", None)),
        "backend_type": _pick(extra.get("backend_type"), raw.get("backend_type"), get_lls_attr(model, "backend_type", None)),
        "sampler_strategy": _pick(extra.get("sampler_strategy"), raw.get("sampler_strategy"), get_lls_attr(model, "sampler_strategy", None)),
        "loader_strategy": _pick(extra.get("loader_strategy"), raw.get("loader_strategy"), get_lls_attr(model, "loader_strategy", None)),
        "supports_inpaint_native": (
            extra["supports_inpaint_native"] if "supports_inpaint_native" in extra
            else raw["supports_inpaint_native"] if "supports_inpaint_native" in raw
            else get_lls_attr(model, "supports_inpaint_native", None)
        ),
        "supports_image_edit_native": (
            extra["supports_image_edit_native"] if "supports_image_edit_native" in extra
            else raw["supports_image_edit_native"] if "supports_image_edit_native" in raw
            else get_lls_attr(model, "supports_image_edit_native", None)
        ),
        "preferred_edit_backend": _pick(extra.get("preferred_edit_backend"), raw.get("preferred_edit_backend"), get_lls_attr(model, "preferred_edit_backend", None)),
        "model_name": _pick(extra.get("checkpoint_name"), extra.get("model_name"), raw.get("checkpoint_name"), raw.get("model_name"), get_lls_attr(model, "model_name", None)),
    }


def resolve_model_profile(model=None, model_info=None, extra_info=None, checkpoint_name: str | None = None, family: str | None = None) -> dict[str, object]:
    overrides = _extract_profile_overrides(model=model, model_info=model_info, extra_info=extra_info)
    model_name = str(checkpoint_name or overrides.get("model_name") or "")
    base = match_builtin_profile(model_name, family or overrides.get("family"), overrides.get("role"))
    merged = dict(base)
    for key in (
        "profile_id",
        "family",
        "role",
        "backend_type",
        "sampler_strategy",
        "loader_strategy",
        "preferred_edit_backend",
    ):
        if overrides.get(key):
            merged[key] = overrides[key]
    for key in ("supports_inpaint_native", "supports_image_edit_native"):
        if overrides.get(key) is not None:
            merged[key] = bool(overrides[key])
    return build_model_profile(**merged).to_dict()
```

```python
# model_profiles/__init__.py
from .base import ModelProfile, build_model_profile, normalize_profile_dict
from .registry import resolve_model_profile

__all__ = [
    "ModelProfile",
    "build_model_profile",
    "normalize_profile_dict",
    "resolve_model_profile",
]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_model_profiles.py' -v`

Expected: PASS with four tests.

- [ ] **Step 5: Commit**

```bash
git add model_profiles tests/test_model_profiles.py tests/test_pro_edit_helpers.py
git commit -m "feat: add model profile core"
```

### Task 2: Make `utils/model_info.py` Profile-Aware Without Breaking Compatibility

**Files:**
- Modify: `utils/model_info.py`
- Modify: `tests/test_pro_edit_capabilities.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pro_edit_capabilities.py
    def test_parse_model_info_exposes_profile_fields(self):
        plugin = load_plugin_package()
        model_info = import_plugin_submodule(plugin, "utils.model_info")

        info = model_info.parse_model_info(
            {
                "checkpoint_name": "demo-flux-kontext-dev.safetensors",
                "family": "FLUX_DEV",
            }
        )

        self.assertEqual(info["profile_id"], "flux_edit")
        self.assertEqual(info["backend_type"], "flux_edit")
        self.assertEqual(info["sampler_strategy"], "flux_guided")

    def test_resolve_edit_capabilities_reads_profile_tags(self):
        plugin = load_plugin_package()
        model_info = import_plugin_submodule(plugin, "utils.model_info")
        model = FakeModel(
            family="SDXL",
            model_role="edit",
            supports_inpaint_native=True,
            supports_image_edit_native=True,
            preferred_edit_backend="sdxl",
            profile_id="sdxl_edit",
            backend_type="sdxl_native",
            sampler_strategy="standard_k",
            loader_strategy="sdxl_checkpoint",
            model_name="demo-sdxl-edit.safetensors",
        )

        capabilities = model_info.resolve_edit_capabilities(model=model, model_info=None)

        self.assertEqual(capabilities["profile_id"], "sdxl_edit")
        self.assertEqual(capabilities["backend_type"], "sdxl_native")
        self.assertEqual(capabilities["sampler_strategy"], "standard_k")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_pro_edit_capabilities.py' -v`

Expected: FAIL because `parse_model_info()` and `resolve_edit_capabilities()` do not yet expose profile fields.

- [ ] **Step 3: Implement the profile-aware compatibility layer**

```python
# utils/model_info.py
def parse_model_info(model_info: dict[str, Any] | str | None) -> dict[str, Any]:
    from ..model_profiles.registry import resolve_model_profile

    raw = parse_jsonish_info(model_info)
    raw_model_name = _get_model_name_alias_value(raw)
    family = canonicalize_family(
        raw.get("family")
        or raw.get("model_family")
        or infer_family_from_name(raw_model_name, "SD1.5")
    )
    defaults = get_family_defaults(family)
    profile = resolve_model_profile(model=None, model_info=raw, checkpoint_name=raw_model_name, family=family)

    info: dict[str, Any] = {**defaults}
    info.update(raw)
    info["family"] = profile["family"]
    info["model_family"] = profile["family"]
    info["model_role"] = profile["role"]
    info["profile_id"] = profile["profile_id"]
    info["backend_type"] = profile["backend_type"]
    info["sampler_strategy"] = profile["sampler_strategy"]
    info["loader_strategy"] = profile["loader_strategy"]
    info["supports_inpaint_native"] = bool(profile["supports_inpaint_native"])
    info["supports_image_edit_native"] = bool(profile["supports_image_edit_native"])
    info["preferred_edit_backend"] = profile["preferred_edit_backend"]
    info.setdefault("checkpoint_name", raw_model_name or "")
    info.setdefault("model_name", info["checkpoint_name"])
    return info


def resolve_edit_capabilities(model=None, model_info: dict[str, Any] | str | None = None) -> dict[str, Any]:
    from ..model_profiles.registry import resolve_model_profile

    info = parse_model_info(model_info)
    profile = resolve_model_profile(
        model=model,
        model_info=model_info,
        checkpoint_name=str(
            info.get("checkpoint_name")
            or info.get("model_name")
            or get_lls_attr(model, "model_name", "")
            or ""
        ),
        family=info.get("family") or info.get("model_family"),
    )
    return {
        "model_family": profile["family"],
        "model_name": str(
            get_lls_attr(model, "model_name", None)
            or info.get("checkpoint_name")
            or info.get("model_name")
            or ""
        ),
        "model_role": profile["role"],
        "profile_id": profile["profile_id"],
        "backend_type": profile["backend_type"],
        "sampler_strategy": profile["sampler_strategy"],
        "loader_strategy": profile["loader_strategy"],
        "supports_inpaint_native": bool(profile["supports_inpaint_native"]),
        "supports_image_edit_native": bool(profile["supports_image_edit_native"]),
        "preferred_edit_backend": profile["preferred_edit_backend"],
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_pro_edit_capabilities.py' -v`

Expected: PASS with the new profile-aware assertions and the existing capability regression coverage.

- [ ] **Step 5: Commit**

```bash
git add utils/model_info.py tests/test_pro_edit_capabilities.py
git commit -m "feat: expose model profiles through model info helpers"
```

### Task 3: Stamp Resolved Profiles In The Loader

**Files:**
- Create: `tests/test_model_profile_loader.py`
- Modify: `model_loader/nodes.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_profile_loader.py
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    from .test_pro_edit_helpers import import_plugin_submodule, load_plugin_package
except ImportError:
    from test_pro_edit_helpers import import_plugin_submodule, load_plugin_package


class TestModelProfileLoader(unittest.TestCase):
    def test_loader_writes_profile_fields_to_outputs(self):
        plugin = load_plugin_package()
        loader_module = import_plugin_submodule(plugin, "model_loader.nodes")
        loader = loader_module.LLSSimpleCheckpointLoader()
        model = SimpleNamespace()
        clip = SimpleNamespace()
        vae = SimpleNamespace()

        with mock.patch.object(loader_module, "folder_paths", object()), \
             mock.patch.object(loader_module, "comfy_sd", object()), \
             mock.patch.object(loader_module, "_resolve_model_path", return_value=("diffusion_models", "/fake/demo-flux-kontext-dev.safetensors")), \
             mock.patch.object(loader_module, "_load_model", return_value=(model, clip, vae)), \
             mock.patch.object(loader_module, "_resolve_text_encoder", return_value=(clip, "embedded", None, None)), \
             mock.patch.object(loader_module, "_resolve_vae", return_value=(vae, "embedded", None)):
            loaded_model, loaded_clip, loaded_vae, text_encoder = loader.load_checkpoint(
                ckpt_name="demo-flux-kontext-dev.safetensors",
                model_family="FLUX_DEV",
                load_mode="simple",
                vae_source="auto",
                text_encoder_source="auto",
                external_vae_name=loader_module.AUTO_PLACEHOLDER,
                external_text_encoder_1=loader_module.AUTO_PLACEHOLDER,
                external_text_encoder_2=loader_module.AUTO_PLACEHOLDER,
            )

        for obj in (loaded_model, loaded_clip, loaded_vae, text_encoder):
            self.assertEqual(obj._lls_profile_id, "flux_edit")
            self.assertEqual(obj._lls_backend_type, "flux_edit")
            self.assertEqual(obj._lls_sampler_strategy, "flux_guided")
            self.assertEqual(obj._lls_loader_strategy, "flux_split_or_bundle")
            self.assertEqual(obj._lls_model_role, "edit")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_model_profile_loader.py' -v`

Expected: FAIL because loader output tags do not yet include `_lls_profile_*` fields.

- [ ] **Step 3: Update the loader to stamp full profiles**

```python
# model_loader/nodes.py
from ..model_profiles.registry import resolve_model_profile


def _build_capability_tags(model_name: str, family: str) -> dict[str, object]:
    profile = resolve_model_profile(
        model=None,
        model_info={"checkpoint_name": model_name, "family": family},
        checkpoint_name=model_name,
        family=family,
    )
    return {
        "profile_id": profile["profile_id"],
        "backend_type": profile["backend_type"],
        "sampler_strategy": profile["sampler_strategy"],
        "loader_strategy": profile["loader_strategy"],
        "model_role": profile["role"],
        "supports_inpaint_native": profile["supports_inpaint_native"],
        "supports_image_edit_native": profile["supports_image_edit_native"],
        "preferred_edit_backend": profile["preferred_edit_backend"],
    }
```

```python
# model_loader/nodes.py
        profile_tags = _build_capability_tags(ckpt_name, family)

        tag_lls_object(
            model,
            family=family,
            model_name=ckpt_name,
            checkpoint_name=ckpt_name,
            load_mode=load_mode,
            **profile_tags,
        )
        tag_lls_object(
            text_encoder,
            family=family,
            model_name=ckpt_name,
            checkpoint_name=ckpt_name,
            text_encoder_type=defaults["text_encoder_type"],
            text_encoder_source=resolved_text_encoder_source,
            text_encoder_name=resolved_text_encoder_name,
            text_encoder_name_1=resolved_text_encoder_name_1,
            text_encoder_name_2=resolved_text_encoder_name_2,
            **profile_tags,
        )
        tag_lls_object(
            vae,
            family=family,
            model_name=ckpt_name,
            checkpoint_name=ckpt_name,
            vae_name=resolved_vae_label,
            vae_source=resolved_vae_source,
            **profile_tags,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_model_profile_loader.py' -v`

Expected: PASS with one loader-tagging test.

- [ ] **Step 5: Commit**

```bash
git add model_loader/nodes.py tests/test_model_profile_loader.py
git commit -m "feat: stamp resolved model profiles in loader"
```

### Task 4: Refactor Pro Backend Routing To Use `backend_type`

**Files:**
- Modify: `pro_edit/backends/base.py`
- Modify: `pro_edit/backends/registry.py`
- Modify: `pro_edit/backends/sdxl.py`
- Modify: `pro_edit/backends/flux.py`
- Modify: `tests/test_pro_edit_registry.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pro_edit_registry.py
    def test_auto_routes_sdxl_profile_by_backend_type(self):
        model = FakeModel(
            family="SDXL",
            model_role="edit",
            supports_inpaint_native=True,
            supports_image_edit_native=True,
            preferred_edit_backend="sdxl",
            profile_id="sdxl_edit",
            backend_type="sdxl_native",
            sampler_strategy="standard_k",
            loader_strategy="sdxl_checkpoint",
        )

        backend, routing = self.registry.resolve_backend("auto", model=model)

        self.assertEqual(backend.backend_name, "sdxl")
        self.assertEqual(routing.profile["profile_id"], "sdxl_edit")
        self.assertEqual(routing.profile["backend_type"], "sdxl_native")

    def test_auto_rejects_base_profile_for_pro_chain(self):
        model = FakeModel(
            family="SDXL",
            model_role="base",
            supports_inpaint_native=False,
            supports_image_edit_native=False,
            preferred_edit_backend=None,
            profile_id="sdxl_base",
            backend_type="none",
            sampler_strategy="standard_k",
            loader_strategy="sdxl_checkpoint",
        )

        with self.assertRaisesRegex(RuntimeError, "Pro image edit is not available for profile 'sdxl_base'"):
            self.registry.resolve_backend("auto", model=model)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_pro_edit_registry.py' -v`

Expected: FAIL because routing records do not yet carry `profile`, and `backend_type = none` is not treated as a first-class unsupported profile state.

- [ ] **Step 3: Route by `backend_type`**

```python
# pro_edit/backends/base.py
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RoutingResult:
    backend_name: str
    routing_reason: str
    capabilities: dict[str, Any]
    profile: dict[str, Any]


def validate_manual_backend(backend_name: str, backend, profile: dict[str, Any]) -> None:
    if backend.supports(profile):
        return
    raise RuntimeError(
        f"[LLS] Pro edit backend '{backend_name}' is incompatible with profile "
        f"'{profile.get('profile_id')}' (family '{profile.get('family')}', role '{profile.get('role')}')."
    )
```

```python
# pro_edit/backends/registry.py
from ...model_profiles.registry import resolve_model_profile
from ...utils.model_info import parse_jsonish_info


def resolve_backend(backend_mode: str, *, model=None, model_info=None, edit_info=None):
    _ensure_builtin_backends()
    mode = str(backend_mode or "auto").strip().lower() or "auto"
    merged_extra_info = dict(parse_jsonish_info(model_info))
    merged_extra_info.update(parse_jsonish_info(edit_info))
    profile = resolve_model_profile(model=model, model_info=model_info, extra_info=merged_extra_info)
    capabilities = {
        "model_family": profile["family"],
        "model_role": profile["role"],
        "profile_id": profile["profile_id"],
        "backend_type": profile["backend_type"],
        "sampler_strategy": profile["sampler_strategy"],
        "loader_strategy": profile["loader_strategy"],
        "supports_inpaint_native": profile["supports_inpaint_native"],
        "supports_image_edit_native": profile["supports_image_edit_native"],
        "preferred_edit_backend": profile["preferred_edit_backend"],
    }

    backend_type = str(profile.get("backend_type") or "none")
    if backend_type == "none":
        raise RuntimeError(
            f"[LLS] Pro image edit is not available for profile '{profile['profile_id']}' "
            f"(family '{profile['family']}', role '{profile['role']}'). "
            "Use a supported native edit/inpaint profile or provide an explicit override."
        )

    backend_name = {
        "sdxl_native": "sdxl",
        "flux_edit": "flux",
    }.get(backend_type)
    if backend_name is None:
        raise RuntimeError(f"[LLS] Unsupported pro edit backend_type '{backend_type}'.")
    if mode != "auto":
        backend = get_backend(mode)
        validate_manual_backend(mode, backend, profile)
        return backend, RoutingResult(backend.backend_name, "manual.override", capabilities, profile)

    backend = get_backend(backend_name)
    validate_manual_backend(backend_name, backend, profile)
    return backend, RoutingResult(backend.backend_name, "profile.backend_type", capabilities, profile)
```

```python
# pro_edit/backends/sdxl.py
    def supports(self, profile):
        return str(profile.get("backend_type") or "") == "sdxl_native"
```

```python
# pro_edit/backends/flux.py
    def supports(self, profile):
        return str(profile.get("backend_type") or "") == "flux_edit"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_pro_edit_registry.py' -v`

Expected: PASS with the updated profile-based routing assertions plus the legacy regression tests adjusted to the new `routing_reason`.

- [ ] **Step 5: Commit**

```bash
git add pro_edit/backends tests/test_pro_edit_registry.py
git commit -m "feat: route pro backends by model profile"
```

### Task 5: Make Pro Prepare Profile-Driven

**Files:**
- Create: `tests/test_pro_edit_profile_prepare.py`
- Modify: `pro_edit/pro_edit_prepare.py`
- Modify: `pro_edit/pro_edit_utils.py`
- Modify: `tests/test_pro_edit_prepare_sdxl.py`
- Modify: `tests/test_pro_edit_prepare_flux.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pro_edit_profile_prepare.py
import unittest

try:
    from .test_pro_edit_helpers import FakeMask, FakeModel, FakeTensor, FakeVAE, load_plugin_package, make_conditioning
except ImportError:
    from test_pro_edit_helpers import FakeMask, FakeModel, FakeTensor, FakeVAE, load_plugin_package, make_conditioning


class TestProEditProfilePrepare(unittest.TestCase):
    def test_prepare_rejects_base_profile(self):
        plugin = load_plugin_package()
        node = plugin.NODE_CLASS_MAPPINGS["LLSProImageEditPrepare"]()
        image = FakeTensor((1, 64, 64, 3), label="image")
        mask = FakeMask((1, 64, 64), mask_bbox=(8, 8, 32, 32), mask_area_ratio=0.2)
        vae = FakeVAE(latent_channels=4, downscale_ratio=8)
        model = FakeModel(
            family="SDXL",
            model_role="base",
            supports_inpaint_native=False,
            supports_image_edit_native=False,
            preferred_edit_backend=None,
            profile_id="sdxl_base",
            backend_type="none",
            sampler_strategy="standard_k",
            loader_strategy="sdxl_checkpoint",
        )

        with self.assertRaisesRegex(RuntimeError, "Pro image edit is not available for profile 'sdxl_base'"):
            node.prepare(
                image=image,
                mask=mask,
                vae=vae,
                positive=make_conditioning("positive"),
                negative=make_conditioning("negative"),
                backend_mode="auto",
                edit_scope="region",
                mask_grow=0,
                mask_blur=0.0,
                mask_threshold=0.5,
                invert_mask=False,
                crop_context=32,
                crop_context_factor=1.0,
                min_size=128,
                max_size=512,
                resize_mode="fit",
                expand_left=0,
                expand_right=0,
                expand_top=0,
                expand_bottom=0,
                canvas_fill="edge",
                auto_recommend="enabled",
                model=model,
                model_info=None,
            )
```

```python
# tests/test_pro_edit_prepare_sdxl.py
        self.assertEqual(edit_info["profile_id"], "sdxl_inpaint")
        self.assertEqual(edit_info["backend_type"], "sdxl_native")
        self.assertEqual(edit_info["sampler_strategy"], "standard_k")
```

```python
# tests/test_pro_edit_prepare_flux.py
        self.assertEqual(edit_info["profile_id"], "flux_edit")
        self.assertEqual(edit_info["backend_type"], "flux_edit")
        self.assertEqual(edit_info["sampler_strategy"], "flux_guided")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_pro_edit_profile_prepare.py' -v`

Expected: FAIL because prepare/edit_info do not yet preserve the resolved profile metadata.

- [ ] **Step 3: Resolve and serialize profiles in prepare**

```python
# pro_edit/pro_edit_utils.py
def build_edit_info(workspace: dict[str, Any], routing, *, backend_hints: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = routing.profile
    info = {
        "backend_name": routing.backend_name,
        "routing_reason": routing.routing_reason,
        "model_family": profile["family"],
        "model_role": profile["role"],
        "profile_id": profile["profile_id"],
        "backend_type": profile["backend_type"],
        "sampler_strategy": profile["sampler_strategy"],
        "loader_strategy": profile["loader_strategy"],
        "supports_inpaint_native": profile["supports_inpaint_native"],
        "supports_image_edit_native": profile["supports_image_edit_native"],
        "preferred_edit_backend": profile["preferred_edit_backend"],
        "edit_scope": workspace["edit_scope"],
        "original_size": list(workspace["original_size"]),
        "work_size": list(workspace["work_size"]),
        "crop_box": workspace["crop_box"],
        "crop_scale": workspace["crop_scale"],
        "canvas_expand": workspace["canvas_expand"],
        "original_box_in_canvas": workspace["original_box_in_canvas"],
        "recommended_denoise": float(workspace["recommended_denoise"]),
        "edit_payload_version": "1.0",
    }
    if backend_hints:
        info.update(backend_hints)
    return info
```

```python
# pro_edit/pro_edit_prepare.py
        backend, routing = resolve_backend(
            backend_mode,
            model=model,
            model_info=model_info,
        )
        prepared = backend.prepare(
            model=model,
            vae=vae,
            work_image=workspace["work_image"],
            work_mask=workspace["work_mask"],
            positive=positive,
            negative=negative,
            workspace=workspace,
            routing=routing,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_pro_edit_profile_prepare.py' -v`

Expected: PASS with the unsupported-base-profile failure.

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_pro_edit_prepare_sdxl.py' -v`

Expected: PASS with profile metadata assertions.

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_pro_edit_prepare_flux.py' -v`

Expected: PASS with profile metadata assertions.

- [ ] **Step 5: Commit**

```bash
git add pro_edit/pro_edit_prepare.py pro_edit/pro_edit_utils.py tests/test_pro_edit_profile_prepare.py tests/test_pro_edit_prepare_sdxl.py tests/test_pro_edit_prepare_flux.py
git commit -m "feat: make pro prepare profile driven"
```

### Task 6: Route The Bridge By `sampler_strategy`

**Files:**
- Modify: `pro_edit/pro_edit_bridge.py`
- Modify: `pro_edit/pro_edit_finish.py`
- Modify: `tests/test_pro_edit_bridge.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pro_edit_bridge.py
        self.assertEqual(sample_info["profile_id"], "sdxl_inpaint")
        self.assertEqual(sample_info["sampler_strategy"], "standard_k")
```

```python
# tests/test_pro_edit_bridge.py
        self.assertEqual(sample_info["profile_id"], "flux_edit")
        self.assertEqual(sample_info["sampler_strategy"], "flux_guided")
```

```python
# tests/test_pro_edit_bridge.py
    def test_bridge_rejects_unknown_sampler_strategy(self):
        model = FakeModel(
            family="FLUX_DEV",
            model_role="edit",
            supports_inpaint_native=False,
            supports_image_edit_native=True,
            preferred_edit_backend="flux",
            profile_id="flux_edit",
            backend_type="flux_edit",
            sampler_strategy="mystery_strategy",
            loader_strategy="flux_split_or_bundle",
        )
        latent = {"samples": FakeLatentTensor((1, 16, 64, 64)), "source": "pro_edit_prepare_region"}

        with self.assertRaisesRegex(RuntimeError, "Unsupported sampler_strategy 'mystery_strategy'"):
            self.node.sample(
                model=model,
                positive=make_conditioning("positive"),
                negative=make_conditioning("negative"),
                latent_image=latent,
                backend_mode="auto",
                quality_preset="Manual",
                seed=9,
                steps=12,
                cfg=1.0,
                sampler_name="euler",
                scheduler="simple",
                denoise=0.8,
                denoise_mode="manual",
                flux_guidance=4.2,
                model_family="Auto",
                edit_info=None,
                model_info=None,
            )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_pro_edit_bridge.py' -v`

Expected: FAIL because `sample_info` does not yet expose `profile_id` / `sampler_strategy`, and bridge logic still branches directly on backend name.

- [ ] **Step 3: Dispatch by `sampler_strategy`**

```python
# pro_edit/pro_edit_bridge.py
from ..model_profiles.registry import resolve_model_profile
from ..utils.model_info import parse_jsonish_info


def _apply_sampler_strategy(strategy: str, *, positive, negative, flux_guidance, defaults):
    if strategy == "standard_k":
        return positive, negative, None
    if strategy == "flux_guided":
        guidance_value = _normalize_flux_guidance(flux_guidance, defaults.get("default_guidance"))
        positive = set_conditioning_values(positive, {"guidance": guidance_value})
        negative = set_conditioning_values(negative, {"guidance": guidance_value})
        return positive, negative, guidance_value
    raise RuntimeError(f"[LLS] Unsupported sampler_strategy '{strategy}'.")
```

```python
# pro_edit/pro_edit_bridge.py
        merged_profile_info = dict(parse_jsonish_info(effective_model_info))
        merged_profile_info.update(parse_jsonish_info(edit_info))
        profile = resolve_model_profile(model=model, model_info=effective_model_info, extra_info=merged_profile_info)
        backend, routing = resolve_backend(
            backend_mode,
            model=model,
            model_info=effective_model_info,
            edit_info=edit_info,
        )
        defaults = get_family_defaults(profile["family"])
        positive, negative, guidance_value = _apply_sampler_strategy(
            profile["sampler_strategy"],
            positive=positive,
            negative=negative,
            flux_guidance=flux_guidance,
            defaults=defaults,
        )
```

```python
# pro_edit/pro_edit_bridge.py
        sample_info = info_to_json(
            {
                "backend_name": routing.backend_name,
                "routing_reason": routing.routing_reason,
                "family": profile["family"],
                "model_role": profile["role"],
                "profile_id": profile["profile_id"],
                "backend_type": profile["backend_type"],
                "sampler_strategy": profile["sampler_strategy"],
                "seed": actual_seed,
                "steps": int(steps),
                "cfg": float(cfg),
                "guidance": guidance_value,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": actual_denoise,
                "denoise_mode": denoise_mode,
                "quality_preset": quality_preset,
            }
        )
```

```python
# pro_edit/pro_edit_finish.py
        info = normalize_edit_info(edit_info)
        info.setdefault("profile_id", "")
        info.setdefault("backend_type", "none")
        info.setdefault("sampler_strategy", "standard_k")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_pro_edit_bridge.py' -v`

Expected: PASS with strategy-based routing and richer `sample_info`.

- [ ] **Step 5: Commit**

```bash
git add pro_edit/pro_edit_bridge.py pro_edit/pro_edit_finish.py tests/test_pro_edit_bridge.py
git commit -m "feat: dispatch pro bridge by sampler strategy"
```

### Task 7: Update Documentation And Run Full Verification

**Files:**
- Modify: `README.md`
- Modify: `tests/test_pro_edit_docs.py`

- [ ] **Step 1: Write the failing documentation tests**

```python
# tests/test_pro_edit_docs.py
        for needle in (
            "profile-driven routing",
            "profile_id",
            "backend_type",
            "sampler_strategy",
            "LLS Simple Checkpoint Loader writes the resolved model profile",
            "FLUX base models should remain on the Simple workflow unless they resolve to flux_edit",
        ):
            self.assertIn(needle, text)
```

- [ ] **Step 2: Run the documentation tests to verify they fail**

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_pro_edit_docs.py' -v`

Expected: FAIL because README does not yet describe the profile-driven routing model.

- [ ] **Step 3: Update the README**

~~~markdown
## Pro Image Edit / Inpaint

The Pro workflow now uses profile-driven routing.

- `LLS Simple Checkpoint Loader` writes the resolved model profile into runtime metadata.
- `LLS Pro Image Edit Prepare` routes by `backend_type`.
- `LLS Pro KSampler Bridge` routes by `sampler_strategy`.

Important profile fields:

- `profile_id`
- `backend_type`
- `sampler_strategy`

Examples:

- `sdxl_inpaint` -> `backend_type = sdxl_native`
- `flux_edit` -> `backend_type = flux_edit`

Base profiles do not silently use the Pro chain:

- `SDXL base` remains a base profile unless explicitly overridden
- `FLUX base models should remain on the Simple workflow unless they resolve to flux_edit`

Manual correction still happens through `model_info` overrides, for example:

~~~json
{"profile_id":"flux_edit","backend_type":"flux_edit","sampler_strategy":"flux_guided","role":"edit"}
~~~
~~~

- [ ] **Step 4: Run the focused docs test**

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_pro_edit_docs.py' -v`

Expected: PASS with one documentation test file.

- [ ] **Step 5: Run full verification**

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_model_profiles.py' -v`

Expected: PASS.

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_model_profile_loader.py' -v`

Expected: PASS.

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_pro_edit*.py' -v`

Expected: PASS, with finish compositing tests allowed to skip when torch is unavailable in the local verification interpreter.

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_model_info_inference.py' -v`

Expected: PASS.

Run: `env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_repair*.py' -v`

Expected: PASS and confirm the existing `LLSSimple*` repair chain remains unaffected.

Run: `python3 -m compileall __init__.py model_profiles pro_edit utils model_loader`

Expected: PASS with no syntax errors.

- [ ] **Step 6: Commit**

```bash
git add README.md tests/test_pro_edit_docs.py docs/superpowers/specs/2026-05-21-pro-model-profile-routing-design.md docs/superpowers/plans/2026-05-21-pro-model-profile-routing.md
git commit -m "docs: add profile driven pro routing design"
```

## Self-Review Checklist

- Spec coverage:
  - explicit `ModelProfile` contract is covered by Task 1
  - profile-aware `utils/model_info.py` compatibility is covered by Task 2
  - loader-side profile stamping is covered by Task 3
  - backend routing by `backend_type` is covered by Task 4
  - bridge routing by `sampler_strategy` is covered by Task 6
  - README and regression verification are covered by Task 7
- Placeholder scan:
  - no `TBD`, `TODO`, or “implement later” markers remain
  - every task has exact file paths, exact commands, and concrete code snippets
- Type consistency:
  - `profile_id`, `backend_type`, `sampler_strategy`, `loader_strategy` are used consistently across model profiles, loader tags, `edit_info`, and `sample_info`
  - `sdxl_native` and `flux_edit` remain the only built-in Pro backend types in this iteration
  - base profiles consistently use `backend_type = none`
