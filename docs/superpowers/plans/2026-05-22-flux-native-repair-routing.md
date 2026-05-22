# FLUX Native Repair Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `LLSSimpleRepairPrepare` so FLUX local repair can follow the official Fill workflow shape while base models still fall back to generic repair behavior.

**Architecture:** Add a repair backend registry parallel to `pro_edit`, with a native FLUX backend that prepares `InpaintModelConditioning` payloads and marks the sampler to patch the model through `DifferentialDiffusion`. Keep the existing latent-mask / VAE-inpaint preparation as the generic fallback path for base or unsupported profiles.

**Tech Stack:** Python, unittest, existing ComfyUI node classes, model profile routing, fake tensor/mask helpers

---

### Task 1: Lock in the failing tests

**Files:**
- Modify: `tests/test_repair_registration.py`
- Modify: `tests/test_repair_utils.py`
- Modify: `tests/test_repair_prepare.py`
- Create: `tests/test_repair_registry.py`

- [ ] **Step 1: Add schema coverage for optional `model` input and backend metadata**
- [ ] **Step 2: Add routing tests showing FLUX fill/edit profiles choose the native backend and base profiles stay on fallback**
- [ ] **Step 3: Add prepare tests showing native FLUX returns `InpaintModelConditioning`-style latent/conditioning instead of the old fallback latent payload**
- [ ] **Step 4: Run focused repair tests and confirm the new expectations fail for the current implementation**

### Task 2: Add repair backend routing and runtime wrappers

**Files:**
- Create: `repair/backends/base.py`
- Create: `repair/backends/registry.py`
- Create: `repair/backends/generic.py`
- Create: `repair/backends/flux.py`
- Create: `repair/runtime.py`

- [ ] **Step 1: Mirror the `pro_edit` routing/result structure for repair**
- [ ] **Step 2: Wrap ComfyUI `InpaintModelConditioning` and `DifferentialDiffusion` with runtime-safe helpers**
- [ ] **Step 3: Implement the FLUX native prepare path and generic fallback path**
- [ ] **Step 4: Re-run focused routing tests until green**

### Task 3: Integrate repair prepare and sampler

**Files:**
- Modify: `repair/repair_prepare.py`
- Modify: `repair/repair_utils.py`
- Modify: `repair/__init__.py`
- Modify: `sampling/nodes.py`

- [ ] **Step 1: Add optional `model` input to `LLSSimpleRepairPrepare` and resolve the model profile from `model` + `model_info`**
- [ ] **Step 2: Move backend-specific latent/conditioning preparation behind the new registry**
- [ ] **Step 3: Extend `repair_info` with backend/routing/execution metadata**
- [ ] **Step 4: Patch FLUX native models with `DifferentialDiffusion` inside `LLSSimpleKSampler` when `repair_info` requests it**
- [ ] **Step 5: Re-run focused repair prepare + sampler tests**

### Task 4: Update the workflow JSON and verify regressions

**Files:**
- Modify: `测试工作流.json`
- Modify: other files from earlier tasks only if verification exposes gaps

- [ ] **Step 1: Rewire the FLUX repair example workflow to feed `model` into `LLSSimpleRepairPrepare` and keep the prompt text on the active path**
- [ ] **Step 2: Run `python3 -m unittest tests.test_repair_registration tests.test_repair_registry tests.test_repair_prepare tests.test_repair_sampler tests.test_repair_utils -v`**
- [ ] **Step 3: Run `python3 -m compileall __init__.py repair sampling model_profiles pro_edit qwen model_loader`**
- [ ] **Step 4: Inspect the final JSON diff to confirm the workflow matches the intended FLUX repair path**
