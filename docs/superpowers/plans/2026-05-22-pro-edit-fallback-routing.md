# Pro Edit Fallback Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep official model profiles intact while allowing base/generation models to enter the `LLS Pro *` edit flow through an automatic fallback repaint path.

**Architecture:** Preserve `model_profiles/` as the source of truth for official model capability. Change Pro routing so `edit_info` still means "edit task", but `sdxl_base`, `flux_base`, and `sd15_base` no longer error in `auto`; instead they route to a fallback latent-mask repaint path inside the Pro nodes. Surface the chosen path through `edit_info` and `sample_info`.

**Tech Stack:** Python, unittest, existing ComfyUI node classes, fake tensor/mask helpers

---

### Task 1: Add failing tests for fallback routing

**Files:**
- Modify: `tests/test_pro_edit_registry.py`
- Modify: `tests/test_pro_edit_profile_prepare.py`
- Modify: `tests/test_pro_edit_bridge.py`
- Modify: `tests/test_pro_edit_docs.py`

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Run focused unittest files and confirm the current implementation rejects base profiles**
- [ ] **Step 3: Only proceed once the failures show the missing fallback path**

### Task 2: Implement Pro fallback execution path

**Files:**
- Modify: `pro_edit/backends/base.py`
- Create: `pro_edit/backends/generic.py`
- Modify: `pro_edit/backends/registry.py`
- Modify: `pro_edit/backends/sdxl.py`
- Modify: `pro_edit/backends/flux.py`
- Modify: `pro_edit/pro_edit_utils.py`

- [ ] **Step 1: Add `execution_path` to routing metadata**
- [ ] **Step 2: Route `backend_type = none` to family/generic fallback instead of raising**
- [ ] **Step 3: Implement minimal fallback prepare behavior using latent samples + noise mask**
- [ ] **Step 4: Re-run focused tests and confirm green**

### Task 3: Expose fallback path in bridge and docs

**Files:**
- Modify: `pro_edit/pro_edit_bridge.py`
- Modify: `README.md`

- [ ] **Step 1: Include `execution_path` in `sample_info`**
- [ ] **Step 2: Document native vs fallback behavior**
- [ ] **Step 3: Run focused docs/bridge tests**

### Task 4: Verify regressions

**Files:**
- Modify as needed from earlier tasks

- [ ] **Step 1: Run `test_pro_edit*.py`**
- [ ] **Step 2: Run `test_repair*.py`**
- [ ] **Step 3: Run `python3 -m compileall __init__.py pro_edit model_profiles utils model_loader repair sampling`**
