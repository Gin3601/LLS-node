# LLS Flux2Klein Background Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a ready-to-use ComfyUI example workflow that performs e-commerce background replacement with `LLSFlux2KleinEditTextEncode` on a fully matched Flux2/Klein model stack.

**Architecture:** Keep the backend node implementation as-is and add a new example workflow JSON plus a workflow regression test. The workflow must use `UNETLoader + CLIPLoader(type=flux2) + VAELoader(flux2-vae)` and derive the edit mask from `RMBG -> InvertMask` so the sampler receives a consistent Flux2/Klein edit payload.

**Tech Stack:** JSON workflow exports, Python `unittest`, existing ComfyUI core nodes, existing LLS custom node registration

---

### Task 1: Lock The Workflow Contract With A Failing Test

**Files:**
- Modify: `tests/test_example_workflows.py`

- [ ] **Step 1: Add a test that loads `LLS-Flux2Klein-电商换背景工作流.json`**
- [ ] **Step 2: Assert the workflow includes `LLSFlux2KleinEditTextEncode`, `UNETLoader`, `CLIPLoader`, `VAELoader`, `RMBG`, `InvertMask`, `ConditioningZeroOut`, `KSampler`, and `VAEDecode`**
- [ ] **Step 3: Assert `CLIPLoader` uses `flux2`, `VAELoader` uses `flux2-vae.safetensors`, and the LLS node uses `mask_mode = use_mask` with `resize_mode = longest_edge`**
- [ ] **Step 4: Run the focused workflow test and confirm it fails before the workflow JSON exists**

### Task 2: Add The New Background-Replacement Workflow

**Files:**
- Create: `LLS-Flux2Klein-电商换背景工作流.json`

- [ ] **Step 1: Build a compact workflow with `LoadImage` main product input feeding both `RMBG` and `LLSFlux2KleinEditTextEncode.image1`**
- [ ] **Step 2: Add one connected scene reference image for `image2` and leave `image3` optional**
- [ ] **Step 3: Use `RMBG -> InvertMask -> LLSFlux2KleinEditTextEncode.mask` and route `conditioning`, `ConditioningZeroOut`, `latent`, `KSampler`, and `VAEDecode` according to the edit flow contract**
- [ ] **Step 4: Route the model through a Flux2/Klein-safe sampler chain and add `PreviewImage` plus `SaveImage` outputs**

### Task 3: Verify The Workflow And Summarize Constraints

**Files:**
- Modify: files from earlier tasks only if verification exposes gaps

- [ ] **Step 1: Run `python3 -m unittest tests.test_example_workflows -v`**
- [ ] **Step 2: Run `python3 -m json.tool 'LLS-Flux2Klein-电商换背景工作流.json' > /tmp/lls_flux2klein_background_workflow.pretty.json` to validate JSON syntax**
- [ ] **Step 3: Re-run the targeted Flux2/Klein unit tests if the workflow change touches shared assumptions**
- [ ] **Step 4: Summarize the new workflow file, ComfyUI search name, minimal connection path, and the remaining compatibility-vs-native Flux2 API gaps**
