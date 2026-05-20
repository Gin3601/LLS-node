# Contextless Routing Final State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `task_context` and `model_info` execution ports from the core LLS nodes, infer family/task from `MODEL` / `CLIP` / `LATENT`, and keep old workflows compatible through standard ComfyUI ports plus local manual overrides.

**Architecture:** Store only lightweight `_lls_*` tags on native ComfyUI objects, centralize family/task inference in `utils/model_info.py`, and make each node emit JSON info strings instead of passing context objects. Keep the core node graph native: `MODEL`, `CLIP`, `VAE`, `LATENT`, `IMAGE`, and `CONDITIONING`.

**Tech Stack:** Python, ComfyUI custom nodes, `unittest`

---

### Task 1: Shared inference helpers

**Files:**
- Modify: `utils/model_info.py`
- Test: `tests/test_loader_prompt_refactor.py`

- [ ] **Step 1: Keep the failing spec as the red test**

Run: `python3 -B -m unittest tests.test_loader_prompt_refactor`
Expected: FAIL on loader / prompt / latent / sampler signatures

- [ ] **Step 2: Add shared helpers for object tagging and family resolution**

Implement helpers in `utils/model_info.py` for:
- safely tagging `model`, `clip`, `vae`
- resolving effective family from `model_family`, `model`, `clip`
- building JSON-safe info payloads for prompt / latent / sample / decode / upscale

- [ ] **Step 3: Re-run targeted inference tests**

Run: `python3 -B -m unittest tests.test_model_info_inference`
Expected: PASS

### Task 2: Core node API refactor

**Files:**
- Modify: `model_loader/nodes.py`
- Modify: `conditioning/nodes.py`
- Modify: `latent/nodes.py`
- Modify: `sampling/nodes.py`
- Modify: `image/nodes.py`
- Modify: `upscale/nodes.py`
- Modify: `utils/nodes.py`
- Test: `tests/test_loader_prompt_refactor.py`
- Test: `tests/test_conditioning_compat.py`
- Test: `tests/test_upscale_switcher.py`

- [ ] **Step 1: Refactor loader outputs and tagging**

Change `LLSSimpleCheckpointLoader` to return `(MODEL, CLIP, VAE, CLIP)`, remove optional `task_context`, and tag returned native objects with `_lls_family`, `_lls_model_name`, `_lls_text_encoder_type`, and `_lls_vae_name`.

- [ ] **Step 2: Refactor encode / latent / sample / image nodes**

Change each node to:
- accept `model_family="Auto"` as a local override
- infer family from `MODEL` / `CLIP`
- infer task mode from `LATENT["source"]`
- return JSON strings (`prompt_info`, `latent_info`, `sample_info`, `decode_info`)

- [ ] **Step 3: Refactor metadata and config nodes**

Update `LLSSaveImage` and `LLSGenerationConfig` to use native objects plus JSON info strings instead of `task_context/model_info`.

- [ ] **Step 4: Run focused tests**

Run: `python3 -B -m unittest tests.test_loader_prompt_refactor tests.test_conditioning_compat tests.test_upscale_switcher`
Expected: PASS

### Task 3: Registration cleanup and full verification

**Files:**
- Modify: `__init__.py`
- Test: `tests/test_loader_prompt_refactor.py`
- Test: `tests/`

- [ ] **Step 1: Unregister legacy task nodes**

Remove the `task` subpackage from `__init__.py` so `LLSTaskController` and `LLSTaskInspector` are not exposed in contextless mode.

- [ ] **Step 2: Run the full suite**

Run: `python3 -B -m unittest discover -s tests`
Expected: PASS

- [ ] **Step 3: Merge only after green**

If and only if the full suite is green, merge `refactor/contextless-routing` back into `main` and remove the project-local worktree.
