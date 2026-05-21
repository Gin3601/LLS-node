import { app } from "../../../scripts/app.js";

const EXTENSION_NAME = "lls.ksampler.compat";
const TARGET_NODE_CLASS = "LLSSimpleKSampler";
const TARGET_NODE_DISPLAY_NAME = "LLS Simple KSampler";
const DENOISE_MODE_INDEX = 11;
const ADAPTER_MODE_INDEX = 12;
const FLUX_GUIDANCE_INDEX = 13;
const MODEL_FAMILY_INDEX = 14;
const DENOISE_MODES = new Set(["manual", "auto_from_repair"]);
const ADAPTER_MODES = new Set(["auto", "sd_classic", "flux", "sd3", "qwen", "zimage"]);

function normalizeText(value) {
  if (typeof value === "string") {
    return value.trim();
  }
  if (value === null || value === undefined) {
    return "";
  }
  return String(value).trim();
}

function isNumericLike(value) {
  if (typeof value === "number") {
    return Number.isFinite(value);
  }
  const text = normalizeText(value);
  if (!text) {
    return false;
  }
  return Number.isFinite(Number(text));
}

function looksLikeShiftedWidgetOrder(values) {
  if (!Array.isArray(values) || values.length <= MODEL_FAMILY_INDEX) {
    return false;
  }

  const denoiseMode = normalizeText(values[DENOISE_MODE_INDEX]);
  const adapterMode = normalizeText(values[ADAPTER_MODE_INDEX]).toLowerCase();
  const fluxGuidance = normalizeText(values[FLUX_GUIDANCE_INDEX]);
  const modelFamily = normalizeText(values[MODEL_FAMILY_INDEX]).toLowerCase();

  return (
    !DENOISE_MODES.has(denoiseMode) &&
    DENOISE_MODES.has(fluxGuidance) &&
    ADAPTER_MODES.has(modelFamily) &&
    !ADAPTER_MODES.has(adapterMode) &&
    (isNumericLike(values[DENOISE_MODE_INDEX]) || denoiseMode === "")
  );
}

function migrateWidgetValues(values) {
  if (!Array.isArray(values) || values.length <= MODEL_FAMILY_INDEX) {
    return values;
  }

  const migrated = values.slice();
  if (looksLikeShiftedWidgetOrder(migrated)) {
    const shiftedDenoiseMode = migrated[DENOISE_MODE_INDEX];
    const shiftedAdapterMode = migrated[ADAPTER_MODE_INDEX];
    const shiftedFluxGuidance = migrated[FLUX_GUIDANCE_INDEX];
    const shiftedModelFamily = migrated[MODEL_FAMILY_INDEX];

    migrated[DENOISE_MODE_INDEX] = normalizeText(shiftedFluxGuidance) || "manual";
    migrated[ADAPTER_MODE_INDEX] = normalizeText(shiftedModelFamily).toLowerCase() || "auto";
    migrated[FLUX_GUIDANCE_INDEX] = shiftedDenoiseMode;
    migrated[MODEL_FAMILY_INDEX] =
      normalizeText(shiftedAdapterMode).toLowerCase() === "auto" ? "Auto" : shiftedAdapterMode;
    return migrated;
  }

  if (normalizeText(migrated[ADAPTER_MODE_INDEX]) === "Auto") {
    migrated[ADAPTER_MODE_INDEX] = "auto";
  }
  if (normalizeText(migrated[MODEL_FAMILY_INDEX]) === "auto") {
    migrated[MODEL_FAMILY_INDEX] = "Auto";
  }
  return migrated;
}

app.registerExtension({
  name: EXTENSION_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TARGET_NODE_CLASS && nodeData.display_name !== TARGET_NODE_DISPLAY_NAME) {
      return;
    }

    const previousConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function configureLLSKSamplerCompat(info) {
      if (info && Array.isArray(info.widgets_values)) {
        info.widgets_values = migrateWidgetValues(info.widgets_values);
      }
      return previousConfigure?.apply(this, arguments);
    };
  },
});
