import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXTENSION_NAME = "LLS.MaskDraw";
const TARGET_NODE_CLASS = "LLSSimpleMaskDraw";
const TARGET_NODE_DISPLAY_NAME = "LLS Simple Mask Draw";
const HISTORY_LIMIT = 20;
const MAX_PREVIEW_EDGE = 420;
const MIN_PREVIEW_EDGE = 220;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function asNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function asBoolean(value, fallback = false) {
  return typeof value === "boolean" ? value : fallback;
}

function findWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name) ?? null;
}

function getNodeClassName(node) {
  return String(node?.comfyClass || node?.constructor?.comfyClass || node?.type || node?.title || "");
}

function hideWidget(widget) {
  if (!widget) {
    return;
  }
  widget.type = "hidden";
  widget.computeSize = () => [0, -4];
}

function chainWidgetCallback(widget, handler) {
  if (!widget || widget.__llsMaskDrawChained) {
    return;
  }
  const previous = widget.callback;
  widget.callback = function chainedMaskDrawCallback() {
    previous?.apply(this, arguments);
    handler?.apply(this, arguments);
  };
  widget.__llsMaskDrawChained = true;
}

function parseState(rawValue) {
  try {
    const parsed = JSON.parse(rawValue || "{}");
    if (!parsed || typeof parsed !== "object") {
      throw new Error("Invalid state payload");
    }
    return {
      version: asNumber(parsed.version, 1),
      mask_png_base64: String(parsed.mask_png_base64 || ""),
      touched: asBoolean(parsed.touched, false),
      editor: parsed.editor && typeof parsed.editor === "object" ? parsed.editor : {},
    };
  } catch {
    return {
      version: 1,
      mask_png_base64: "",
      touched: false,
      editor: {},
    };
  }
}

function dataUrlToBase64(dataUrl) {
  const commaIndex = String(dataUrl || "").indexOf(",");
  return commaIndex >= 0 ? dataUrl.slice(commaIndex + 1) : "";
}

function getInputLink(node, inputName) {
  const input = node.inputs?.find((item) => item.name === inputName);
  if (!input || input.link == null) {
    return null;
  }
  return node.graph?.links?.[input.link] ?? null;
}

function getUpstreamNode(node, inputName) {
  const link = getInputLink(node, inputName);
  if (!link) {
    return null;
  }
  return node.graph?.getNodeById?.(link.origin_id) ?? null;
}

function normalizePath(rawValue) {
  return String(rawValue || "").trim().replace(/\\/g, "/");
}

function splitAnnotatedPath(rawValue) {
  const normalized = normalizePath(rawValue);
  const match = normalized.match(/^(.*?)(?:\s+\[([^\]]+)\])?$/);
  const path = match?.[1] ?? normalized;
  const annotatedType = match?.[2] ?? "";
  const slashIndex = path.lastIndexOf("/");
  return {
    path,
    annotatedType,
    subfolder: slashIndex >= 0 ? path.slice(0, slashIndex) : "",
    filename: slashIndex >= 0 ? path.slice(slashIndex + 1) : path,
  };
}

function buildViewUrl(rawValue, fallbackType, extraParams = {}) {
  const normalized = normalizePath(rawValue);
  if (!normalized) {
    return null;
  }

  const params = new URLSearchParams();
  if (normalized.startsWith("blake3:")) {
    params.set("filename", normalized);
  } else {
    const parsed = splitAnnotatedPath(normalized);
    if (!parsed.filename) {
      return null;
    }
    params.set("filename", parsed.filename);
    if (parsed.subfolder) {
      params.set("subfolder", parsed.subfolder);
    }
    params.set("type", parsed.annotatedType || fallbackType || "input");
  }

  for (const [key, value] of Object.entries(extraParams)) {
    if (value != null && value !== "") {
      params.set(key, String(value));
    }
  }

  const relativePath = `/view?${params.toString()}`;
  return typeof api?.apiURL === "function" ? api.apiURL(relativePath) : relativePath;
}

function resolveImageSource(node) {
  const upstream = getUpstreamNode(node, "image");
  const className = getNodeClassName(upstream);
  const imageWidget = findWidget(upstream, "image");
  if (!imageWidget?.value) {
    return null;
  }

  if (className === "LoadImage") {
    return {
      url: buildViewUrl(imageWidget.value, "input"),
      label: "LoadImage",
    };
  }

  if (className === "LoadImageOutput") {
    return {
      url: buildViewUrl(imageWidget.value, "output"),
      label: "LoadImageOutput",
    };
  }

  return null;
}

function resolveMaskSource(node) {
  const upstream = getUpstreamNode(node, "input_mask");
  const className = getNodeClassName(upstream);
  const imageWidget = findWidget(upstream, "image");
  if (!imageWidget?.value) {
    return null;
  }

  if (className === "LoadImageMask") {
    const channelName = String(findWidget(upstream, "channel")?.value || "alpha").toLowerCase();
    if (channelName === "alpha") {
      return {
        url: buildViewUrl(imageWidget.value, "input", { channel: "a" }),
        channel: "a",
        invert: true,
        label: "LoadImageMask.alpha",
      };
    }

    const channelMap = {
      red: "r",
      green: "g",
      blue: "b",
    };
    const mappedChannel = channelMap[channelName];
    if (!mappedChannel) {
      return null;
    }

    return {
      url: buildViewUrl(imageWidget.value, "input"),
      channel: mappedChannel,
      invert: false,
      label: `LoadImageMask.${channelName}`,
    };
  }

  if (className === "LoadImage") {
    return {
      url: buildViewUrl(imageWidget.value, "input", { channel: "a" }),
      channel: "a",
      invert: true,
      label: "LoadImage.alpha",
    };
  }

  return null;
}

function loadImageElement(url) {
  return new Promise((resolve, reject) => {
    if (!url) {
      resolve(null);
      return;
    }

    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Failed to load image: ${url}`));
    image.src = url;
  });
}

function computeDisplaySize(width, height) {
  const safeWidth = Math.max(1, Math.round(width));
  const safeHeight = Math.max(1, Math.round(height));
  const maxEdge = Math.max(safeWidth, safeHeight);
  let scale = 1;

  if (maxEdge > MAX_PREVIEW_EDGE) {
    scale = MAX_PREVIEW_EDGE / maxEdge;
  } else if (maxEdge < MIN_PREVIEW_EDGE) {
    scale = MIN_PREVIEW_EDGE / maxEdge;
  }

  return {
    width: Math.max(1, Math.round(safeWidth * scale)),
    height: Math.max(1, Math.round(safeHeight * scale)),
  };
}

function drawHintText(context, lines) {
  context.fillStyle = "#171717";
  context.fillRect(0, 0, context.canvas.width, context.canvas.height);
  context.fillStyle = "#9a9a9a";
  context.font = "12px sans-serif";
  lines.forEach((line, index) => {
    context.fillText(line, 14, 24 + index * 18);
  });
}

function drawMaskOverlay(stageContext, overlayContext, overlayCanvas, maskCanvas, overlayAlpha, invertMask) {
  if (!maskCanvas.width || !maskCanvas.height) {
    return;
  }

  overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  overlayContext.drawImage(maskCanvas, 0, 0, overlayCanvas.width, overlayCanvas.height);
  const imageData = overlayContext.getImageData(0, 0, overlayCanvas.width, overlayCanvas.height);
  const pixels = imageData.data;

  for (let index = 0; index < pixels.length; index += 4) {
    let value = pixels[index];
    if (invertMask) {
      value = 255 - value;
    }
    pixels[index] = 255;
    pixels[index + 1] = 0;
    pixels[index + 2] = 0;
    pixels[index + 3] = Math.round((value / 255) * overlayAlpha * 255);
  }

  overlayContext.putImageData(imageData, 0, 0);
  stageContext.drawImage(overlayCanvas, 0, 0);
}

function attachMaskEditor(node) {
  if (node.__llsMaskDrawEditor || typeof node.addDOMWidget !== "function") {
    return node.__llsMaskDrawEditor ?? null;
  }

  const stateWidget = findWidget(node, "mask_state_json");
  const drawModeWidget = findWidget(node, "draw_mode");
  const brushSizeWidget = findWidget(node, "brush_size");
  const brushSoftnessWidget = findWidget(node, "brush_softness");
  const overlayAlphaWidget = findWidget(node, "overlay_alpha");
  const invertWidget = findWidget(node, "invert_mask");

  hideWidget(stateWidget);

  const savedState = parseState(stateWidget?.value);

  const container = document.createElement("div");
  container.className = "lls-mask-draw-editor";
  container.style.display = "flex";
  container.style.flexDirection = "column";
  container.style.alignItems = "flex-start";
  container.style.gap = "8px";
  container.style.width = "100%";
  container.style.boxSizing = "border-box";

  const toolbar = document.createElement("div");
  toolbar.style.display = "flex";
  toolbar.style.flexWrap = "wrap";
  toolbar.style.gap = "6px";
  toolbar.style.width = "100%";

  const frame = document.createElement("div");
  frame.style.display = "inline-flex";
  frame.style.border = "1px solid #4e4e4e";
  frame.style.borderRadius = "8px";
  frame.style.overflow = "hidden";
  frame.style.background = "#111";

  const stage = document.createElement("canvas");
  stage.style.display = "block";
  stage.style.cursor = "crosshair";
  stage.style.touchAction = "none";
  stage.style.userSelect = "none";
  frame.appendChild(stage);

  const status = document.createElement("div");
  status.style.fontSize = "12px";
  status.style.lineHeight = "1.45";
  status.style.color = "#b8b8b8";
  status.style.maxWidth = "420px";
  status.style.whiteSpace = "normal";

  container.append(toolbar, frame, status);

  const domWidget = node.addDOMWidget("lls_mask_editor", "lls_mask_editor", container);
  domWidget.serialize = false;
  if (domWidget.options) {
    domWidget.options.hideOnZoom = false;
  }

  const stageContext = stage.getContext("2d");
  const overlayCanvas = document.createElement("canvas");
  const overlayContext = overlayCanvas.getContext("2d", { willReadFrequently: true });
  const maskCanvas = document.createElement("canvas");
  let maskContext = maskCanvas.getContext("2d", { willReadFrequently: true });

  const history = {
    undo: [],
    redo: [],
  };

  const state = {
    baseImage: null,
    localMaskActive: Boolean(savedState.touched && savedState.mask_png_base64),
    drawMode: String(savedState.editor.draw_mode || "brush"),
    brushSize: clamp(asNumber(savedState.editor.brush_size, 32), 1, 512),
    brushSoftness: clamp(asNumber(savedState.editor.brush_softness, 0.5), 0, 1),
    overlayAlpha: clamp(asNumber(savedState.editor.overlay_alpha, 0.4), 0, 1),
    drawing: false,
    dirtyStroke: false,
    lastPoint: null,
    loadToken: 0,
  };

  let canPersist = false;

  function updateStatus(message) {
    status.textContent = message;
  }

  function resetHistory() {
    history.undo.length = 0;
    history.redo.length = 0;
  }

  function fillMaskBlack() {
    if (!maskCanvas.width || !maskCanvas.height || !maskContext) {
      return;
    }
    maskContext.save();
    maskContext.globalCompositeOperation = "source-over";
    maskContext.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    maskContext.fillStyle = "rgb(0, 0, 0)";
    maskContext.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    maskContext.restore();
  }

  function ensureMaskCanvasSize(width, height, options = {}) {
    const preserve = options.preserve !== false;
    const nextWidth = Math.max(1, Math.round(width));
    const nextHeight = Math.max(1, Math.round(height));
    if (maskCanvas.width === nextWidth && maskCanvas.height === nextHeight) {
      return;
    }

    let previousCanvas = null;
    if (preserve && maskCanvas.width > 0 && maskCanvas.height > 0) {
      previousCanvas = document.createElement("canvas");
      previousCanvas.width = maskCanvas.width;
      previousCanvas.height = maskCanvas.height;
      previousCanvas.getContext("2d").drawImage(maskCanvas, 0, 0);
    }

    maskCanvas.width = nextWidth;
    maskCanvas.height = nextHeight;
    maskContext = maskCanvas.getContext("2d", { willReadFrequently: true });
    fillMaskBlack();

    if (previousCanvas) {
      maskContext.drawImage(previousCanvas, 0, 0, nextWidth, nextHeight);
    }
  }

  function snapshotMask() {
    if (!maskCanvas.width || !maskCanvas.height) {
      return "";
    }
    return maskCanvas.toDataURL("image/png");
  }

  function persistState() {
    if (!canPersist || !stateWidget) {
      return;
    }

    stateWidget.value = JSON.stringify({
      version: 1,
      mask_png_base64: state.localMaskActive ? dataUrlToBase64(snapshotMask()) : "",
      touched: state.localMaskActive,
      editor: {
        draw_mode: state.drawMode,
        brush_size: state.brushSize,
        brush_softness: state.brushSoftness,
        overlay_alpha: state.overlayAlpha,
      },
    });

    app.graph?.setDirtyCanvas(true, true);
  }

  function readWidgetValues(options = {}) {
    const persist = options.persist !== false;
    state.drawMode = String(drawModeWidget?.value ?? state.drawMode || "brush");
    state.brushSize = clamp(asNumber(brushSizeWidget?.value, state.brushSize || 32), 1, 512);
    state.brushSoftness = clamp(asNumber(brushSoftnessWidget?.value, state.brushSoftness ?? 0.5), 0, 1);
    state.overlayAlpha = clamp(asNumber(overlayAlphaWidget?.value, state.overlayAlpha ?? 0.4), 0, 1);
    if (persist) {
      persistState();
    }
  }

  function updateNodeSize() {
    const sourceWidth = state.baseImage?.naturalWidth || maskCanvas.width || 320;
    const sourceHeight = state.baseImage?.naturalHeight || maskCanvas.height || 240;
    const displaySize = computeDisplaySize(sourceWidth, sourceHeight);

    stage.width = displaySize.width;
    stage.height = displaySize.height;
    stage.style.width = `${displaySize.width}px`;
    stage.style.height = `${displaySize.height}px`;
    overlayCanvas.width = displaySize.width;
    overlayCanvas.height = displaySize.height;
    frame.style.width = `${displaySize.width}px`;
    container.style.width = `${displaySize.width}px`;

    if (typeof node.setSize === "function") {
      requestAnimationFrame(() => {
        node.setSize([
          Math.max(360, displaySize.width + 28),
          Math.max(250, displaySize.height + 150),
        ]);
        app.graph?.setDirtyCanvas(true, true);
      });
    }
  }

  function renderStage() {
    updateNodeSize();
    stageContext.clearRect(0, 0, stage.width, stage.height);

    if (state.baseImage) {
      stageContext.drawImage(state.baseImage, 0, 0, stage.width, stage.height);
    } else {
      drawHintText(stageContext, state.localMaskActive
        ? [
            "Source preview unavailable.",
            "Connect Load Image / Load Image Output",
            "to show the edited mask over the image.",
          ]
        : [
            "Connect Load Image / Load Image Output",
            "to start drawing before execution.",
          ]);
    }

    drawMaskOverlay(
      stageContext,
      overlayContext,
      overlayCanvas,
      maskCanvas,
      state.overlayAlpha,
      asBoolean(invertWidget?.value, false),
    );

    app.graph?.setDirtyCanvas(true, true);
  }

  function renderSourceMask(image, channel, invert) {
    if (!maskCanvas.width || !maskCanvas.height) {
      return;
    }

    const scratchCanvas = document.createElement("canvas");
    scratchCanvas.width = maskCanvas.width;
    scratchCanvas.height = maskCanvas.height;
    const scratchContext = scratchCanvas.getContext("2d", { willReadFrequently: true });
    scratchContext.drawImage(image, 0, 0, scratchCanvas.width, scratchCanvas.height);

    const imageData = scratchContext.getImageData(0, 0, scratchCanvas.width, scratchCanvas.height);
    const pixels = imageData.data;

    for (let index = 0; index < pixels.length; index += 4) {
      let value = pixels[index];
      if (channel === "g") {
        value = pixels[index + 1];
      } else if (channel === "b") {
        value = pixels[index + 2];
      } else if (channel === "a") {
        value = pixels[index + 3];
      }
      if (invert) {
        value = 255 - value;
      }
      pixels[index] = value;
      pixels[index + 1] = value;
      pixels[index + 2] = value;
      pixels[index + 3] = 255;
    }

    scratchContext.putImageData(imageData, 0, 0);
    fillMaskBlack();
    maskContext.drawImage(scratchCanvas, 0, 0);
  }

  async function restoreMaskFromDataUrl(dataUrl) {
    const image = await loadImageElement(dataUrl);
    if (!image) {
      return false;
    }

    ensureMaskCanvasSize(image.naturalWidth || image.width, image.naturalHeight || image.height, { preserve: false });
    fillMaskBlack();
    maskContext.drawImage(image, 0, 0, maskCanvas.width, maskCanvas.height);
    return true;
  }

  async function restoreSerializedLocalMask(maskBase64) {
    if (!maskBase64) {
      return false;
    }

    try {
      const restored = await restoreMaskFromDataUrl(`data:image/png;base64,${maskBase64}`);
      if (restored) {
        resetHistory();
      }
      return restored;
    } catch {
      return false;
    }
  }

  function pushUndoSnapshot() {
    const snapshot = snapshotMask();
    if (!snapshot) {
      return;
    }
    history.undo.push(snapshot);
    if (history.undo.length > HISTORY_LIMIT) {
      history.undo.shift();
    }
    history.redo.length = 0;
  }

  async function restoreHistorySnapshot(snapshot) {
    if (!snapshot) {
      return false;
    }
    const restored = await restoreMaskFromDataUrl(snapshot);
    renderStage();
    return restored;
  }

  function getMaskPoint(event) {
    if (!maskCanvas.width || !maskCanvas.height) {
      return null;
    }

    const bounds = stage.getBoundingClientRect();
    if (!bounds.width || !bounds.height) {
      return null;
    }

    return {
      x: clamp((event.clientX - bounds.left) * (maskCanvas.width / bounds.width), 0, maskCanvas.width),
      y: clamp((event.clientY - bounds.top) * (maskCanvas.height / bounds.height), 0, maskCanvas.height),
    };
  }

  function applyBrushStamp(x, y) {
    const radius = Math.max(1, state.brushSize) / 2;
    const edge = clamp(1 - state.brushSoftness, 0, 1);
    const channelValue = state.drawMode === "erase" ? 0 : 255;
    const gradient = maskContext.createRadialGradient(x, y, 0, x, y, radius);
    gradient.addColorStop(0, `rgba(${channelValue}, ${channelValue}, ${channelValue}, 1)`);
    gradient.addColorStop(edge, `rgba(${channelValue}, ${channelValue}, ${channelValue}, 1)`);
    gradient.addColorStop(1, `rgba(${channelValue}, ${channelValue}, ${channelValue}, 0)`);

    maskContext.save();
    maskContext.globalCompositeOperation = "source-over";
    maskContext.fillStyle = gradient;
    maskContext.beginPath();
    maskContext.arc(x, y, radius, 0, Math.PI * 2);
    maskContext.fill();
    maskContext.restore();
  }

  function applyStroke(fromPoint, toPoint) {
    if (!toPoint) {
      return;
    }

    if (!fromPoint) {
      applyBrushStamp(toPoint.x, toPoint.y);
      return;
    }

    const deltaX = toPoint.x - fromPoint.x;
    const deltaY = toPoint.y - fromPoint.y;
    const distance = Math.hypot(deltaX, deltaY);
    const spacing = Math.max(1, state.brushSize * 0.2);
    const steps = Math.max(1, Math.ceil(distance / spacing));

    for (let step = 1; step <= steps; step += 1) {
      const progress = step / steps;
      applyBrushStamp(
        fromPoint.x + deltaX * progress,
        fromPoint.y + deltaY * progress,
      );
    }
  }

  async function reloadFromInputs() {
    const requestToken = ++state.loadToken;
    const imageSource = resolveImageSource(node);
    const maskSource = state.localMaskActive ? null : resolveMaskSource(node);

    const baseImagePromise = imageSource?.url
      ? loadImageElement(imageSource.url).catch(() => null)
      : Promise.resolve(null);
    const maskImagePromise = maskSource?.url
      ? loadImageElement(maskSource.url).catch(() => null)
      : Promise.resolve(null);

    const [baseImage, maskImage] = await Promise.all([baseImagePromise, maskImagePromise]);
    if (requestToken !== state.loadToken) {
      return;
    }

    state.baseImage = baseImage;

    const targetWidth = baseImage?.naturalWidth || maskCanvas.width || maskImage?.naturalWidth || maskImage?.width || 0;
    const targetHeight = baseImage?.naturalHeight || maskCanvas.height || maskImage?.naturalHeight || maskImage?.height || 0;

    if (targetWidth && targetHeight) {
      if (state.localMaskActive) {
        ensureMaskCanvasSize(targetWidth, targetHeight, { preserve: true });
      } else {
        resetHistory();
        ensureMaskCanvasSize(targetWidth, targetHeight, { preserve: false });
        if (maskImage) {
          renderSourceMask(maskImage, maskSource.channel, maskSource.invert);
        }
      }
    }

    if (!imageSource) {
      updateStatus(state.localMaskActive
        ? "Local mask restored. Connect Load Image or Load Image Output for source preview."
        : "Connect Load Image or Load Image Output to paint before execution.");
    } else if (!baseImage) {
      updateStatus("Source preview could not be loaded from the upstream file-backed node.");
    } else if (state.localMaskActive) {
      updateStatus("Editing the local mask raster. input_mask is ignored until this local mask is replaced.");
    } else if (maskSource && maskImage) {
      updateStatus("input_mask loaded as the editable base layer.");
    } else if (maskSource && !maskImage) {
      updateStatus("Source image loaded. input_mask preview is unavailable for this upstream node.");
    } else {
      updateStatus("Source image loaded. Drawing starts from an empty mask.");
    }

    renderStage();
  }

  function createButton(label, handler) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.style.border = "1px solid #5d5d5d";
    button.style.borderRadius = "6px";
    button.style.background = "#202020";
    button.style.color = "#f0f0f0";
    button.style.padding = "4px 10px";
    button.style.cursor = "pointer";
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      handler(event);
    });
    return button;
  }

  toolbar.append(
    createButton("Clear", () => {
      if (!maskCanvas.width || !maskCanvas.height) {
        return;
      }
      pushUndoSnapshot();
      fillMaskBlack();
      state.localMaskActive = true;
      persistState();
      updateStatus("Local mask cleared to black.");
      renderStage();
    }),
    createButton("Undo", async () => {
      if (!history.undo.length) {
        return;
      }
      const previous = history.undo.pop();
      const current = snapshotMask();
      if (current) {
        history.redo.push(current);
      }
      if (await restoreHistorySnapshot(previous)) {
        state.localMaskActive = true;
        persistState();
        updateStatus("Restored the previous local mask state.");
      }
    }),
    createButton("Redo", async () => {
      if (!history.redo.length) {
        return;
      }
      const next = history.redo.pop();
      const current = snapshotMask();
      if (current) {
        history.undo.push(current);
      }
      if (await restoreHistorySnapshot(next)) {
        state.localMaskActive = true;
        persistState();
        updateStatus("Reapplied the next local mask state.");
      }
    }),
    createButton("Invert", () => {
      if (!invertWidget) {
        return;
      }
      invertWidget.value = !Boolean(invertWidget.value);
      invertWidget.callback?.(invertWidget.value);
      renderStage();
    }),
  );

  stage.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    event.stopPropagation();

    const point = getMaskPoint(event);
    if (!point) {
      updateStatus("Source image is not ready. Connect Load Image or Load Image Output first.");
      renderStage();
      return;
    }

    stage.setPointerCapture?.(event.pointerId);
    pushUndoSnapshot();
    state.drawing = true;
    state.dirtyStroke = true;
    state.localMaskActive = true;
    applyStroke(null, point);
    state.lastPoint = point;
    renderStage();
  });

  stage.addEventListener("pointermove", (event) => {
    if (!state.drawing) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();

    const point = getMaskPoint(event);
    if (!point) {
      return;
    }

    applyStroke(state.lastPoint, point);
    state.lastPoint = point;
    renderStage();
  });

  function endStroke(event) {
    if (!state.drawing) {
      return;
    }
    event?.preventDefault?.();
    event?.stopPropagation?.();

    state.drawing = false;
    state.lastPoint = null;
    if (state.dirtyStroke) {
      persistState();
      state.dirtyStroke = false;
      updateStatus("Local mask updated.");
    }
    renderStage();
  }

  stage.addEventListener("pointerup", endStroke);
  stage.addEventListener("pointercancel", endStroke);
  stage.addEventListener("lostpointercapture", endStroke);

  chainWidgetCallback(drawModeWidget, () => {
    readWidgetValues({ persist: true });
    renderStage();
  });
  chainWidgetCallback(brushSizeWidget, () => {
    readWidgetValues({ persist: true });
    renderStage();
  });
  chainWidgetCallback(brushSoftnessWidget, () => {
    readWidgetValues({ persist: true });
    renderStage();
  });
  chainWidgetCallback(overlayAlphaWidget, () => {
    readWidgetValues({ persist: true });
    renderStage();
  });
  chainWidgetCallback(invertWidget, () => {
    renderStage();
  });

  async function initializeFromState(syncSerializedState) {
    const parsedState = syncSerializedState ? parseState(stateWidget?.value) : savedState;
    readWidgetValues({ persist: false });

    if (parsedState.touched && parsedState.mask_png_base64) {
      state.localMaskActive = await restoreSerializedLocalMask(parsedState.mask_png_base64);
    } else if (!state.localMaskActive) {
      state.localMaskActive = false;
    }

    await reloadFromInputs();
    canPersist = true;
    renderStage();
  }

  const editorApi = {
    handleGraphConfigured() {
      void initializeFromState(true);
    },
    handleConnectionsChange() {
      void reloadFromInputs();
    },
  };

  node.__llsMaskDrawEditor = editorApi;
  updateStatus("Connect Load Image or Load Image Output to start drawing before execution.");
  renderStage();
  void initializeFromState(false);
  return editorApi;
}

app.registerExtension({
  name: EXTENSION_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TARGET_NODE_CLASS && nodeData.display_name !== TARGET_NODE_DISPLAY_NAME) {
      return;
    }

    const previousOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function onMaskDrawNodeCreated() {
      const result = previousOnNodeCreated?.apply(this, arguments);
      attachMaskEditor(this);
      return result;
    };

    const previousOnGraphConfigured = nodeType.prototype.onGraphConfigured;
    nodeType.prototype.onGraphConfigured = function onMaskDrawGraphConfigured() {
      const result = previousOnGraphConfigured?.apply(this, arguments);
      attachMaskEditor(this)?.handleGraphConfigured?.();
      return result;
    };

    const previousOnConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function onMaskDrawConnectionsChange() {
      const result = previousOnConnectionsChange?.apply(this, arguments);
      attachMaskEditor(this)?.handleConnectionsChange?.();
      return result;
    };
  },
});
