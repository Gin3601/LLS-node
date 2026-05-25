import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXTENSION_NAME = "LLS.ImageComposite";
const TARGET_NODE_CLASS = "LLSSimpleImageComposite";
const TARGET_NODE_DISPLAY_NAME = "LLS Simple Image Composite";
const MAX_PREVIEW_EDGE = 420;
const MIN_PREVIEW_EDGE = 220;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function asNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function findWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name) ?? null;
}

function getNodeClassName(node) {
  return String(node?.comfyClass || node?.constructor?.comfyClass || node?.type || node?.title || "");
}

function chainWidgetCallback(widget, handler) {
  if (!widget || widget.__llsImageCompositeChained) {
    return;
  }

  const previous = widget.callback;
  widget.callback = function chainedImageCompositeCallback() {
    previous?.apply(this, arguments);
    handler?.apply(this, arguments);
  };
  widget.__llsImageCompositeChained = true;
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

function buildViewUrl(rawValue, fallbackType) {
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

  const relativePath = `/view?${params.toString()}`;
  return typeof api?.apiURL === "function" ? api.apiURL(relativePath) : relativePath;
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

function resolveImageSource(node, inputName) {
  const upstream = getUpstreamNode(node, inputName);
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

function drawHint(context, lines) {
  context.clearRect(0, 0, context.canvas.width, context.canvas.height);
  context.fillStyle = "#151515";
  context.fillRect(0, 0, context.canvas.width, context.canvas.height);
  context.fillStyle = "#b7b7b7";
  context.font = "12px sans-serif";
  lines.forEach((line, index) => {
    context.fillText(line, 14, 24 + index * 18);
  });
}

function attachCompositePreview(node) {
  if (node.__llsImageCompositePreview || typeof node.addDOMWidget !== "function") {
    return node.__llsImageCompositePreview ?? null;
  }

  const xWidget = findWidget(node, "x_offset");
  const yWidget = findWidget(node, "y_offset");
  const anchorWidget = findWidget(node, "anchor_mode");
  const rotationOriginWidget = findWidget(node, "rotation_origin_mode");
  const opacityWidget = findWidget(node, "opacity");
  const scaleWidget = findWidget(node, "scale");
  const rotationWidget = findWidget(node, "rotation");

  const container = document.createElement("div");
  container.style.display = "flex";
  container.style.flexDirection = "column";
  container.style.alignItems = "flex-start";
  container.style.gap = "8px";
  container.style.width = "100%";
  container.style.boxSizing = "border-box";

  const frame = document.createElement("div");
  frame.style.display = "inline-flex";
  frame.style.border = "1px solid #4e4e4e";
  frame.style.borderRadius = "8px";
  frame.style.overflow = "hidden";
  frame.style.background = "#111";

  const canvas = document.createElement("canvas");
  canvas.style.display = "block";
  canvas.style.cursor = "grab";
  canvas.style.touchAction = "none";
  canvas.style.userSelect = "none";
  frame.appendChild(canvas);

  const status = document.createElement("div");
  status.style.fontSize = "12px";
  status.style.lineHeight = "1.45";
  status.style.color = "#b8b8b8";
  status.style.maxWidth = "420px";
  status.style.whiteSpace = "normal";

  container.append(frame, status);

  const domWidget = node.addDOMWidget("lls_image_composite_preview", "preview", container);
  domWidget.serialize = false;
  if (domWidget.options) {
    domWidget.options.hideOnZoom = false;
  }

  const context = canvas.getContext("2d");

  const state = {
    backgroundImage: null,
    overlayImage: null,
    backgroundSource: null,
    overlaySource: null,
    backgroundConnectionSupported: true,
    overlayConnectionSupported: true,
    loadToken: 0,
    dragging: false,
    dragStartClientX: 0,
    dragStartClientY: 0,
    dragStartOffsetX: 0,
    dragStartOffsetY: 0,
  };

  function updateStatus(message) {
    status.textContent = message;
  }

  function updateNodeSize() {
    const width = state.backgroundImage?.naturalWidth || 320;
    const height = state.backgroundImage?.naturalHeight || 220;
    const displaySize = computeDisplaySize(width, height);

    canvas.width = displaySize.width;
    canvas.height = displaySize.height;
    canvas.style.width = `${displaySize.width}px`;
    canvas.style.height = `${displaySize.height}px`;
    frame.style.width = `${displaySize.width}px`;
    container.style.width = `${displaySize.width}px`;

    if (typeof node.setSize === "function") {
      requestAnimationFrame(() => {
        node.setSize([
          Math.max(360, displaySize.width + 28),
          Math.max(250, displaySize.height + 84),
        ]);
        app.graph?.setDirtyCanvas(true, true);
      });
    }
  }

  function renderPreview() {
    updateNodeSize();

    const backgroundImage = state.backgroundImage;
    const overlayImage = state.overlayImage;
    if (!backgroundImage || !overlayImage) {
      const lines = [];
      if (!getInputLink(node, "background_image") || !getInputLink(node, "overlay_image")) {
        lines.push("Connect background_image and overlay_image");
        lines.push("from Load Image / Load Image Output for preview.");
      } else {
        lines.push("Preview supports Load Image / Load Image Output");
        lines.push("sources in this first version.");
      }
      drawHint(context, lines);
      return;
    }

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(backgroundImage, 0, 0, canvas.width, canvas.height);

    const backgroundWidth = Math.max(1, backgroundImage.naturalWidth || backgroundImage.width || canvas.width);
    const backgroundHeight = Math.max(1, backgroundImage.naturalHeight || backgroundImage.height || canvas.height);
    const canvasScaleX = canvas.width / backgroundWidth;
    const canvasScaleY = canvas.height / backgroundHeight;
    const scaleValue = Math.max(0.01, asNumber(scaleWidget?.value, 1));
    const rotationValue = asNumber(rotationWidget?.value, 0);
    const opacityValue = clamp(asNumber(opacityWidget?.value, 1), 0, 1);
    const anchorMode = String(anchorWidget?.value || "top_left");
    const rotationOriginMode = String(rotationOriginWidget?.value || "center");
    const xOffset = asNumber(xWidget?.value, 0);
    const yOffset = asNumber(yWidget?.value, 0);

    const overlayWidth = Math.max(1, (overlayImage.naturalWidth || overlayImage.width || 1) * scaleValue);
    const overlayHeight = Math.max(1, (overlayImage.naturalHeight || overlayImage.height || 1) * scaleValue);
    const topLeftX = anchorMode === "center" ? xOffset - overlayWidth / 2 : xOffset;
    const topLeftY = anchorMode === "center" ? yOffset - overlayHeight / 2 : yOffset;
    const rotationCenterX = rotationOriginMode === "center" ? topLeftX + overlayWidth / 2 : topLeftX;
    const rotationCenterY = rotationOriginMode === "center" ? topLeftY + overlayHeight / 2 : topLeftY;

    context.save();
    context.globalAlpha = opacityValue;
    context.translate(rotationCenterX * canvasScaleX, rotationCenterY * canvasScaleY);
    context.rotate((rotationValue * Math.PI) / 180);
    context.translate(-rotationCenterX * canvasScaleX, -rotationCenterY * canvasScaleY);
    context.drawImage(
      overlayImage,
      topLeftX * canvasScaleX,
      topLeftY * canvasScaleY,
      overlayWidth * canvasScaleX,
      overlayHeight * canvasScaleY,
    );
    context.restore();
  }

  function setWidgetValue(widget, value) {
    if (!widget) {
      return;
    }
    widget.value = value;
    widget.callback?.(value);
  }

  async function reloadSources() {
    const backgroundSource = resolveImageSource(node, "background_image");
    const overlaySource = resolveImageSource(node, "overlay_image");
    state.backgroundSource = backgroundSource?.url ?? null;
    state.overlaySource = overlaySource?.url ?? null;
    state.backgroundConnectionSupported = !getInputLink(node, "background_image") || Boolean(backgroundSource?.url);
    state.overlayConnectionSupported = !getInputLink(node, "overlay_image") || Boolean(overlaySource?.url);

    const token = ++state.loadToken;
    updateStatus("Loading composite preview...");

    try {
      const [backgroundImage, overlayImage] = await Promise.all([
        loadImageElement(state.backgroundSource),
        loadImageElement(state.overlaySource),
      ]);
      if (token !== state.loadToken) {
        return;
      }

      state.backgroundImage = backgroundImage;
      state.overlayImage = overlayImage;
      if (backgroundImage && overlayImage) {
        const backgroundLabel = backgroundSource?.label || "background";
        const overlayLabel = overlaySource?.label || "overlay";
        updateStatus(
          `Drag to update x_offset / y_offset. Preview source: ${backgroundLabel} + ${overlayLabel}. ` +
          "Use scale, rotation, opacity, anchor_mode, and rotation_origin_mode widgets to refine placement.",
        );
      } else if (!state.backgroundConnectionSupported || !state.overlayConnectionSupported) {
        updateStatus("Preview supports Load Image / Load Image Output sources in this first version.");
      } else {
        updateStatus("Connect Load Image / Load Image Output to both inputs for preview.");
      }
      renderPreview();
    } catch (error) {
      if (token !== state.loadToken) {
        return;
      }
      state.backgroundImage = null;
      state.overlayImage = null;
      updateStatus(error?.message || "Failed to load composite preview sources.");
      renderPreview();
    }
  }

  function handlePointerDown(event) {
    if (!state.backgroundImage || !state.overlayImage) {
      return;
    }

    state.dragging = true;
    state.dragStartClientX = event.clientX;
    state.dragStartClientY = event.clientY;
    state.dragStartOffsetX = asNumber(xWidget?.value, 0);
    state.dragStartOffsetY = asNumber(yWidget?.value, 0);
    canvas.style.cursor = "grabbing";
    canvas.setPointerCapture?.(event.pointerId);
  }

  function handlePointerMove(event) {
    if (!state.dragging || !state.backgroundImage) {
      return;
    }

    const backgroundWidth = Math.max(1, state.backgroundImage.naturalWidth || state.backgroundImage.width || canvas.width);
    const backgroundHeight = Math.max(1, state.backgroundImage.naturalHeight || state.backgroundImage.height || canvas.height);
    const deltaX = Math.round((event.clientX - state.dragStartClientX) * (backgroundWidth / canvas.width));
    const deltaY = Math.round((event.clientY - state.dragStartClientY) * (backgroundHeight / canvas.height));
    setWidgetValue(xWidget, state.dragStartOffsetX + deltaX);
    setWidgetValue(yWidget, state.dragStartOffsetY + deltaY);
    renderPreview();
    node.setDirtyCanvas?.(true, true);
  }

  function endDrag(event) {
    if (!state.dragging) {
      return;
    }
    state.dragging = false;
    canvas.style.cursor = "grab";
    if (event?.pointerId != null) {
      canvas.releasePointerCapture?.(event.pointerId);
    }
  }

  canvas.addEventListener("pointerdown", handlePointerDown);
  canvas.addEventListener("pointermove", handlePointerMove);
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("pointerleave", endDrag);

  [
    xWidget,
    yWidget,
    anchorWidget,
    rotationOriginWidget,
    opacityWidget,
    scaleWidget,
    rotationWidget,
  ].forEach((widget) => chainWidgetCallback(widget, renderPreview));

  const apiHandle = {
    handleGraphConfigured() {
      void reloadSources();
    },
    handleConnectionsChange() {
      void reloadSources();
    },
  };

  node.__llsImageCompositePreview = apiHandle;
  updateStatus("Connect Load Image / Load Image Output to both inputs for preview.");
  renderPreview();
  return apiHandle;
}

app.registerExtension({
  name: EXTENSION_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TARGET_NODE_CLASS && nodeData.display_name !== TARGET_NODE_DISPLAY_NAME) {
      return;
    }

    const previousOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function onImageCompositeNodeCreated() {
      const result = previousOnNodeCreated?.apply(this, arguments);
      attachCompositePreview(this);
      return result;
    };

    const previousOnGraphConfigured = nodeType.prototype.onGraphConfigured;
    nodeType.prototype.onGraphConfigured = function onImageCompositeGraphConfigured() {
      const result = previousOnGraphConfigured?.apply(this, arguments);
      attachCompositePreview(this)?.handleGraphConfigured?.();
      return result;
    };

    const previousOnConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function onImageCompositeConnectionsChange() {
      const result = previousOnConnectionsChange?.apply(this, arguments);
      attachCompositePreview(this)?.handleConnectionsChange?.();
      return result;
    };
  },
});
