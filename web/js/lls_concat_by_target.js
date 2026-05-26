import { app } from "../../scripts/app.js";

const EXTENSION_NAME = "LLS.ConcatByTarget";
const TARGET_NODE_CLASS = "LLSConcatByTarget";
const TARGET_NODE_DISPLAY_NAME = "LLS Concat By Target";
const INPUT_LABELS = {
  a: "image/mask_A",
  b: "image/mask_B",
};

function relabelConcatInputs(node) {
  let changed = false;

  for (const input of node.inputs ?? []) {
    const targetLabel = INPUT_LABELS[input.name];
    if (targetLabel && input.label !== targetLabel) {
      input.label = targetLabel;
      changed = true;
    }
  }

  if (changed) {
    node.setDirtyCanvas?.(true, true);
  }
}

app.registerExtension({
  name: EXTENSION_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TARGET_NODE_CLASS && nodeData.display_name !== TARGET_NODE_DISPLAY_NAME) {
      return;
    }

    const previousOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function onConcatByTargetNodeCreated() {
      const result = previousOnNodeCreated?.apply(this, arguments);
      relabelConcatInputs(this);
      return result;
    };

    const previousOnGraphConfigured = nodeType.prototype.onGraphConfigured;
    nodeType.prototype.onGraphConfigured = function onConcatByTargetGraphConfigured() {
      const result = previousOnGraphConfigured?.apply(this, arguments);
      relabelConcatInputs(this);
      return result;
    };

    const previousOnConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function onConcatByTargetConnectionsChange() {
      const result = previousOnConnectionsChange?.apply(this, arguments);
      relabelConcatInputs(this);
      return result;
    };
  },
});
