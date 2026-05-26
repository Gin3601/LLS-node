import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestExampleWorkflows(unittest.TestCase):
    def _load(self, name: str) -> dict:
        return json.loads((ROOT / name).read_text(encoding="utf-8"))

    def _find_node(self, workflow: dict, node_type: str) -> dict:
        for node in workflow.get("nodes", []):
            if node.get("type") == node_type:
                return node
        self.fail(f"Missing node type {node_type!r} in workflow")

    def _find_input(self, node: dict, name: str) -> dict:
        for item in node.get("inputs", []):
            if item.get("name") == name:
                return item
        self.fail(f"Missing input {name!r} in node {node.get('type')!r}")

    def test_universal_txt2img_ksampler_keeps_seed_control_widget(self):
        workflow = self._load("LLS-universal-txt2img-workflow.json")
        node = self._find_node(workflow, "LLSSimpleKSampler")
        widgets = node.get("widgets_values", [])

        self.assertGreaterEqual(len(widgets), 12)
        self.assertEqual(widgets[0], "Family Default")
        self.assertEqual(widgets[2], "randomize")
        self.assertEqual(widgets[3], 20)

    def test_universal_inpaint_ksampler_keeps_seed_control_widget(self):
        workflow = self._load("LLS-universal-inpaint-workflow.json")
        node = self._find_node(workflow, "LLSSimpleKSampler")
        widgets = node.get("widgets_values", [])

        self.assertGreaterEqual(len(widgets), 12)
        self.assertEqual(widgets[0], "Family Default")
        self.assertEqual(widgets[2], "randomize")
        self.assertEqual(widgets[3], 20)

    def test_flux2klein_background_replace_workflow_uses_lls_edit_and_flux2_stack(self):
        workflow = self._load("LLS-Flux2Klein-电商换背景工作流.json")

        encode = self._find_node(workflow, "LLSFlux2KleinEditTextEncode")
        unet = self._find_node(workflow, "UNETLoader")
        clip = self._find_node(workflow, "CLIPLoader")
        vae = self._find_node(workflow, "VAELoader")
        self._find_node(workflow, "RMBG")
        self._find_node(workflow, "InvertMask")
        self._find_node(workflow, "ConditioningZeroOut")
        self._find_node(workflow, "KSampler")
        self._find_node(workflow, "VAEDecode")

        self.assertEqual(unet.get("widgets_values", [None])[0], "flux-2-klein-9b.safetensors")
        self.assertEqual(clip.get("widgets_values", [None, None])[1], "flux2")
        self.assertEqual(vae.get("widgets_values", [None])[0], "flux2-vae.safetensors")

        widgets = encode.get("widgets_values", [])
        self.assertGreaterEqual(len(widgets), 4)
        self.assertEqual(widgets[2], "longest_edge")
        self.assertEqual(widgets[3], "use_mask")

        self.assertIsNotNone(self._find_input(encode, "image1").get("link"))
        self.assertIsNotNone(self._find_input(encode, "image2").get("link"))
        self.assertIsNone(self._find_input(encode, "image3").get("link"))
        self.assertIsNotNone(self._find_input(encode, "mask").get("link"))


if __name__ == "__main__":
    unittest.main()
