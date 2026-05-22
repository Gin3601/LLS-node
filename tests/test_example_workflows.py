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


if __name__ == "__main__":
    unittest.main()
