import json
import unittest

try:
    from .test_mask_draw_helpers import (
        HAS_MASK_DRAW_RUNTIME_DEPS,
        MASK_DRAW_RUNTIME_DEPS_MESSAGE,
        load_mask_draw_utils,
        make_image,
        make_mask,
        make_mask_state_json,
        torch,
    )
except ImportError:
    from test_mask_draw_helpers import (
        HAS_MASK_DRAW_RUNTIME_DEPS,
        MASK_DRAW_RUNTIME_DEPS_MESSAGE,
        load_mask_draw_utils,
        make_image,
        make_mask,
        make_mask_state_json,
        torch,
    )


class TestMaskDrawUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.utils = load_mask_draw_utils()

    def test_parse_mask_state_defaults_invalid_json(self):
        state = self.utils.parse_mask_state("{invalid")

        self.assertFalse(state["touched"])
        self.assertEqual(state["mask_png_base64"], "")
        self.assertEqual(state["version"], 1)

    def test_parse_mask_state_defaults_invalid_editor_and_dirty_touched_values(self):
        for payload in (
            {"editor": [1], "touched": "false"},
            {"editor": "bad", "touched": "true"},
        ):
            with self.subTest(payload=payload):
                state = self.utils.parse_mask_state(json.dumps(payload))

                self.assertEqual(state["editor"], {})
                self.assertFalse(state["touched"])

    @unittest.skipUnless(HAS_MASK_DRAW_RUNTIME_DEPS, MASK_DRAW_RUNTIME_DEPS_MESSAGE)
    def test_resolve_output_mask_prefers_saved_mask_when_touched(self):
        image = make_image(width=6, height=4)
        input_mask = make_mask(width=6, height=4, value=0.0)
        state_json = make_mask_state_json(width=6, height=4, value=1.0, touched=True)

        mask = self.utils.resolve_output_mask(
            image=image,
            input_mask=input_mask,
            mask_state_json=state_json,
            invert_mask=False,
        )

        self.assertEqual(tuple(mask.shape), (1, 4, 6))
        self.assertTrue(torch.all(mask == 1.0))

    @unittest.skipUnless(HAS_MASK_DRAW_RUNTIME_DEPS, MASK_DRAW_RUNTIME_DEPS_MESSAGE)
    def test_resolve_output_mask_repeats_saved_mask_to_image_batch(self):
        image = torch.full((3, 4, 6, 3), 0.25, dtype=torch.float32)
        state_json = make_mask_state_json(width=6, height=4, value=1.0, touched=True)

        mask = self.utils.resolve_output_mask(
            image=image,
            input_mask=None,
            mask_state_json=state_json,
            invert_mask=False,
        )

        self.assertEqual(tuple(mask.shape), (3, 4, 6))
        self.assertTrue(torch.all(mask == 1.0))

    @unittest.skipUnless(HAS_MASK_DRAW_RUNTIME_DEPS, MASK_DRAW_RUNTIME_DEPS_MESSAGE)
    def test_resolve_output_mask_uses_input_mask_when_untouched(self):
        image = make_image(width=6, height=4)
        input_mask = make_mask(width=3, height=2, value=1.0)

        mask = self.utils.resolve_output_mask(
            image=image,
            input_mask=input_mask,
            mask_state_json="{}",
            invert_mask=False,
        )

        self.assertEqual(tuple(mask.shape), (1, 4, 6))
        self.assertTrue(torch.all(mask == 1.0))

    @unittest.skipUnless(HAS_MASK_DRAW_RUNTIME_DEPS, MASK_DRAW_RUNTIME_DEPS_MESSAGE)
    def test_resolve_output_mask_repeats_single_input_mask_to_image_batch(self):
        image = torch.full((3, 4, 6, 3), 0.25, dtype=torch.float32)
        input_mask = make_mask(width=6, height=4, value=1.0)

        mask = self.utils.resolve_output_mask(
            image=image,
            input_mask=input_mask,
            mask_state_json="{}",
            invert_mask=False,
        )

        self.assertEqual(tuple(mask.shape), (3, 4, 6))
        self.assertTrue(torch.all(mask == 1.0))

    @unittest.skipUnless(HAS_MASK_DRAW_RUNTIME_DEPS, MASK_DRAW_RUNTIME_DEPS_MESSAGE)
    def test_resolve_output_mask_rejects_input_mask_batch_mismatch(self):
        image = torch.full((3, 4, 6, 3), 0.25, dtype=torch.float32)
        input_mask = torch.full((2, 4, 6), 1.0, dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "mask batch size must match image batch size"):
            self.utils.resolve_output_mask(
                image=image,
                input_mask=input_mask,
                mask_state_json="{}",
                invert_mask=False,
            )

    @unittest.skipUnless(HAS_MASK_DRAW_RUNTIME_DEPS, MASK_DRAW_RUNTIME_DEPS_MESSAGE)
    def test_resolve_output_mask_returns_black_after_clear_state(self):
        image = make_image(width=6, height=4)
        input_mask = make_mask(width=6, height=4, value=1.0)
        state_json = make_mask_state_json(width=6, height=4, value=0.0, touched=True)

        mask = self.utils.resolve_output_mask(
            image=image,
            input_mask=input_mask,
            mask_state_json=state_json,
            invert_mask=False,
        )

        self.assertTrue(torch.all(mask == 0.0))

    @unittest.skipUnless(HAS_MASK_DRAW_RUNTIME_DEPS, MASK_DRAW_RUNTIME_DEPS_MESSAGE)
    def test_resolve_output_mask_falls_back_to_input_mask_when_saved_mask_is_corrupt(self):
        image = make_image(width=6, height=4)
        input_mask = make_mask(width=6, height=4, value=1.0)
        state_json = '{"version": 1, "mask_png_base64": "not-base64", "touched": true, "editor": {}}'

        mask = self.utils.resolve_output_mask(
            image=image,
            input_mask=input_mask,
            mask_state_json=state_json,
            invert_mask=False,
        )

        self.assertEqual(tuple(mask.shape), (1, 4, 6))
        self.assertTrue(torch.all(mask == 1.0))

    @unittest.skipUnless(HAS_MASK_DRAW_RUNTIME_DEPS, MASK_DRAW_RUNTIME_DEPS_MESSAGE)
    def test_resolve_output_mask_applies_invert(self):
        image = make_image(width=6, height=4)
        state_json = make_mask_state_json(width=6, height=4, value=1.0, touched=True)

        mask = self.utils.resolve_output_mask(
            image=image,
            input_mask=None,
            mask_state_json=state_json,
            invert_mask=True,
        )

        self.assertTrue(torch.all(mask == 0.0))

    @unittest.skipUnless(HAS_MASK_DRAW_RUNTIME_DEPS, MASK_DRAW_RUNTIME_DEPS_MESSAGE)
    def test_build_preview_image_preserves_shape_and_adds_red_overlay(self):
        image = make_image(width=6, height=4, color=0.2)
        mask = make_mask(width=6, height=4, value=1.0)

        preview = self.utils.build_preview_image(image=image, mask=mask, overlay_alpha=0.4)

        self.assertEqual(tuple(preview.shape), (1, 4, 6, 3))
        self.assertGreater(float(preview[..., 0].mean()), float(image[..., 0].mean()))
        self.assertLess(float(preview[..., 1].mean()), float(image[..., 1].mean()))
        self.assertLess(float(preview[..., 2].mean()), float(image[..., 2].mean()))


if __name__ == "__main__":
    unittest.main()
