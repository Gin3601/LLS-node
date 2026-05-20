import importlib
import importlib.util
import pathlib
import sys
import unittest

try:
    from .test_repair_helpers import FakeMask
except ImportError:  # pragma: no cover - discovery mode imports from top level
    from test_repair_helpers import FakeMask


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_NAME = "lls_node_test_repair_utils"


def load_repair_utils():
    for name in list(sys.modules):
        if name == MODULE_NAME or name.startswith(f"{MODULE_NAME}."):
            sys.modules.pop(name)

    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return importlib.import_module(f"{MODULE_NAME}.repair.repair_utils")


class TestRepairUtils(unittest.TestCase):
    def test_normalize_model_info_canonicalizes_legacy_family_aliases(self):
        utils = load_repair_utils()

        model_info = utils.normalize_model_info({"model_family": "SD15"})
        self.assertEqual(model_info["model_family"], "SD1.5")

    def test_normalize_model_info_coerces_string_booleans(self):
        utils = load_repair_utils()

        self.assertFalse(
            utils.normalize_model_info({"supports_inpaint_native": "false"})["supports_inpaint_native"]
        )
        self.assertFalse(
            utils.normalize_model_info({"supports_inpaint_native": "0"})["supports_inpaint_native"]
        )
        self.assertTrue(
            utils.normalize_model_info({"supports_inpaint_native": "true"})["supports_inpaint_native"]
        )
        self.assertTrue(
            utils.normalize_model_info({"supports_inpaint_native": 1})["supports_inpaint_native"]
        )

    def test_canonicalized_family_feeds_sd_classic_adapter_resolution(self):
        utils = load_repair_utils()

        family = utils.normalize_model_info({"model_family": "SD15"})["model_family"]
        self.assertEqual(utils.resolve_adapter_mode("auto", family), "sd_classic")

    def test_resolve_repair_scope_prefers_canvas_then_crop_then_region(self):
        utils = load_repair_utils()

        self.assertEqual(
            utils.resolve_repair_scope(
                "auto",
                mask_area_ratio=0.0,
                mask_bbox=None,
                image_size=(1024, 1024),
                canvas_expand=(128, 0, 0, 0),
            ),
            "canvas",
        )
        self.assertEqual(
            utils.resolve_repair_scope(
                "auto",
                mask_area_ratio=0.08,
                mask_bbox=(100, 100, 260, 260),
                image_size=(1024, 1024),
                canvas_expand=(0, 0, 0, 0),
            ),
            "crop",
        )
        self.assertEqual(
            utils.resolve_repair_scope(
                "auto",
                mask_area_ratio=0.62,
                mask_bbox=(0, 0, 1024, 1024),
                image_size=(1024, 1024),
                canvas_expand=(0, 0, 0, 0),
            ),
            "region",
        )

    def test_resolve_repair_kernel_prefers_native_then_latent_then_vae(self):
        utils = load_repair_utils()

        self.assertEqual(
            utils.resolve_repair_kernel(
                "auto",
                scope="region",
                task_hint="repair",
                mask_area_ratio=0.05,
                model_info={"model_role": "fill", "supports_inpaint_native": True},
            )[0],
            "native_fill",
        )
        self.assertEqual(
            utils.resolve_repair_kernel(
                "auto",
                scope="region",
                task_hint="appearance",
                mask_area_ratio=0.05,
                model_info={"model_role": "normal", "supports_inpaint_native": False},
            )[0],
            "latent_mask",
        )
        self.assertEqual(
            utils.resolve_repair_kernel(
                "auto",
                scope="canvas",
                task_hint="fill",
                mask_area_ratio=0.50,
                model_info={"model_role": "normal", "supports_inpaint_native": False},
            )[0],
            "vae_inpaint",
        )

    def test_resolve_repair_kernel_downgrades_explicit_native_fill_when_unsupported(self):
        utils = load_repair_utils()

        kernel, warnings = utils.resolve_repair_kernel(
            "native_fill",
            scope="region",
            task_hint="repair",
            mask_area_ratio=0.05,
            model_info={"supports_inpaint_native": "false"},
        )
        self.assertEqual(kernel, "vae_inpaint")
        self.assertEqual(
            warnings,
            ["native_fill requested but unsupported; falling back to vae_inpaint"],
        )

    def test_recommend_denoise_applies_required_rules(self):
        utils = load_repair_utils()

        self.assertEqual(utils.recommend_denoise("repair", "region", "latent_mask", "enabled"), 0.45)
        self.assertEqual(utils.recommend_denoise("fill", "canvas", "vae_inpaint", "enabled"), 0.90)
        self.assertEqual(utils.recommend_denoise("replace", "crop", "vae_inpaint", "enabled"), 0.65)
        self.assertEqual(utils.recommend_denoise("auto", "crop", "vae_inpaint", "disabled"), 0.50)

    def test_crop_and_canvas_geometry_helpers(self):
        utils = load_repair_utils()

        self.assertEqual(
            utils.clamp_box((-20, 10, 1200, 2048), (1024, 768)),
            (0, 10, 1024, 768),
        )

        crop_box = utils.compute_crop_box(
            mask_bbox=(900, 900, 1100, 1100),
            image_size=(1024, 1024),
            crop_context=64,
            crop_context_factor=1.5,
        )
        self.assertEqual(crop_box, (754, 754, 1024, 1024))

        keep_aspect = utils.resolve_work_size(
            crop_size=(288, 288),
            min_size=512,
            max_size=1024,
            resize_mode="fit",
        )
        self.assertEqual(keep_aspect, (512, 512, 512 / 288))

        square = utils.resolve_work_size(
            crop_size=(400, 200),
            min_size=256,
            max_size=320,
            resize_mode="pad",
        )
        self.assertEqual(square, (320, 320, 320 / 400))

        canvas_info = utils.build_canvas_info((640, 480), 64, 32, 16, 8)
        self.assertEqual(canvas_info["work_size"], (736, 504))
        self.assertEqual(canvas_info["original_box"], (64, 16, 704, 496))

    def test_normalize_repair_info_backfills_required_defaults(self):
        utils = load_repair_utils()

        info = utils.normalize_repair_info({"repair_scope": "region", "repair_kernel": "latent_mask"})
        self.assertEqual(info["repair_scope"], "region")
        self.assertEqual(info["repair_kernel"], "latent_mask")
        self.assertEqual(info["task_hint"], "auto")
        self.assertEqual(info["original_size"], (0, 0))
        self.assertEqual(info["work_size"], (0, 0))
        self.assertEqual(info["canvas_expand"], (0, 0, 0, 0))
        self.assertEqual(info["mask_grow"], 8)
        self.assertEqual(info["mask_blur"], 8.0)
        self.assertEqual(info["mask_threshold"], 0.5)
        self.assertFalse(info["invert_mask"])
        self.assertEqual(info["recommended_denoise"], 0.55)
        self.assertEqual(info["model_family"], "UNKNOWN")
        self.assertEqual(info["model_role"], "unknown")
        self.assertEqual(info["repair_payload_version"], "1.0")
        self.assertEqual(info["warnings"], [])

    def test_normalize_repair_info_canonicalizes_family_fields(self):
        utils = load_repair_utils()

        info = utils.normalize_repair_info({"model_family": "SD15", "repair_scope": "region"})
        self.assertEqual(info["model_family"], "SD1.5")

    def test_normalize_repair_info_coerces_string_boolean_invert_mask(self):
        utils = load_repair_utils()

        self.assertFalse(utils.normalize_repair_info({"invert_mask": "false"})["invert_mask"])
        self.assertFalse(utils.normalize_repair_info({"invert_mask": "0"})["invert_mask"])
        self.assertTrue(utils.normalize_repair_info({"invert_mask": "true"})["invert_mask"])
        self.assertTrue(utils.normalize_repair_info({"invert_mask": 1})["invert_mask"])

    def test_preprocess_mask_applies_requested_operations(self):
        utils = load_repair_utils()

        mask = FakeMask(
            (1, 256, 256),
            mask_bbox=(100, 100, 140, 140),
            mask_area_ratio=((140 - 100) * (140 - 100)) / float(256 * 256),
            label="preprocess-mask",
        )
        processed = utils.preprocess_mask(
            mask,
            (256, 256),
            invert_mask=False,
            mask_threshold=0.8,
            mask_grow=12,
            mask_blur=6.0,
        )

        self.assertIn("normalize", processed.label)
        self.assertIn("threshold[0.8]", processed.label)
        self.assertIn("grow[12]", processed.label)
        self.assertIn("blur[6.0]", processed.label)
        self.assertNotEqual(processed.mask_bbox, mask.mask_bbox)
        self.assertGreater(processed.mask_area_ratio, mask.mask_area_ratio)

    def test_normalize_repair_info_normalizes_original_box_in_canvas(self):
        utils = load_repair_utils()

        info = utils.normalize_repair_info({"original_box_in_canvas": ["8", 16, 128, 144]})
        self.assertEqual(info["original_box_in_canvas"], (8, 16, 128, 144))


if __name__ == "__main__":
    unittest.main()
