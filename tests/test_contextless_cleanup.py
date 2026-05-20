import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestContextlessCleanup(unittest.TestCase):
    def test_legacy_task_context_files_are_removed(self):
        self.assertFalse((ROOT / "utils" / "task_context.py").exists())
        self.assertFalse((ROOT / "task" / "__init__.py").exists())
        self.assertFalse((ROOT / "task" / "nodes.py").exists())

    def test_stale_planning_docs_are_removed(self):
        self.assertFalse((ROOT / "docs" / "superpowers" / "plans" / "2026-05-20-auto-task-router.md").exists())
        self.assertFalse((ROOT / "docs" / "superpowers" / "specs" / "2026-05-20-auto-task-router-design.md").exists())
        self.assertFalse(
            (ROOT / "docs" / "superpowers" / "plans" / "2026-05-20-contextless-routing-final-state.md").exists()
        )

    def test_gitignore_blocks_python_cache_artifacts(self):
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("__pycache__/", gitignore)
        self.assertIn("*.py[cod]", gitignore)


if __name__ == "__main__":
    unittest.main()
