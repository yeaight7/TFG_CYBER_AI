import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "graphify_auto_update.py"
SPEC = importlib.util.spec_from_file_location("graphify_auto_update", MODULE_PATH)
if SPEC is None:
    raise ImportError(f"Unable to create import spec for {MODULE_PATH}")

MODULE = importlib.util.module_from_spec(SPEC)
if SPEC.loader is None:
    raise ImportError(f"Unable to load module from {MODULE_PATH}: missing loader")
SPEC.loader.exec_module(MODULE)


class GraphifyAutoUpdateSemanticSourceTests(unittest.TestCase):
    def test_marks_markdown_in_docs_and_experiments_as_semantic_sources(self) -> None:
        self.assertTrue(MODULE._is_semantic_source("docs/README.md"))
        self.assertTrue(MODULE._is_semantic_source("experiments/cicids2017_qrdqn_experiments.md"))

    def test_excludes_personal_research_and_skills_markdown(self) -> None:
        self.assertFalse(MODULE._is_semantic_source("docs/Personal Research/deep-defense-research/README.md"))
        self.assertFalse(MODULE._is_semantic_source(".github/skills/graphify/SKILL.md"))


if __name__ == "__main__":
    unittest.main()
