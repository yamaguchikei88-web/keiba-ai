import importlib
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import project_paths


class ProjectPathsTests(unittest.TestCase):
    def test_defaults_are_repo_relative(self):
        with patch.dict(os.environ, {}, clear=True):
            paths = importlib.reload(project_paths)
            self.assertEqual(paths.DB_PATH, paths.PROJECT_ROOT / "data" / "keiba.db")
            self.assertEqual(paths.model_path("model.pkl"), paths.PROJECT_ROOT / "models" / "model.pkl")

    def test_environment_overrides_artifact_locations(self):
        with patch.dict(os.environ, {
            "KEIBA_DATA_DIR": "shared-data",
            "KEIBA_MODEL_DIR": "shared-models",
            "KEIBA_DB_PATH": "shared-data/races.sqlite",
            "KEIBA_REGISTRY_DB_PATH": "shared-data/registry.sqlite",
        }, clear=True):
            paths = importlib.reload(project_paths)
            self.assertEqual(paths.DB_PATH, Path("shared-data/races.sqlite"))
            self.assertEqual(paths.REGISTRY_DB_PATH, Path("shared-data/registry.sqlite"))
            self.assertEqual(paths.model_path("v1.pkl"), Path("shared-models/v1.pkl"))
        importlib.reload(project_paths)


if __name__ == "__main__":
    unittest.main()
