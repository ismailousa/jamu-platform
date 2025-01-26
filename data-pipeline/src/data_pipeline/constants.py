from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODULES_DIR = PROJECT_ROOT / "src/data_pipeline"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

CONFIG_FILE_PATH = MODULES_DIR / "config.yml"