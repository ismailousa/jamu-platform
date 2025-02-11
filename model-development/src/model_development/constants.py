from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

print(PROJECT_ROOT)

MODULES_DIR = PROJECT_ROOT / "src" / "model_development"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

CONFIG_FILE = MODULES_DIR / "config.yml"

