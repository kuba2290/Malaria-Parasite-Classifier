from pathlib import Path
from utils.common import read_yaml

CONFIG_FILE_PATH = Path("path/to/your/config.yaml")
try:
    config = read_yaml(CONFIG_FILE_PATH)
    artifacts_root = config.artifacts_root
    print(f"Artifacts root: {artifacts_root}")
except ValueError as e:
    print(f"Error reading configuration: {e}")
except AttributeError as e:
    print(f"Configuration error: '{e}'. Check if 'artifacts_root' is correctly spelled in your config.yaml")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
PARAMS_FILE_PATH = Path("params.yaml")