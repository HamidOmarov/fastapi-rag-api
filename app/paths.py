import os
from pathlib import Path
DATA_DIR = Path(os.getenv("DATA_DIR", str(DATA_DIR)))
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR = DATA_DIR / "index"; INDEX_DIR.mkdir(exist_ok=True)
HISTORY_JSON = DATA_DIR / "history.json"

