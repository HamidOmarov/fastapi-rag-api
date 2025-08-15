# storage.py (no BOM)
import os, tempfile
from pathlib import Path

def _first_writable(candidates):
    for c in candidates:
        if not c:
            continue
        try:
            p = Path(c)
            p.mkdir(parents=True, exist_ok=True)
            t = p / ".write_test"
            t.write_text("ok", encoding="utf-8")
            try:
                t.unlink()
            except OSError:
                pass
            return p
        except Exception:
            continue
    return Path(tempfile.mkdtemp(prefix="rag_"))

DATA_DIR = _first_writable([
    os.getenv("DATA_DIR") or None,
    "/data",
    "/app/data",
    str(Path.home() / ".cache" / "rag_data"),
    "/tmp/rag_data",
])

INDEX_DIR    = DATA_DIR / "index"
HISTORY_JSON = DATA_DIR / "history.json"