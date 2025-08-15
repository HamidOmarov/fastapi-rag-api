# app/store.py
from collections import defaultdict
from typing import List, Dict

# in-memory chat tarixi (prod ГјГ§Гјn Redis/Postgres mЙ™slЙ™hЙ™tdir)
_history: Dict[str, List[dict]] = defaultdict(list)

def add_history(session_id: str, role: str, content: str):
    _history[session_id].append({"role": role, "content": content})

def get_history(session_id: str) -> List[dict]:
    return _history.get(session_id, [])

