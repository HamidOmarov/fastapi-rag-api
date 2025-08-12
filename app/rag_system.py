# app/rag_system.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import faiss
import numpy as np

# Prefer pypdf; fallback to PyPDF2 if needed
try:
    from pypdf import PdfReader
except Exception:
    from PyPDF2 import PdfReader  # type: ignore

from sentence_transformers import SentenceTransformer

# ---------------- Paths & Cache (HF-safe) ----------------
# Writeable base is /app in HF Spaces. Allow ENV overrides.
ROOT_DIR = Path(os.getenv("APP_ROOT", "/app"))
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT_DIR / "data")))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(DATA_DIR / "uploads")))
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(DATA_DIR / "index")))
CACHE_DIR = Path(os.getenv("HF_HOME", str(ROOT_DIR / ".cache")))  # transformers prefers HF_HOME

for d in (DATA_DIR, UPLOAD_DIR, INDEX_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------- Config ----------------
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OUTPUT_LANG = os.getenv("OUTPUT_LANG", "en").lower()

# ---------------- Helpers ----------------
AZ_CHARS = set("əğıöşçüİıĞÖŞÇÜƏ")

def _fix_mojibake(s: str) -> str:
    """Fix common UTF-8-as-Latin-1 mojibake."""
    if not s:
        return s
    if any(ch in s for ch in ("Ã", "Ä", "Å", "Ð", "Þ", "þ")):
        try:
            return s.encode("latin-1", "ignore").decode("utf-8", "ignore")
        except Exception:
            return s
    return s

def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[\.!\?])\s+|[\r\n]+", text) if s.strip()]

def _mostly_numeric(s: str) -> bool:
    alnum = [c for c in s if c.isalnum()]
    if not alnum:
        return True
    digits = sum(c.isdigit() for c in alnum)
    return digits / max(1, len(alnum)) > 0.3

NUM_TOKEN_RE = re.compile(r"\b(\d+[.,]?\d*|%|m²|azn|usd|eur|set|mt)\b", re.IGNORECASE)

def _tabular_like(s: str) -> bool:
    hits = len(NUM_TOKEN_RE.findall(s))
    return hits >= 2 or "Page" in s or len(s) < 20

def _clean_for_summary(text: str) -> str:
    out = []
    for ln in text.splitlines():
        t = " ".join(ln.split())
        if not t or _mostly_numeric(t) or _tabular_like(t):
            continue
        out.append(t)
    return " ".join(out)

def _sim_jaccard(a: str, b: str) -> float:
    aw = set(a.lower().split())
    bw = set(b.lower().split())
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)

STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","for","with","by",
    "this","that","these","those","is","are","was","were","be","been","being",
    "at","as","it","its","from","into","about","over","after","before","than",
    "such","can","could","should","would","may","might","will","shall"
}

def _keywords(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", text.lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]

def _looks_azerbaijani(s: str) -> bool:
    has_az = any(ch in AZ_CHARS for ch in s)
    non_ascii_ratio = sum(ord(c) > 127 for c in s) / max(1, len(s))
    return has_az or non_ascii_ratio > 0.15

# ---------------- RAG Core ----------------
class SimpleRAG:
    def __init__(
        self,
        index_path: Path = INDEX_DIR / "faiss.index",
        meta_path: Path = INDEX_DIR / "meta.npy",
        model_name: str = MODEL_NAME,
        cache_dir: Path = CACHE_DIR,
    ):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)

        self.model = SentenceTransformer(self.model_name, cache_folder=str(self.cache_dir))
        self.embed_dim = int(self.model.get_sentence_embedding_dimension())

        self.index: faiss.Index = faiss.IndexFlatIP(self.embed_dim)
        self.chunks: List[str] = []
        self.last_added: List[str] = []
        self._translator = None  # lazy init

        self._load()

    # ---------- Persistence ----------
    def _load(self) -> None:
        if self.meta_path.exists():
            try:
                self.chunks = np.load(self.meta_path, allow_pickle=True).tolist()
            except Exception:
                self.chunks = []
        if self.index_path.exists():
            try:
                idx = faiss.read_index(str(self.index_path))
                if getattr(idx, "d", None) == self.embed_dim:
                    self.index = idx
            except Exception:
                pass

    def _persist(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        np.save(self.meta_path, np.array(self.chunks, dtype=object))

    # ---------- Utilities ----------
    @property
    def is_empty(self) -> bool:
        return getattr(self.index, "ntotal", 0) == 0 or not self.chunks

    @staticmethod
    def _pdf_to_texts(pdf_path: Path, step: int = 800) -> List[str]:
        reader = PdfReader(str(pdf_path))
        pages: List[str] = []
        for p in reader.pages:
            t = p.extract_text() or ""
            t = _fix_mojibake(t)
            if t.strip():
                pages.append(t)
        chunks: List[str] = []
        for txt in pages:
            for i in range(0, len(txt), step):
                part = txt[i : i + step].strip()
                if part:
                    chunks.append(part)
        return chunks

    # ---------- Indexing ----------
    def add_pdf(self, pdf_path: Path) -> int:
        texts = self._pdf_to_texts(pdf_path)
        if not texts:
            return 0
        emb = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        self.index.add(emb.astype(np.float32))
        self.chunks.extend(texts)
        self.last_added = texts[:]
        self._persist()
        return len(texts)

    # ---------- Search ----------
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if self.is_empty:
            return []
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        k = max(1, min(int(k or 5), getattr(self.index, "ntotal", 1)))
        D, I = self.index.search(q, k)
        out: List[Tuple[str, float]] = []
        if I.size > 0 and self.chunks:
            for idx, score in zip(I[0], D[0]):
                if 0 <= idx < len(self.chunks):
                    out.append((self.chunks[idx], float(score)))
        return out

    # ---------- Translation (optional) ----------
    def _translate_to_en(self, texts: List[str]) -> List[str]:
        if not texts:
            return texts
        try:
            from transformers import pipeline
            if self._translator is None:
                self._translator = pipeline(
                    "translation",
                    model="Helsinki-NLP/opus-mt-az-en",
                    cache_dir=str(self.cache_dir),
                    device=-1,
                )
            outs = self._translator(texts, max_length=400)
            return [o["translation_text"].strip() for o in outs]
        except Exception:
            return texts

    # ---------- Fallbacks ----------
    def _keyword_fallback(self, question: str, pool: List[str], limit_sentences: int = 4) -> List[str]:
        qk = set(_keywords(question))
        if not qk:
            return []
        candidates: List[Tuple[float, str]] = []
        for text in pool[:200]:
            cleaned = _clean_for_summary(text)
            for s in _split_sentences(cleaned):
                if _tabular_like(s) or _mostly_numeric(s):
                    continue
                toks = set(_keywords(s))
                if not toks:
                    continue
                overlap = len(qk & toks)
                if overlap == 0:
                    continue
                length_penalty = max(8, min(40, len(s.split())))
                score = overlap + min(0.5, overlap / length_penalty)
                candidates.append((score, s))
        candidates.sort(key=lambda x: x[0], reverse=True)
        out: List[str] = []
        for _, s in candidates:
            if any(_sim_jaccard(s, t) >= 0.82 for t in out):
                continue
            out.append(s)
            if len(out) >= limit_sentences:
                break
        return out

    # ---------- Answer Synthesis ----------
    def synthesize_answer(self, question: str, contexts: List[str], max_sentences: int = 4) -> str:
        if not contexts and self.is_empty:
            return "No relevant context found. Index is empty — upload a PDF first."

        # Fix mojibake in contexts
        contexts = [_fix_mojibake(c) for c in (contexts or [])]

        # Build candidate sentences from nearby contexts
        local_pool: List[str] = []
        for c in (contexts or [])[:5]:
            cleaned = _clean_for_summary(c)
            for s in _split_sentences(cleaned):
                w = s.split()
                if not (8 <= len(w) <= 35):
                    continue
                if _tabular_like(s) or _mostly_numeric(s):
                    continue
                local_pool.append(" ".join(w))

        selected: List[str] = []
        if local_pool:
            q_emb = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
            cand_emb = self.model.encode(local_pool, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
            scores = (cand_emb @ q_emb.T).ravel()
            order = np.argsort(-scores)
            for i in order:
                s = local_pool[i].strip()
                if any(_sim_jaccard(s, t) >= 0.82 for t in selected):
                    continue
                selected.append(s)
                if len(selected) >= max_sentences:
                    break

        if not selected:
            selected = self._keyword_fallback(question, self.chunks, limit_sentences=max_sentences)

        if not selected:
            return "No readable sentences matched the question. Try a more specific query."

        if OUTPUT_LANG == "en" and any(ord(ch) > 127 for ch in " ".join(selected)):
            selected = self._translate_to_en(selected)

        bullets = "\n".join(f"- {s}" for s in selected)
        return f"Answer (based on document context):\n{bullets}"


# Public API
__all__ = [
    "SimpleRAG",
    "UPLOAD_DIR",
    "INDEX_DIR",
]
