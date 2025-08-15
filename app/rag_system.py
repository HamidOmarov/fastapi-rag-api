# app/rag_system.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import faiss
import numpy as np
from ftfy import fix_text as _ftfy_fix

# Prefer pypdf; fallback to PyPDF2 if needed
try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception:  # pragma: no cover
        PdfReader = None  # will try pdfminer if available

# sentence-transformers encoder
from sentence_transformers import SentenceTransformer


# ---------------- Paths & Cache (HF-safe) ----------------
ROOT_DIR = Path(os.getenv("APP_ROOT", "/app"))  # HF Spaces writeable base
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT_DIR / "data")))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(DATA_DIR / "uploads")))
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(DATA_DIR / "index")))
CACHE_DIR = Path(os.getenv("HF_HOME", str(ROOT_DIR / ".cache")))  # transformers uses HF_HOME

for d in (DATA_DIR, UPLOAD_DIR, INDEX_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------- Config ----------------
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OUTPUT_LANG = os.getenv("OUTPUT_LANG", "en").strip().lower()  # "en" в†’ translate AZв†’EN


# ---------------- Text helpers ----------------
# Join AZ letters split by spaces (e.g., "H ЖЏ F T ЖЏ" в†’ "HЖЏFTЖЏ")
AZ_LATIN = "A-Za-zЖЏЙ™ДћДџД°Д±Г–Г¶ЕћЕџГ‡Г§ГњГј"
_SINGLE_LETTER_RUN = re.compile(rf"\b(?:[{AZ_LATIN}]\s+){{2,}}[{AZ_LATIN}]\b")

def _fix_intra_word_spaces(s: str) -> str:
    if not s:
        return s
    return _SINGLE_LETTER_RUN.sub(lambda m: re.sub(r"\s+", "", m.group(0)), s)

def _fix_mojibake(s: str) -> str:
    """Fix common UTF-8-as-Latin-1 mojibake quickly; then ftfy."""
    if not s:
        return s
    if any(sym in s for sym in ("Гѓ", "Г„", "Г…", "Гђ", "Гћ", "Гѕ", "Гў")):
        try:
            s = s.encode("latin-1", "ignore").decode("utf-8", "ignore")
        except Exception:
            pass
    # ftfy final pass (safe on already-correct text)
    return _ftfy_fix(s)

def _clean_for_summary(text: str) -> str:
    """Remove ultra-short / numeric / tabular-ish lines, collapse spaces."""
    NUM_TOKEN_RE = re.compile(r"\b(\d+[.,]?\d*|%|mВІ|azn|usd|eur|mt|m2)\b", re.IGNORECASE)

    def _mostly_numeric(s: str) -> bool:
        alnum = [c for c in s if c.isalnum()]
        if not alnum:
            return True
        digits = sum(c.isdigit() for c in alnum)
        return digits / max(1, len(alnum)) > 0.30

    def _tabular_like(s: str) -> bool:
        hits = len(NUM_TOKEN_RE.findall(s))
        return hits >= 2 or "Page" in s or len(s) < 20

    out = []
    for ln in text.splitlines():
        t = " ".join(ln.split())
        if not t or _mostly_numeric(t) or _tabular_like(t):
            continue
        out.append(t)
    return " ".join(out)

def _split_sentences(text: str) -> List[str]:
    # simple splitter ok for extractive snippets
    return [s.strip() for s in re.split(r"(?<=[\.!\?])\s+|[\r\n]+", text) if s.strip()]

STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","for","with","by",
    "this","that","these","those","is","are","was","were","be","been","being",
    "at","as","it","its","from","into","about","over","after","before","than",
    "such","can","could","should","would","may","might","will","shall",
}

def _keywords(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-zГЂ-Г–Г-Г¶Гё-Гї0-9]+", text.lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]

def _sim_jaccard(a: str, b: str) -> float:
    aw = set(a.lower().split())
    bw = set(b.lower().split())
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


# ---------------- RAG Core ----------------
class SimpleRAG:
    """
    Minimal RAG core:
    - FAISS (IP) over sentence-transformers embeddings
    - PDF в†’ texts with robust decoding (pypdf/PyPDF2 + ftfy; optional pdfminer fallback)
    - Extractive answer synthesis with embedding ranking + keyword fallback
    """

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

    # ---------- Public utils ----------
    @property
    def is_empty(self) -> bool:
        return getattr(self.index, "ntotal", 0) == 0 or not self.chunks

    @property
    def faiss_ntotal(self) -> int:
        return int(getattr(self.index, "ntotal", 0))

    @property
    def model_dim(self) -> int:
        return int(self.embed_dim)

    def reset_index(self) -> None:
        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.chunks = []
        self.last_added = []
        try:
            if self.index_path.exists():
                self.index_path.unlink()
        except Exception:
            pass
        try:
            if self.meta_path.exists():
                self.meta_path.unlink()
        except Exception:
            pass

    # ---------- PDF в†’ texts ----------
    @staticmethod
    def _pdf_to_texts(pdf_path: Path, step: int = 800) -> List[str]:
        texts: List[str] = []

        # A) pypdf / PyPDF2
        if PdfReader is not None:
            try:
                reader = PdfReader(str(pdf_path))
                for p in getattr(reader, "pages", []):
                    t = p.extract_text() or ""
                    t = _fix_mojibake(t)
                    t = _fix_intra_word_spaces(t)
                    if t.strip():
                        texts.append(t)
            except Exception:
                pass

        # B) Optional pdfminer fallback if nothing extracted
        if not texts:
            try:
                from pdfminer.high_level import extract_text  # type: ignore
                raw = extract_text(str(pdf_path)) or ""
                raw = _fix_mojibake(raw)
                raw = _fix_intra_word_spaces(raw)
                if raw.strip():
                    texts = [raw]
            except Exception:
                pass

        # Split to fixed-size chunks (simple & fast)
        chunks: List[str] = []
        for txt in texts:
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
        # final cleaning for safety
        texts = [_fix_mojibake(_fix_intra_word_spaces(t)) for t in texts]

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
        k = max(1, min(int(k or 5), self.faiss_ntotal or 1))
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
            from transformers import pipeline  # lazy import
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
            return texts  # graceful fallback

    # ---------- Fallbacks ----------
    def _keyword_fallback(self, question: str, pool: List[str], limit_sentences: int = 4) -> List[str]:
        qk = set(_keywords(question))
        if not qk:
            return []
        candidates: List[Tuple[float, str]] = []
        for text in pool[:200]:
            cleaned = _clean_for_summary(text)
            for s in _split_sentences(cleaned):
                w = s.split()
                if not (8 <= len(w) <= 40):
                    continue
                toks = set(_keywords(s))
                if not toks:
                    continue
                overlap = len(qk & toks)
                if overlap == 0:
                    continue
                length_penalty = max(8, min(40, len(w)))
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
            return "No relevant context found. Index is empty вЂ” upload a PDF first."

        # Strong decoding & spacing fixes on contexts
        contexts = [_fix_mojibake(_fix_intra_word_spaces(c)) for c in (contexts or [])]

        # Build candidate sentences from top contexts
        local_pool: List[str] = []
        for c in (contexts or [])[:5]:
            cleaned = _clean_for_summary(c)
            for s in _split_sentences(cleaned):
                w = s.split()
                if not (8 <= len(w) <= 40):
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

        # Fallback via keywords over entire corpus
        if not selected:
            selected = self._keyword_fallback(question, self.chunks, limit_sentences=max_sentences)

        if not selected:
            return "No readable sentences matched the question. Try a more specific query."

        # Optional AZв†’EN translate if output language is English and text is non-ASCII
        if OUTPUT_LANG == "en" and any(ord(ch) > 127 for ch in " ".join(selected)):
            try:
                selected = self._translate_to_en(selected)
            except Exception:
                pass

        bullets = "\n".join(f"- {s}" for s in selected)
        return f"Answer (based on document context):\n{bullets}"


# Public API
__all__ = [
    "SimpleRAG",
    "UPLOAD_DIR",
    "INDEX_DIR",
]
