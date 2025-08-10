# app/rag_system.py
from __future__ import annotations

import os, re
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Paths & caches
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"
CACHE_DIR = Path(os.getenv("HF_HOME", str(ROOT_DIR / ".cache")))
for d in (DATA_DIR, UPLOAD_DIR, INDEX_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def _split_sentences(text: str) -> List[str]:
    # Split by sentence end or newlines
    return [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+|[\r\n]+', text) if s.strip()]

def _mostly_numeric(s: str) -> bool:
    alnum = [c for c in s if c.isalnum()]
    if not alnum:
        return True
    digits = sum(c.isdigit() for c in alnum)
    return digits / len(alnum) > 0.5

def _clean_for_summary(text: str) -> str:
    # Drop lines that are mostly numbers / too short
    lines = []
    for ln in text.splitlines():
        t = " ".join(ln.split())
        if len(t) < 10:
            continue
        if _mostly_numeric(t):
            continue
        lines.append(t)
    return " ".join(lines)

class SimpleRAG:
    """
    - PDF -> text chunking
    - Sentence-Transformers embeddings (cosine/IP)
    - FAISS index
    - Extractive answer in EN
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
        self.embed_dim = self.model.get_sentence_embedding_dimension()

        self.index: faiss.Index = None  # type: ignore
        self.chunks: List[str] = []
        self._load()

    def _load(self) -> None:
        if self.meta_path.exists():
            try:
                self.chunks = np.load(self.meta_path, allow_pickle=True).tolist()
            except Exception:
                self.chunks = []
        if self.index_path.exists():
            try:
                idx = faiss.read_index(str(self.index_path))
                self.index = idx if getattr(idx, "d", None) == self.embed_dim else faiss.IndexFlatIP(self.embed_dim)
            except Exception:
                self.index = faiss.IndexFlatIP(self.embed_dim)
        else:
            self.index = faiss.IndexFlatIP(self.embed_dim)

    def _persist(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        np.save(self.meta_path, np.array(self.chunks, dtype=object))

    @staticmethod
    def _pdf_to_texts(pdf_path: Path, step: int = 800) -> List[str]:
        reader = PdfReader(str(pdf_path))
        pages = []
        for p in reader.pages:
            t = p.extract_text() or ""
            if t.strip():
                pages.append(t)
        chunks: List[str] = []
        for txt in pages:
            for i in range(0, len(txt), step):
                part = txt[i:i+step].strip()
                if part:
                    chunks.append(part)
        return chunks

    def add_pdf(self, pdf_path: Path) -> int:
        texts = self._pdf_to_texts(pdf_path)
        if not texts:
            return 0
        emb = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        self.index.add(emb.astype(np.float32))
        self.chunks.extend(texts)
        self._persist()
        return len(texts)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        D, I = self.index.search(q, min(k, max(1, self.index.ntotal)))
        out: List[Tuple[str, float]] = []
        if I.size > 0 and self.chunks:
            for idx, score in zip(I[0], D[0]):
                if 0 <= idx < len(self.chunks):
                    out.append((self.chunks[idx], float(score)))
        return out

    # -------- Improved English answer --------
    def synthesize_answer(self, question: str, contexts: List[str], max_sentences: int = 5) -> str:
        if not contexts:
            return "No relevant context found. Please upload a PDF or ask a more specific question."

        # Prepare candidate sentences
        candidates: List[str] = []
        for c in contexts[:5]:
            cleaned = _clean_for_summary(c)
            for s in _split_sentences(cleaned):
                if 20 <= len(s) <= 240 and not _mostly_numeric(s):
                    candidates.append(s)

        # Fallback if still nothing
        if not candidates:
            return "The document appears to be mostly tabular/numeric; no clear sentences to summarize."

        # Rank candidates by cosine similarity to the question
        q_emb = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        cand_emb = self.model.encode(candidates, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores = (cand_emb @ q_emb.T).ravel()
        order = np.argsort(-scores)

        # Pick top sentences with simple de-dup
        selected: List[str] = []
        seen = set()
        for i in order:
            s = candidates[i].strip()
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            selected.append(s)
            if len(selected) >= max_sentences:
                break

        bullet = "\n".join(f"- {s}" for s in selected)
        note = " (The PDF seems largely tabular; extracted the most relevant lines.)" if all(_mostly_numeric(c) for c in contexts) else ""
        return f"Answer (based on document context):\n{bullet}{note}"


# Module-level alias
def synthesize_answer(question: str, contexts: List[str]) -> str:
    return SimpleRAG().synthesize_answer(question, contexts)


__all__ = ["SimpleRAG", "synthesize_answer", "DATA_DIR", "UPLOAD_DIR", "INDEX_DIR", "CACHE_DIR", "MODEL_NAME"]
