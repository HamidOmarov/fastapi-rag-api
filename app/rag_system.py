# app/rag_system.py
from __future__ import annotations

import os, re
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"
CACHE_DIR = Path(os.getenv("HF_HOME", str(ROOT_DIR / ".cache")))
for d in (DATA_DIR, UPLOAD_DIR, INDEX_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Output dili – EN üçün "en" saxla (default en)
OUTPUT_LANG = os.getenv("OUTPUT_LANG", "en").lower()

# --- util funksiyalar ---
NUM_PAT = re.compile(r"(\d+([.,]\d+)?|%|m²|AZN|usd|eur|\bset\b|\bmt\b)", re.IGNORECASE)

def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+|[\r\n]+', text) if s.strip()]

def _mostly_numeric(s: str) -> bool:
    # daha aqressiv threshold
    alnum = [c for c in s if c.isalnum()]
    if not alnum:
        return True
    digits = sum(c.isdigit() for c in alnum)
    return digits / max(1, len(alnum)) > 0.3

def _tabular_like(s: str) -> bool:
    # rəqəmlər/ölçülər/valyuta bol olan sətirləri at
    hits = len(NUM_PAT.findall(s))
    return hits >= 2 or "Page" in s or len(s) < 20

def _clean_for_summary(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        t = " ".join(ln.split())
        if not t:
            continue
        if _mostly_numeric(t) or _tabular_like(t):
            continue
        lines.append(t)
    return " ".join(lines)

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
        self.embed_dim = self.model.get_sentence_embedding_dimension()

        # translator lazy-load
        self._translator = None

        self.index: faiss.Index = None  # type: ignore
        self.chunks: List[str] = []
        self._load()

    # ---- translator (az->en) ----
    def _translate_to_en(self, texts: List[str]) -> List[str]:
        if OUTPUT_LANG != "en" or not texts:
            return texts
        try:
            if self._translator is None:
                from transformers import pipeline
                # Helsinki-NLP az->en
                self._translator = pipeline(
                    "translation",
                    model="Helsinki-NLP/opus-mt-az-en",
                    cache_dir=str(self.cache_dir),
                    device=-1,
                )
            outs = self._translator(texts, max_length=400)
            return [o["translation_text"] for o in outs]
        except Exception:
            # tərcümə alınmasa, orijinalı qaytar
            return texts

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

    def synthesize_answer(self, question: str, contexts: List[str], max_sentences: int = 4) -> str:
        if not contexts:
            return "No relevant context found. Please upload a PDF or ask a more specific question."

        # 1) Candidate sentence-lər (aggressive clean)
         candidates: List[str] = []
        for c in contexts[:5]:
             cleaned = _clean_for_summary(c)
            for s in _split_sentences(cleaned):
                # uzunluq və keyfiyyət filtrləri
                w = s.split()
                if not (8 <= len(w) <= 35):
                   continue
                if _tabular_like(s) or _mostly_numeric(s):
                   continue
                candidates.append(" ".join(w))  # normalizasiya: bir boşluq

        if not candidates:
            return "The document appears largely tabular/numeric; couldn't extract readable sentences."

        # 2) Oxşarlığa görə sıralama
        q_emb = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        cand_emb = self.model.encode(candidates, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores = (cand_emb @ q_emb.T).ravel()
        order = np.argsort(-scores)

        # 3) Near-duplicate dedup (Jaccard söz seti) – threshold 0.82
        selected: List[str] = []
        for i in order:
        s = candidates[i].strip()
            if any(_sim_jaccard(s, t) >= 0.82 for t in selected):
                continue
            selected.append(s)
            if len(selected) >= max_sentences:
                break

        if not selected:
            return "The document appears largely tabular/numeric; couldn't extract readable sentences."

        # 4) HƏMİŞƏ EN tərcümə (istəyin belədir)
        if os.getenv("OUTPUT_LANG", "en").lower() == "en":
            selected = self._translate_to_en(selected)

        bullets = "\n".join(f"- {s}" for s in selected)
        return f"Answer (based on document context):\n{bullets}"

def _sim_jaccard(a: str, b: str) -> float:
    aw = set(a.lower().split())
    bw = set(b.lower().split())
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)

def synthesize_answer(question: str, contexts: List[str]) -> str:
    return SimpleRAG().synthesize_answer(question, contexts)


__all__ = ["SimpleRAG", "synthesize_answer", "DATA_DIR", "UPLOAD_DIR", "INDEX_DIR", "CACHE_DIR", "MODEL_NAME"]
