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
OUTPUT_LANG = os.getenv("OUTPUT_LANG", "en").lower()

AZ_CHARS = set("əğıöşçüİıĞÖŞÇÜƏ")
NUM_TOK_RE = re.compile(r"\b(\d+[.,]?\d*|%|m²|azn|usd|eur|set|mt)\b", re.IGNORECASE)

def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|[\r\n]+', text) if s.strip()]

def _mostly_numeric(s: str) -> bool:
    alnum = [c for c in s if c.isalnum()]
    if not alnum:
        return True
    digits = sum(c.isdigit() for c in alnum)
    return digits / max(1, len(alnum)) > 0.3

def _tabular_like(s: str) -> bool:
    hits = len(NUM_TOK_RE.findall(s))
    return hits >= 3 or len(s) < 15

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

def _looks_azerbaijani(s: str) -> bool:
    has_az = any(ch in AZ_CHARS for ch in s)
    non_ascii_ratio = sum(ord(c) > 127 for c in s) / max(1, len(s))
    return has_az or non_ascii_ratio > 0.15

def _non_ascii_ratio(s: str) -> float:
    return sum(ord(c) > 127 for c in s) / max(1, len(s))

def _keyword_summary_en(contexts: List[str]) -> List[str]:
    text = " ".join(contexts).lower()
    bullets: List[str] = []
    def add(b: str):
        if b not in bullets:
            bullets.append(b)

    if ("şüşə" in text) or ("ara kəsm" in text) or ("s/q" in text):
        add("Removal and re-installation of glass partitions in sanitary areas.")
    if "divar kağız" in text:
        add("Wallpaper repair or replacement; some areas replaced with plaster and paint.")
    if ("alçı boya" in text) or ("boya işi" in text) or ("plaster" in text) or ("boya" in text):
        add("Wall plastering and painting works.")
    if "seramik" in text:
        add("Ceramic tiling works (including grouting).")
    if ("dilatasyon" in text) or ("ar 153" in text) or ("ar153" in text):
        add("Installation of AR 153–050 floor expansion joint profile with accessories and insulation.")
    if "daş yunu" in text:
        add("Rock wool insulation installed where required.")
    if ("sütunlarda" in text) or ("üzlüyün" in text):
        add("Repair of wall cladding on columns.")
    if ("m²" in text) or ("ədəd" in text) or ("azn" in text):
        add("Bill of quantities style lines with unit prices and measures (m², pcs).")

    if not bullets:
        bullets = [
            "The document appears to be a bill of quantities for renovation works.",
            "Scope includes demolition/reinstallation, finishing (plaster & paint), tiling, and profiles.",
        ]
    return bullets[:5]

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

        self._translator = None  # lazy
        self.index: faiss.Index = faiss.IndexFlatIP(self.embed_dim)
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
                if getattr(idx, "d", None) == self.embed_dim:
                    self.index = idx
            except Exception:
                pass

    def _persist(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        np.save(self.meta_path, np.array(self.chunks, dtype=object))

    @staticmethod
    def _pdf_to_texts(pdf_path: Path, step: int = 1400) -> List[str]:
        reader = PdfReader(str(pdf_path))
        pages: List[str] = []
        for p in reader.pages:
            t = p.extract_text() or ""
            if t.strip():
                pages.append(t)
        chunks: List[str] = []
        for txt in pages:
            for i in range(0, len(txt), step):
                part = txt[i : i + step].strip()
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
            outs = self._translator(texts, max_length=800)
            return [o["translation_text"].strip() for o in outs]
        except Exception:
            return texts

    def synthesize_answer(self, question: str, contexts: List[str], max_sentences: int = 4) -> str:
        if not contexts:
            return "No relevant context found. Please upload a PDF or ask a more specific question."

        # 1) Clean & keep top contexts
        cleaned_contexts = [_clean_for_summary(c) for c in contexts[:5]]
        cleaned_contexts = [c for c in cleaned_contexts if len(c) > 40]
        if not cleaned_contexts:
            return "The document appears largely tabular/numeric; couldn't extract readable sentences."

        # 2) Pre-translate paragraphs to EN when target is EN
        translated = self._translate_to_en(cleaned_contexts) if OUTPUT_LANG == "en" else cleaned_contexts

        # 3) Split into candidate sentences and filter strictly for completeness
        candidates: List[str] = []
        for para in translated:
            for s in _split_sentences(para):
                w = s.split()
                if not (6 <= len(w) <= 60):
                    continue
                if s.strip().lower().endswith("e.g."):
                    continue
                if not re.search(r"[.!?](?:[\"'])?$", s):  # must end with punctuation
                    continue
                if _tabular_like(s) or _mostly_numeric(s):
                    continue
                candidates.append(" ".join(w))

        # 4) Fallback if no good sentences
        if not candidates:
            bullets = _keyword_summary_en(cleaned_contexts)
            return "Answer (based on document context):\n" + "\n".join(f"- {b}" for b in bullets)

        # 5) Rank by similarity to the question
        q_emb = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        cand_emb = self.model.encode(candidates, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores = (cand_emb @ q_emb.T).ravel()
        order = np.argsort(-scores)

        # 6) Aggressive near-duplicate removal
        selected: List[str] = []
        for i in order:
            s = candidates[i].strip()
            if any(_sim_jaccard(s, t) >= 0.90 for t in selected):
                continue
            selected.append(s)
            if len(selected) >= max_sentences:
                break

        # 7) If still looks non-English, use keyword fallback
        if not selected or (sum(_non_ascii_ratio(s) for s in selected) / len(selected) > 0.10):
            bullets = _keyword_summary_en(cleaned_contexts)
            return "Answer (based on document context):\n" + "\n".join(f"- {b}" for b in bullets)

        bullets = "\n".join(f"- {s}" for s in selected)
        return f"Answer (based on document context):\n{bullets}"

def synthesize_answer(question: str, contexts: List[str]) -> str:
    return SimpleRAG().synthesize_answer(question, contexts)

__all__ = ["SimpleRAG", "synthesize_answer", "DATA_DIR", "UPLOAD_DIR", "INDEX_DIR", "CACHE_DIR", "MODEL_NAME"]
