# app/rag_system.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# Konfiqurasiya & qovluqlar
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"

# HF Spaces-də yazma icazəsi olan cache qovluğu
CACHE_DIR = Path(os.getenv("HF_HOME", str(ROOT_DIR / ".cache")))
for d in (DATA_DIR, UPLOAD_DIR, INDEX_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Model adı ENV-dən dəyişdirilə bilər
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class SimpleRAG:
    """
    Sadə RAG nüvəsi:
    - PDF -> mətn parçalama
    - Sentence-Transformers embeddings
    - FAISS Index (IP / cosine bərabərləşdirilmiş)
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

        # Model
        # cache_folder Spaces-də /.cache icazə xətasının qarşısını alır
        self.model = SentenceTransformer(self.model_name, cache_folder=str(self.cache_dir))
        self.embed_dim = self.model.get_sentence_embedding_dimension()

        # FAISS index və meta (chunks)
        self.index: faiss.Index = None  # type: ignore
        self.chunks: List[str] = []

        self._load()

    # -----------------------------
    # Yükləmə / Saxlama
    # -----------------------------
    def _load(self) -> None:
        # Chunks (meta) yüklə
        if self.meta_path.exists():
            try:
                self.chunks = np.load(self.meta_path, allow_pickle=True).tolist()
            except Exception:
                # zədələnmişsə sıfırla
                self.chunks = []

        # FAISS index yüklə
        if self.index_path.exists():
            try:
                idx = faiss.read_index(str(self.index_path))
                # ölçü uyğunluğunu yoxla
                if hasattr(idx, "d") and idx.d == self.embed_dim:
                    self.index = idx
                else:
                    # uyğunsuzluqda sıfırdan qur
                    self.index = faiss.IndexFlatIP(self.embed_dim)
            except Exception:
                self.index = faiss.IndexFlatIP(self.embed_dim)
        else:
            self.index = faiss.IndexFlatIP(self.embed_dim)

    def _persist(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        np.save(self.meta_path, np.array(self.chunks, dtype=object))

    # -----------------------------
    # PDF -> Mətn -> Parçalama
    # -----------------------------
    @staticmethod
    def _pdf_to_texts(pdf_path: Path, step: int = 800) -> List[str]:
        reader = PdfReader(str(pdf_path))
        pages_text: List[str] = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                pages_text.append(t)

        chunks: List[str] = []
        for txt in pages_text:
            for i in range(0, len(txt), step):
                chunk = txt[i : i + step].strip()
                if chunk:
                    chunks.append(chunk)
        return chunks

    # -----------------------------
    # Index-ə əlavə
    # -----------------------------
    def add_pdf(self, pdf_path: Path) -> int:
        texts = self._pdf_to_texts(pdf_path)
        if not texts:
            return 0

        emb = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        # FAISS-ə əlavə
        self.index.add(emb.astype(np.float32))
        # Meta-ya əlavə
        self.chunks.extend(texts)
        # Diskə yaz
        self._persist()
        return len(texts)

    # -----------------------------
    # Axtarış
    # -----------------------------
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None:
            return []

        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        # FAISS float32 gözləyir
        D, I = self.index.search(q.astype(np.float32), min(k, max(1, self.index.ntotal)))
        results: List[Tuple[str, float]] = []

        if I.size > 0 and self.chunks:
            for idx, score in zip(I[0], D[0]):
                if 0 <= idx < len(self.chunks):
                    results.append((self.chunks[idx], float(score)))
        return results

    # -----------------------------
    # Cavab Sinttezi (LLM-siz demo)
    # -----------------------------
    def synthesize_answer(self, question: str, contexts: List[str]) -> str:
        if not contexts:
            return "Kontekst tapılmadı. Sualı daha dəqiq verin və ya PDF yükləyin."
        joined = "\n---\n".join(contexts[:3])
        return (
            f"Sual: {question}\n\n"
            f"Cavab (kontekstdən çıxarış):\n{joined}\n\n"
            f"(Qeyd: Demo rejimi — LLM inteqrasiyası üçün sonradan OpenAI/Groq və s. əlavə edilə bilər.)"
        )


# Köhnə import yolunu dəstəkləmək üçün eyni funksiyanı modul səviyyəsində də saxlayırıq
def synthesize_answer(question: str, contexts: List[str]) -> str:
    if not contexts:
        return "Kontekst tapılmadı. Sualı daha dəqiq verin və ya PDF yükləyin."
    joined = "\n---\n".join(contexts[:3])
    return (
        f"Sual: {question}\n\n"
        f"Cavab (kontekstdən çıxarış):\n{joined}\n\n"
        f"(Qeyd: Demo rejimi — LLM inteqrasiyası üçün sonradan OpenAI/Groq və s. əlavə edilə bilər.)"
    )


# Faylı import edən tərəfin rahatlığı üçün bu qovluqları export edirik
__all__ = [
    "SimpleRAG",
    "synthesize_answer",
    "DATA_DIR",
    "UPLOAD_DIR",
    "INDEX_DIR",
    "CACHE_DIR",
    "MODEL_NAME",
]
