# app/rag_system.py
from pathlib import Path
from typing import List, Tuple
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

class SimpleRAG:
    def __init__(self, index_path: Path = INDEX_DIR / "faiss.index", meta_path: Path = INDEX_DIR / "meta.npy"):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.chunks: List[str] = []
        self._load()

    def _load(self):
        # meta (chunks) yüklə
        if self.meta_path.exists():
            self.chunks = np.load(self.meta_path, allow_pickle=True).tolist()
        # faiss index yüklə
        if self.index_path.exists():
            # dim modelin çıxış ölçüsü
            dim = self.model.get_sentence_embedding_dimension()
            self.index = faiss.read_index(str(self.index_path))
            # təhlükəsizlik: ölçüsü uyğun olmalıdır
            if self.index.d != dim:
                # uyğunsuzluqda sıfırdan başla
                self.index = faiss.IndexFlatIP(dim)
        else:
            dim = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatIP(dim)

    def _persist(self):
        faiss.write_index(self.index, str(self.index_path))
        np.save(self.meta_path, np.array(self.chunks, dtype=object))

    @staticmethod
    def _pdf_to_texts(pdf_path: Path) -> List[str]:
        reader = PdfReader(str(pdf_path))
        full_text = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                full_text.append(t)
        # sadə parçalama: ~500 hərf
        chunks = []
        for txt in full_text:
            step = 800
            for i in range(0, len(txt), step):
                chunks.append(txt[i:i+step])
        return chunks

    def add_pdf(self, pdf_path: Path) -> int:
        texts = self._pdf_to_texts(pdf_path)
        if not texts:
            return 0
        emb = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.index.add(emb)
        self.chunks.extend(texts)
        self._persist()
        return len(texts)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q, k)
        results = []
        if I.size > 0 and len(self.chunks) > 0:
            for idx, score in zip(I[0], D[0]):
                if 0 <= idx < len(self.chunks):
                    results.append((self.chunks[idx], float(score)))
        return results

# sadə cavab formalaşdırıcı (LLM yoxdursa, kontekst + heuristika)
def synthesize_answer(question: str, contexts: List[str]) -> str:
    if not contexts:
        return "Kontekst tapılmadı. Sualı daha dəqiq verin və ya PDF yükləyin."
    joined = "\n---\n".join(contexts[:3])
    return (
        f"Sual: {question}\n\n"
        f"Cavab (kontekstdən çıxarış):\n{joined}\n\n"
        f"(Qeyd: Demo rejimi — LLM inteqrasiyası üçün / later: OpenAI/Groq və s.)"
    )
