from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from src.run_metrics import run_metrics

try:
    import faiss
except ImportError:  # pragma: no cover - optional dependency at runtime
    faiss = None


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RetrievedChunk:
    text: str
    score: float
    index: int


class FaissTextIndex:
    def __init__(self, chunks: list[str], embeddings: np.ndarray, index) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self.index = index

    @classmethod
    def from_text(
        cls,
        text: str,
        chunk_size: int = 1200,
        overlap: int = 150,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> "FaissTextIndex":
        run_metrics.increment("rag.index_build.calls")
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            raise ValueError("Cannot build a FAISS index from empty text.")

        embeddings = embed_texts(chunks, model_name=model_name)
        with run_metrics.timed("rag.index_build.seconds"):
            index = build_faiss_index(embeddings)
        return cls(chunks=chunks, embeddings=embeddings, index=index)

    def search(
        self,
        query: str,
        top_k: int = 5,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        run_metrics.increment("rag.search.calls")
        query_embedding = embed_texts([query], model_name=model_name)
        with run_metrics.timed("rag.search.seconds"):
            distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))

        results = []
        for score, chunk_index in zip(distances[0], indices[0]):
            if chunk_index < 0:
                continue
            results.append(
                RetrievedChunk(
                    text=self.chunks[chunk_index],
                    score=float(score),
                    index=int(chunk_index),
                )
            )
        return results

    def build_context(
        self,
        query: str,
        top_k: int = 5,
        max_context_chars: int = 6000,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> str:
        selected = []
        total_chars = 0

        for chunk in self.search(query=query, top_k=top_k, model_name=model_name):
            if total_chars + len(chunk.text) > max_context_chars and selected:
                break
            selected.append(chunk.text)
            total_chars += len(chunk.text)

        return "\n\n".join(selected)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(normalized), step):
        chunk = normalized[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(normalized):
            break
    run_metrics.increment("rag.chunking.calls")
    run_metrics.increment("rag.chunks.created", len(chunks))
    return chunks


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    run_metrics.increment("rag.embedding.calls")
    run_metrics.increment("rag.embedding.texts_total", len(texts))
    with run_metrics.timed("rag.embedding.seconds"):
        model = get_embedding_model(model_name)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    return np.asarray(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray):
    if faiss is None:
        raise RuntimeError("faiss is not installed. Install faiss-cpu to use this module.")

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Expected a non-empty 2D embeddings array.")

    dimension = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def build_rag_context(
    document_text: str,
    query: str,
    top_k: int = 5,
    max_context_chars: int = 6000,
    chunk_size: int = 1200,
    overlap: int = 150,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> str:
    index = FaissTextIndex.from_text(
        document_text,
        chunk_size=chunk_size,
        overlap=overlap,
        model_name=model_name,
    )
    return index.build_context(
        query=query,
        top_k=top_k,
        max_context_chars=max_context_chars,
        model_name=model_name,
    )
