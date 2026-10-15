"""Dense retrieval over the corpus with a FAISS index."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to length 1, so inner product = cosine similarity."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class DenseRetriever:
    """Exact cosine-similarity search. With ~2,800 pages, exact search is fast; no approximate index needed."""

    def __init__(self, doc_vectors: np.ndarray, doc_ids: list[str]):
        if len(doc_vectors) != len(doc_ids):
            raise ValueError(f"{len(doc_vectors)} vectors but {len(doc_ids)} ids")
        self.doc_ids = list(doc_ids)
        vectors = normalize(doc_vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    @classmethod
    def from_files(cls, vectors_path: str | Path, ids_path: str | Path) -> "DenseRetriever":
        return cls(np.load(vectors_path), json.loads(Path(ids_path).read_text()))

    def search(self, query_vectors: np.ndarray, k: int) -> tuple[list[list[str]], np.ndarray]:
        """Top-k doc ids (best first) and their similarity scores, one row per query."""
        k = min(k, len(self.doc_ids))
        scores, idx = self.index.search(normalize(query_vectors), k)
        return [[self.doc_ids[i] for i in row] for row in idx], scores