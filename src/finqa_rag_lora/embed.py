"""Embed the corpus pages with a Gemini embedding model and cache the vectors.

Run from the project root (try a small --limit first to check cost and setup):
    uv run python -m finqa_rag_lora.embed --limit 20
    uv run python -m finqa_rag_lora.embed
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import yaml

# An embed function takes a batch of texts and returns one vector per text.
EmbedFn = Callable[[Sequence[str]], list[list[float]]]


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_gemini_embed_fn(model: str, task_type: str | None, output_dim: int | None) -> EmbedFn:
    """Build an embed function that calls the Gemini embedding API.

    Settings come from .env (GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_GENAI_USE_VERTEXAI).
    """
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv()
    client = genai.Client()
    config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=output_dim)

    def embed(texts: Sequence[str]) -> list[list[float]]:
        # One Content per text, so each text gets its own vector
        # (several Parts in one Content would be merged into a single embedding).
        contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]
        resp = client.models.embed_content(model=model, contents=contents, config=config)
        return [e.values for e in resp.embeddings]

    return embed


def _is_retryable(err: Exception) -> bool:
    """Rate limits (429) and server errors (5xx) are worth retrying; bad requests are not."""
    code = getattr(err, "code", None)
    return code == 429 or (isinstance(code, int) and code >= 500)


def embed_with_retry(embed_fn: EmbedFn, texts: Sequence[str], max_retries: int = 5,
                     base_delay: float = 2.0, sleep: Callable[[float], None] = time.sleep) -> list[list[float]]:
    """Call embed_fn, waiting longer after each retryable failure (exponential backoff)."""
    for attempt in range(max_retries + 1):
        try:
            return embed_fn(texts)
        except Exception as err:
            if attempt == max_retries or not _is_retryable(err):
                raise
            delay = min(base_delay * 2**attempt, 60.0)  # cap: quotas reset each minute
            print(f"  retryable error ({getattr(err, 'code', '?')}); waiting {delay:.0f}s")
            sleep(delay)
    raise AssertionError("unreachable")


def embed_texts(texts: Sequence[str], embed_fn: EmbedFn, batch_size: int,
                max_retries: int = 5, sleep: Callable[[float], None] = time.sleep,
                requests_per_minute: float | None = None,
                done: list[list[float]] | None = None,
                on_batch: Callable[[int, list[list[float]]], None] | None = None,
                clock: Callable[[], float] = time.monotonic) -> np.ndarray:
    """Embed texts in batches; returns an array of shape (len(texts), dim), in input order.

    requests_per_minute: pace requests evenly to stay under the API quota (None = no pacing).
    done:     vectors already embedded for the first texts (to resume after a crash).
    on_batch: called as on_batch(start, batch_vectors) after each batch (used to save progress).
    """
    empty = [i for i, t in enumerate(texts) if not str(t or "").strip()]
    if empty:
        raise ValueError(f"{len(empty)} empty texts (positions {empty[:5]}...); the API rejects them. "
                         "Remove them before embedding.")

    vectors: list[list[float]] = list(done or [])
    min_gap = 60.0 / requests_per_minute if requests_per_minute else 0.0
    last_call = None
    if vectors:
        print(f"  resuming from {len(vectors)}/{len(texts)}")

    for start in range(len(vectors), len(texts), batch_size):
        if last_call is not None and min_gap:
            wait = min_gap - (clock() - last_call)
            if wait > 0:
                sleep(wait)
        last_call = clock()

        batch = list(texts[start:start + batch_size])
        out = embed_with_retry(embed_fn, batch, max_retries=max_retries, sleep=sleep)
        if len(out) != len(batch):
            raise RuntimeError(
                f"Sent {len(batch)} texts but got {len(out)} vectors. "
                "This model may not accept several texts per request: set batch_size: 1 in the config."
            )
        if on_batch:
            on_batch(start, out)
        vectors.extend(out)
        print(f"  embedded {len(vectors)}/{len(texts)}")
    return np.asarray(vectors, dtype=np.float32)


def load_cached(vectors_path: Path, ids_path: Path, expected_ids: list[str]) -> np.ndarray | None:
    """Return cached vectors only if they exist AND match the current corpus, in order."""
    if not (vectors_path.exists() and ids_path.exists()):
        return None
    cached_ids = json.loads(ids_path.read_text())
    if cached_ids != expected_ids:
        print("Cached embeddings don't match the corpus; re-embedding.")
        return None
    return np.load(vectors_path)


def _load_progress(progress_dir: Path, ids: list[str]) -> list[list[float]]:
    """Vectors saved by an earlier, interrupted run (one small file per batch), if they match this corpus."""
    vectors: list[list[float]] = []
    for chunk_path in sorted(progress_dir.glob("batch_*.npz")):
        chunk = np.load(chunk_path)
        start = int(chunk_path.stem.split("_")[1])
        chunk_ids = chunk["ids"].tolist()
        if start != len(vectors) or chunk_ids != ids[start:start + len(chunk_ids)]:
            print("Saved progress doesn't match the corpus; starting over.")
            shutil.rmtree(progress_dir)
            return []
        vectors.extend(chunk["vectors"].tolist())
    return vectors


def embed_corpus(corpus: pd.DataFrame, embed_fn: EmbedFn, vectors_path: Path, ids_path: Path,
                 batch_size: int, max_retries: int = 5, sleep: Callable[[float], None] = time.sleep,
                 requests_per_minute: float | None = None) -> np.ndarray:
    """Embed every page in the corpus, or reuse the cache if it's still valid.

    Progress is saved after every batch, so an interrupted run picks up where it stopped.
    """
    ids = corpus["doc_id"].tolist()
    cached = load_cached(vectors_path, ids_path, ids)
    if cached is not None:
        print(f"Using cached embeddings: {vectors_path} {cached.shape}")
        return cached

    progress_dir = vectors_path.with_name(vectors_path.stem + "_progress")
    progress_dir.mkdir(parents=True, exist_ok=True)

    def save_batch(start: int, batch_vectors: list[list[float]]) -> None:
        progress_dir.mkdir(parents=True, exist_ok=True)  # may have been cleared by _load_progress
        final = progress_dir / f"batch_{start:06d}.npz"
        tmp = progress_dir / f"tmp_{start:06d}.npz"
        np.savez(tmp, vectors=np.asarray(batch_vectors, dtype=np.float32),
                 ids=np.asarray(ids[start:start + len(batch_vectors)]))
        tmp.replace(final)  # rename is atomic: a crash mid-write can't leave a half-written batch

    vectors = embed_texts(corpus["text"].tolist(), embed_fn, batch_size, max_retries, sleep,
                          requests_per_minute=requests_per_minute,
                          done=_load_progress(progress_dir, ids), on_batch=save_batch)
    np.save(vectors_path, vectors)
    ids_path.write_text(json.dumps(ids))
    shutil.rmtree(progress_dir)
    return vectors


def main(config_path: str, limit: int | None) -> None:
    cfg = load_config(config_path)
    corpus = pd.read_parquet(cfg["corpus_path"])
    vectors_path, ids_path = Path(cfg["vectors_path"]), Path(cfg["ids_path"])

    if limit:  # smoke test: small sample, separate files so the real cache isn't touched
        corpus = corpus.head(limit)
        vectors_path = vectors_path.with_name(f"{vectors_path.stem}_sample{limit}.npy")
        ids_path = ids_path.with_name(f"{ids_path.stem}_sample{limit}.json")

    embed_fn = make_gemini_embed_fn(cfg["model"], cfg["doc_task_type"], cfg["output_dimensionality"])
    vectors = embed_corpus(corpus, embed_fn, vectors_path, ids_path, cfg["batch_size"], cfg["max_retries"],
                           requests_per_minute=cfg.get("requests_per_minute"))

    norms = np.linalg.norm(vectors, axis=1)
    print(f"pages embedded: {len(vectors)}  dim: {vectors.shape[1]}  -> {vectors_path}")
    print(f"vector norms:   min {norms.min():.3f}  max {norms.max():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embed.yaml")
    parser.add_argument("--limit", type=int, default=None, help="embed only the first N pages (smoke test)")
    args = parser.parse_args()
    main(args.config, args.limit)