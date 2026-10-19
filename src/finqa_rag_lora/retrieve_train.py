"""Retrieve pages for the TRAIN questions, to build LoRA training prompts.

Two differences from test-time retrieval:
  1. Only train pages are searched. Otherwise a test page could appear as a distractor in a
     training prompt, and the model would see test material during training (leakage).
  2. The gold page is always included. Training examples must be answerable; when retrieval
     misses the gold page, it replaces the lowest-ranked result at a random position.

Run from the project root:
    uv run python -m finqa_rag_lora.retrieve_train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from finqa_rag_lora.embed import embed_corpus, load_config, make_gemini_embed_fn
from finqa_rag_lora.retrieve import DenseRetriever


def train_page_ids(train_q: pd.DataFrame, held_out: list[pd.DataFrame]) -> set[str]:
    """Pages used by train questions, minus any page that test or dev questions use."""
    held = set().union(*(set(h["gold_doc_id"]) for h in held_out)) if held_out else set()
    return set(train_q["gold_doc_id"]) - held


def ensure_gold(retrieved: list[str], gold: str, k: int, rng: np.random.Generator) -> list[str]:
    """Top-k pages, with the gold page guaranteed to be one of them."""
    top = list(retrieved[:k])
    if gold in top:
        return top
    top = top[:k - 1]
    top.insert(int(rng.integers(0, k)), gold)
    return top


def main(embed_config: str, k: int, seed: int) -> None:
    ecfg = load_config(embed_config)
    data_dir = Path(ecfg["corpus_path"]).parent
    train_q = pd.read_json(data_dir / "questions_train.jsonl", lines=True)
    held_out = [pd.read_json(data_dir / f"eval_{s}.jsonl", lines=True) for s in ("test", "dev")]

    # Restrict the index to train pages.
    all_ids = json.loads(Path(ecfg["ids_path"]).read_text())
    all_vecs = np.load(ecfg["vectors_path"])
    allowed = train_page_ids(train_q, held_out)
    keep = [i for i, d in enumerate(all_ids) if d in allowed]
    retriever = DenseRetriever(all_vecs[keep], [all_ids[i] for i in keep])
    print(f"Searching {len(keep)} train pages (of {len(all_ids)} total) for {len(train_q)} train questions")

    # Embed the train questions (cached and resumable, like everything else).
    query_fn = make_gemini_embed_fn(ecfg["model"], ecfg["query_task_type"], ecfg["output_dimensionality"])
    q_vecs = embed_corpus(pd.DataFrame({"doc_id": train_q["id"], "text": train_q["question"]}), query_fn,
                          data_dir / "query_embeddings_train.npy", data_dir / "query_ids_train.json",
                          ecfg["batch_size"], ecfg["max_retries"],
                          requests_per_minute=ecfg.get("requests_per_minute"))

    retrieved, _ = retriever.search(q_vecs, k)
    rng = np.random.default_rng(seed)
    rows, hits = [], 0
    for q, ids in zip(train_q.itertuples(), retrieved):
        hits += q.gold_doc_id in ids
        rows.append({"id": q.id, "gold_doc_id": q.gold_doc_id, "gold_retrieved": q.gold_doc_id in ids,
                     "retrieved": ensure_gold(ids, q.gold_doc_id, k, rng)})

    out = Path("results") / "retrieval_train.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(out, orient="records", lines=True)
    print(f"Recall@{k} on train (before adding missing gold pages): {hits / len(rows):.3f}")
    print(f"Saved {len(rows)} training contexts -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embed-config", default="configs/embed.yaml")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.embed_config, args.k, args.seed)