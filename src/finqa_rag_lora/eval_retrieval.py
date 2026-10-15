"""Measure retrieval quality: how often is the gold page among the top k?

Run from the project root:
    uv run python -m finqa_rag_lora.eval_retrieval              # test set (300 questions)
    uv run python -m finqa_rag_lora.eval_retrieval --split dev  # dev set (100 questions)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from finqa_rag_lora.embed import embed_corpus, load_config, make_gemini_embed_fn
from finqa_rag_lora.retrieve import DenseRetriever


def gold_rank(retrieved: list[str], gold: str) -> int | None:
    """1-based position of the gold page in the results, or None if it wasn't retrieved."""
    return retrieved.index(gold) + 1 if gold in retrieved else None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    """Share of questions whose gold page is in the top k."""
    return sum(r is not None and r <= k for r in ranks) / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    """Mean reciprocal rank: 1 if gold is first, 1/2 if second, ..., 0 if not retrieved."""
    return sum(1 / r if r is not None else 0.0 for r in ranks) / len(ranks)


def classify_miss(top1: dict, gold: dict) -> str:
    """For a question whose top result is wrong, how wrong was it?"""
    if top1["company_name"] != gold["company_name"]:
        return "wrong company"
    if str(top1["report_year"]) != str(gold["report_year"]):
        return "right company, wrong year"
    return "right company and year, wrong page"


def main(split: str, embed_config: str, retrieval_config: str) -> None:
    ecfg, rcfg = load_config(embed_config), load_config(retrieval_config)
    questions = pd.read_json(Path(rcfg["eval_dir"]) / f"eval_{split}.jsonl", lines=True)
    corpus = pd.read_parquet(ecfg["corpus_path"]).set_index("doc_id")

    # Embed the questions with the QUERY task type (cached, like the corpus).
    query_fn = make_gemini_embed_fn(ecfg["model"], ecfg["query_task_type"], ecfg["output_dimensionality"])
    q_df = pd.DataFrame({"doc_id": questions["id"], "text": questions["question"]})
    q_vectors = embed_corpus(q_df, query_fn,
                             Path(rcfg["query_cache_dir"]) / f"query_embeddings_{split}.npy",
                             Path(rcfg["query_cache_dir"]) / f"query_ids_{split}.json",
                             ecfg["batch_size"], ecfg["max_retries"],
                             requests_per_minute=ecfg.get("requests_per_minute"))

    retriever = DenseRetriever.from_files(ecfg["vectors_path"], ecfg["ids_path"])
    max_k = max(rcfg["ks"])
    retrieved, scores = retriever.search(q_vectors, max_k)

    rows, ranks = [], []
    for q, ids, sc in zip(questions.itertuples(), retrieved, scores):
        rank = gold_rank(ids, q.gold_doc_id)
        ranks.append(rank)
        top1 = corpus.loc[ids[0]]
        rows.append({
            "id": q.id, "question": q.question, "gold_doc_id": q.gold_doc_id, "gold_rank": rank,
            "retrieved": ids, "scores": [round(float(s), 4) for s in sc],
            "miss_type": None if rank == 1 else classify_miss(top1, corpus.loc[q.gold_doc_id]),
        })

    metrics = {f"recall@{k}": round(recall_at_k(ranks, k), 4) for k in rcfg["ks"]}
    metrics["mrr"] = round(mrr(ranks), 4)

    out_dir = Path(rcfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(out_dir / f"retrieval_{split}.jsonl", orient="records", lines=True)
    (out_dir / f"retrieval_{split}_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nRetrieval on {split} ({len(questions)} questions)")
    for name, value in metrics.items():
        print(f"  {name:<10} {value:.3f}")
    misses = pd.Series([r["miss_type"] for r in rows if r["miss_type"]]).value_counts()
    print(f"\nWhen the top result is wrong ({misses.sum()} questions):")
    for kind, n in misses.items():
        print(f"  {kind:<36} {n}")
    print(f"\nPer-question results: {out_dir / f'retrieval_{split}.jsonl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["test", "dev"])
    parser.add_argument("--embed-config", default="configs/embed.yaml")
    parser.add_argument("--config", default="configs/retrieval.yaml")
    args = parser.parse_args()
    main(args.split, args.embed_config, args.config)