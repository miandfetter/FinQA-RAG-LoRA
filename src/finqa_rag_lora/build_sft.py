"""Turn the teacher's correct answers into LoRA training examples (chat format).

Keeps a teacher answer only if it is strictly correct AND ends in a valid CALC line, so the
student learns both the right reasoning and the right output format. Wrong answers, including
those caused by bad labels, are dropped.

Run from the project root, after the teacher run on the train split:
    uv run python -m finqa_rag_lora.build_sft --predictions results/qa/<teacher_run>/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from finqa_rag_lora.prompts import SYSTEM_PROMPT, build_prompt

_LAST_CALC = re.compile(r"(?is)^(.*CALC\W*?:[^\n]*)")


def trim_after_calc(response: str) -> str:
    """Cut anything the teacher wrote after its final CALC line."""
    match = _LAST_CALC.match(response.strip())
    return match.group(1).strip() if match else response.strip()


def to_example(row, corpus: pd.DataFrame) -> dict:
    docs = [{"text": corpus.at[d, "text"], "company_name": corpus.at[d, "company_name"],
             "report_year": corpus.at[d, "report_year"]} for d in row.doc_ids]
    return {"id": row.id, "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(row.question, docs)},
        {"role": "assistant", "content": trim_after_calc(row.response)},
    ]}


def build_sft(preds: pd.DataFrame, corpus: pd.DataFrame, val_fraction: float, seed: int):
    """Returns (train_examples, val_examples). The validation set is for watching training loss."""
    keep = preds[preds["strict"] & preds["format_ok"]].sort_values("id")
    examples = [to_example(r, corpus) for r in keep.itertuples()]
    order = np.random.default_rng(seed).permutation(len(examples))
    n_val = int(round(len(examples) * val_fraction))
    val_idx = set(order[:n_val].tolist())
    train = [e for i, e in enumerate(examples) if i not in val_idx]
    val = [e for i, e in enumerate(examples) if i in val_idx]
    return train, val


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main(predictions: str, corpus_path: str, out_dir: str, val_fraction: float, seed: int) -> None:
    preds = pd.read_json(predictions, lines=True)
    corpus = pd.read_parquet(corpus_path).set_index("doc_id")
    train, val = build_sft(preds, corpus, val_fraction, seed)
    out = Path(out_dir)
    write_jsonl(train, out / "sft_train.jsonl")
    write_jsonl(val, out / "sft_val.jsonl")
    kept = len(train) + len(val)
    print(f"Teacher answers: {len(preds)}  kept (strictly correct, valid CALC): {kept} ({kept / len(preds):.1%})")
    print(f"  train: {len(train)} -> {out / 'sft_train.jsonl'}")
    print(f"  val:   {len(val)} -> {out / 'sft_val.jsonl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--corpus", default="data/corpus.parquet")
    parser.add_argument("--out-dir", default="data/sft")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.predictions, args.corpus, args.out_dir, args.val_fraction, args.seed)