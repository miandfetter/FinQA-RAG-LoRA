"""Build the retrieval corpus and fixed evaluation sets for T²-RAGBench FinQA.

Run from the project root:
    uv run python -m src.data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

DATASET_NAME = "G4KMU/t2-ragbench"
DATASET_CONFIG = "FinQA"


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_finqa() -> pd.DataFrame:
    """Load all FinQA splits into one DataFrame (the `split` column says which is which)."""
    from datasets import load_dataset  # imported here so tests don't need it

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)
    return pd.concat([ds[name].to_pandas() for name in ds.keys()], ignore_index=True)


def build_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique page, drawn from ALL splits so retrieval is realistically hard."""
    # Guard the assumption that a context_id always maps to the same text.
    texts_per_id = df.groupby("context_id")["context"].nunique()
    inconsistent = texts_per_id[texts_per_id > 1]
    assert inconsistent.empty, f"{len(inconsistent)} context_ids have more than one text, e.g. {list(inconsistent.index[:3])}"

    corpus = (
        df.drop_duplicates("context_id")
        .rename(columns={"context_id": "doc_id", "context": "text"})
        [["doc_id", "text", "company_name", "report_year", "file_name"]]
        .sort_values("doc_id")
        .reset_index(drop=True)
    )
    corpus["report_year"] = corpus["report_year"].astype(str)
    return corpus


def questions_for_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """All usable questions in one split, in the format the rest of the pipeline expects."""
    pool = df[df["split"] == split]
    pool = pool[pool["question"].fillna("").str.strip() != ""]  # a few questions in the dataset are empty
    return (
        pool.rename(columns={"program_answer": "answer", "context_id": "gold_doc_id"})
        [["id", "question", "answer", "gold_doc_id", "company_name", "report_year"]]
        .reset_index(drop=True)
        .assign(report_year=lambda d: d["report_year"].astype(str))
    )


def sample_eval_set(df: pd.DataFrame, split: str, n: int, seed: int) -> pd.DataFrame:
    """A fixed random sample of questions from one split."""
    pool = questions_for_split(df, split)
    if n > len(pool):
        raise ValueError(f"Asked for {n} questions but split '{split}' only has {len(pool)}")
    return pool.sample(n=n, random_state=seed).reset_index(drop=True)


def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    df = load_finqa()

    corpus = build_corpus(df)
    corpus.to_parquet(out / "corpus.parquet", index=False)

    test_set = sample_eval_set(df, "test", cfg["n_test"], cfg["seed"])
    dev_set = sample_eval_set(df, "dev", cfg["n_dev"], cfg["seed"])
    save_jsonl(test_set, out / "eval_test.jsonl")
    save_jsonl(dev_set, out / "eval_dev.jsonl")
    train_questions = questions_for_split(df, "train")  # ALL train questions: source of LoRA training data
    save_jsonl(train_questions, out / "questions_train.jsonl")

    # Small, committed copy of the test IDs so anyone can reproduce the eval set.
    ids_path = Path(cfg["test_ids_path"])
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text("\n".join(test_set["id"]) + "\n")

    empty = (df["question"].fillna("").str.strip() == "").groupby(df["split"]).sum()
    print(f"questions loaded: {len(df)}  (empty, excluded from sampling: {dict(empty[empty > 0])})")
    print(f"corpus pages:     {len(corpus)}  -> {out / 'corpus.parquet'}")
    print(f"test sample:      {len(test_set)}  -> {out / 'eval_test.jsonl'}")
    print(f"dev sample:       {len(dev_set)}  -> {out / 'eval_dev.jsonl'}")
    print(f"train questions:  {len(train_questions)}  -> {out / 'questions_train.jsonl'}")
    print(f"test IDs:         {ids_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    main(parser.parse_args().config)