"""End-to-end question answering: give a model pages, read its answer, score it.

Modes:
  retrieved  the top-k pages from your retrieval run (the real pipeline)
  oracle     only the gold page (what accuracy would be with perfect retrieval)

Run from the project root (try a few questions first):
    uv run python -m finqa_rag_lora.eval_qa --mode retrieved --limit 5
    uv run python -m finqa_rag_lora.eval_qa --mode retrieved
    uv run python -m finqa_rag_lora.eval_qa --mode oracle

Progress is saved after every question; rerunning the same command resumes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import pandas as pd

from finqa_rag_lora.embed import load_config
from finqa_rag_lora.generate import Generation, RateLimiter, call_with_retry, make_generator
from finqa_rag_lora.prompts import SYSTEM_PROMPT, build_prompt, parse_answer
from finqa_rag_lora.scoring import score

# Settings that change the results. If any differ from a saved run, resuming is refused.
RESULT_KEYS = ["generator", "model", "temperature", "max_output_tokens", "thinking_budget", "k"]
MAX_CONSECUTIVE_ERRORS = 10


def run_settings(cfg: dict, mode: str, split: str) -> dict:
    return {**{key: cfg.get(key) for key in RESULT_KEYS}, "mode": mode, "split": split,
            "system_prompt": SYSTEM_PROMPT}


def contexts_for(questions: pd.DataFrame, mode: str, k: int, retrieval_path: Path) -> dict[str, list[str]]:
    """Which page ids each question's prompt will contain."""
    if mode == "oracle":
        return {q.id: [q.gold_doc_id] for q in questions.itertuples()}
    if not retrieval_path.exists():
        raise FileNotFoundError(f"{retrieval_path} not found: run finqa_rag_lora.eval_retrieval first.")
    retrieved = pd.read_json(retrieval_path, lines=True).set_index("id")["retrieved"]
    missing = set(questions["id"]) - set(retrieved.index)
    if missing:
        raise ValueError(f"{len(missing)} eval questions aren't in {retrieval_path}. "
                         "The eval set changed since retrieval ran: rerun finqa_rag_lora.eval_retrieval.")
    return {qid: list(retrieved[qid])[:k] for qid in questions["id"]}


def answer_one(q, doc_ids: list[str], corpus: pd.DataFrame, generator: Callable[[str], Generation],
               limiter: RateLimiter, max_retries: int) -> dict:
    docs = [{"text": corpus.at[d, "text"], "company_name": corpus.at[d, "company_name"],
             "report_year": corpus.at[d, "report_year"]} for d in doc_ids]
    prompt = build_prompt(q.question, docs)
    limiter.acquire()
    start = time.monotonic()
    gen = call_with_retry(generator, prompt, max_retries=max_retries)
    latency = time.monotonic() - start
    parsed = parse_answer(gen.text)
    gold = float(q.answer)
    return {
        "id": q.id, "question": q.question, "gold": gold, "pred": parsed.value,
        **score(parsed.value, gold),
        "format_ok": parsed.format_ok, "expression": parsed.expression, "parse_error": parsed.error,
        "doc_ids": doc_ids, "gold_in_context": q.gold_doc_id in doc_ids,
        "response": gen.text, "finish_reason": gen.finish_reason,
        "latency_s": round(latency, 2), **gen.usage,
    }


def compute_metrics(preds: pd.DataFrame) -> dict:
    def rate(mask) -> float:
        return round(float(mask.mean()), 4) if len(mask) else float("nan")

    m = {
        "n": len(preds),
        "accuracy_strict": rate(preds["strict"]),
        "accuracy_lenient": rate(preds["lenient"]),
        "format_rate": rate(preds["format_ok"]),
        "sign_flip_errors": int(preds["sign_flip"].sum()),
        "gold_in_context_rate": rate(preds["gold_in_context"]),
        "accuracy_strict_gold_in_context": rate(preds.loc[preds["gold_in_context"], "strict"]),
        "mean_latency_s": round(float(preds["latency_s"].mean()), 2),
    }
    for col in ["prompt_tokens", "output_tokens", "thinking_tokens"]:
        if col in preds and preds[col].notna().any():
            m[f"mean_{col}"] = round(float(preds[col].mean()), 1)
    return m


def log_to_mlflow(run_name: str, settings: dict, metrics: dict) -> None:
    try:
        import mlflow
    except ImportError:
        return
    try:
        mlflow.set_experiment("finqa-qa")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: v for k, v in settings.items() if k != "system_prompt"})
            mlflow.log_text(settings["system_prompt"], "system_prompt.txt")
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    except Exception as err:  # logging must never lose a finished run
        print(f"(MLflow logging skipped: {err})")


def main(config_path: str, mode: str, split: str, limit: int | None, overwrite: bool,
         generator_factory: Callable[[dict], Callable[[str], Generation]] = make_generator) -> dict | None:
    cfg = load_config(config_path)
    questions = pd.read_json(Path(cfg["eval_dir"]) / f"eval_{split}.jsonl", lines=True)
    if limit:
        questions = questions.head(limit)
    corpus = pd.read_parquet(cfg["corpus_path"]).set_index("doc_id")
    contexts = contexts_for(questions, mode, cfg["k"],
                            Path(cfg["retrieval_dir"]) / f"retrieval_{split}.jsonl")

    run_name = f"{cfg['run_name']}_{mode}_{split}" + (f"_limit{limit}" if limit else "")
    run_dir = Path(cfg["results_dir"]) / run_name
    settings = run_settings(cfg, mode, split)

    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    settings_path, preds_path = run_dir / "settings.json", run_dir / "predictions.jsonl"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError(f"{run_dir} was produced with different settings. Change run_name in the config "
                         "for a new experiment, or pass --overwrite to replace it.")
    settings_path.write_text(json.dumps(settings, indent=2))

    done = set(pd.read_json(preds_path, lines=True)["id"]) if preds_path.exists() and preds_path.stat().st_size else set()
    todo = [q for q in questions.itertuples() if q.id not in done]
    print(f"{run_name}: {len(done)} already answered, {len(todo)} to go")

    generator = generator_factory(cfg) if todo else None
    limiter = RateLimiter(cfg.get("requests_per_minute"))
    failures, consecutive = {}, 0
    with ThreadPoolExecutor(max_workers=cfg.get("concurrency", 4)) as pool, open(preds_path, "a") as out:
        futures = {pool.submit(answer_one, q, contexts[q.id], corpus, generator, limiter,
                               cfg.get("max_retries", 8)): q.id for q in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            qid = futures[fut]
            try:
                row = fut.result()
            except Exception as err:
                failures[qid] = f"{type(err).__name__}: {err}"
                consecutive += 1
                if consecutive >= MAX_CONSECUTIVE_ERRORS:
                    pool.shutdown(cancel_futures=True)
                    raise RuntimeError(f"Stopping: {consecutive} calls failed in a row. Last error: {failures[qid]}") from err
                continue
            consecutive = 0
            out.write(json.dumps(row) + "\n")
            out.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {len(done) + i - len(failures)}/{len(questions)} answered")

    if failures:
        print(f"\n{len(failures)} questions failed (they'll be retried if you rerun). First errors:")
        for qid, err in list(failures.items())[:3]:
            print(f"  {qid}: {err[:200]}")
        return None

    preds = pd.read_json(preds_path, lines=True)
    preds = preds[preds["id"].isin(questions["id"])]
    metrics = compute_metrics(preds)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log_to_mlflow(run_name, settings, metrics)

    print(f"\n{run_name} ({metrics['n']} questions)")
    for key, value in metrics.items():
        print(f"  {key:<34} {value}")
    print(f"\nPer-question results: {preds_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/qa_gemini.yaml")
    parser.add_argument("--mode", default="retrieved", choices=["retrieved", "oracle"])
    parser.add_argument("--split", default="test", choices=["test", "dev"])
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke test)")
    parser.add_argument("--overwrite", action="store_true", help="delete this run's saved results and start over")
    args = parser.parse_args()
    main(args.config, args.mode, args.split, args.limit, args.overwrite)