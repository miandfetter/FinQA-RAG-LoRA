"""Compare QA runs question by question: accuracy table + paired significance tests.

All runs answer the SAME 300 test questions, so the fair comparison is paired: count the
questions where only one of two models is right. McNemar's exact test asks whether that split
is lopsided enough to rule out chance; the bootstrap gives a 95% interval for the accuracy gap.

    python -m finqa_rag_lora.compare --runs qwen3_4b_base qwen3_4b_lora_r16 gemini_flash
"""

from __future__ import annotations

import argparse
import json
import random
from math import comb
from pathlib import Path

import pandas as pd


def load_run(results_dir: Path, run: str, mode: str, split: str) -> pd.DataFrame:
    path = results_dir / f"{run}_{mode}_{split}" / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: download the run from the bucket first.")
    df = pd.read_json(path, lines=True).drop_duplicates("id", keep="last").set_index("id")
    return df


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value (a binomial test on the disagreeing questions)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def bootstrap_diff(a: list[bool], b: list[bool], n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    """95% interval for accuracy(b) - accuracy(a), resampling questions (keeps the pairing)."""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]


def paired(a: pd.DataFrame, b: pd.DataFrame, metric: str = "strict") -> dict:
    ids = a.index.intersection(b.index)
    if len(ids) != len(a) or len(ids) != len(b):
        raise ValueError(f"runs cover different questions ({len(a)} vs {len(b)}, {len(ids)} shared)")
    x, y = a.loc[ids, metric].astype(bool).tolist(), b.loc[ids, metric].astype(bool).tolist()
    only_a = sum(xi and not yi for xi, yi in zip(x, y))
    only_b = sum(yi and not xi for xi, yi in zip(x, y))
    lo, hi = bootstrap_diff(x, y)
    return {"n": len(ids), "diff": (sum(y) - sum(x)) / len(ids), "ci95": (lo, hi),
            "only_a": only_a, "only_b": only_b, "p_value": mcnemar_exact(only_a, only_b)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="first run is the baseline")
    p.add_argument("--modes", nargs="+", default=["retrieved", "oracle"])
    p.add_argument("--split", default="test")
    p.add_argument("--results-dir", default="results/qa")
    p.add_argument("--out", default="results/comparison.json")
    args = p.parse_args()
    results_dir = Path(args.results_dir)

    report: dict = {}
    rows = []
    for mode in args.modes:
        runs = {r: load_run(results_dir, r, mode, args.split) for r in args.runs}
        for r, df in runs.items():
            rows.append({"run": r, "mode": mode, "n": len(df),
                         "strict": df["strict"].mean() * 100, "lenient": df["lenient"].mean() * 100,
                         "format_ok": df["format_ok"].mean() * 100})
        base, others = args.runs[0], args.runs[1:]
        pairs = [(base, o) for o in others] + [(a, b) for i, a in enumerate(others) for b in others[i + 1:]]
        report[mode] = {}
        for a, b in pairs:
            res = paired(runs[a], runs[b])
            report[mode][f"{b} vs {a}"] = res

    table = pd.DataFrame(rows)
    print("\nAccuracy (%) on the test questions\n")
    print(table.pivot(index="run", columns="mode", values="strict").reindex(args.runs)
          .round(1).to_string())
    print("\nFull table\n")
    print(table.round(1).to_string(index=False))
    print("\nPaired comparisons (strict accuracy)\n")
    for mode, comps in report.items():
        for name, r in comps.items():
            lo, hi = r["ci95"]
            print(f"  [{mode:9}] {name:40} {r['diff'] * 100:+5.1f} pts  95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}]  "
                  f"only-new-right {r['only_b']:3d} / only-old-right {r['only_a']:3d}  p = {r['p_value']:.3g}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"accuracy": rows, "paired": report}, indent=2, default=list))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()