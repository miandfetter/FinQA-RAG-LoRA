import json

import pandas as pd
import pytest

from finqa_rag_lora.compare import load_run, mcnemar_exact, paired


def _write(results_dir, run, strict):
    d = results_dir / f"{run}_retrieved_test"
    d.mkdir(parents=True)
    with open(d / "predictions.jsonl", "w") as f:
        for i, s in enumerate(strict):
            f.write(json.dumps({"id": f"q{i}", "strict": s, "lenient": s, "format_ok": True}) + "\n")


def test_mcnemar_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)      # all 10 disagreements favour one model


def test_paired_counts_and_diff(tmp_path):
    _write(tmp_path, "a", [True, True, False, False, True])
    _write(tmp_path, "b", [True, False, True, True, True])
    r = paired(load_run(tmp_path, "a", "retrieved", "test"), load_run(tmp_path, "b", "retrieved", "test"))
    assert r["only_a"] == 1 and r["only_b"] == 2 and r["diff"] == pytest.approx(0.2)
    lo, hi = r["ci95"]
    assert lo <= r["diff"] <= hi


def test_different_question_sets_rejected(tmp_path):
    _write(tmp_path, "a", [True] * 5)
    _write(tmp_path, "b", [True] * 4)
    with pytest.raises(ValueError, match="different questions"):
        paired(load_run(tmp_path, "a", "retrieved", "test"), load_run(tmp_path, "b", "retrieved", "test"))