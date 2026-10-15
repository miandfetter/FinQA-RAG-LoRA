"""End-to-end tests of the QA runner with a fake model: no API calls."""

import json

import pandas as pd
import pytest
import yaml

from finqa_rag_lora.eval_qa import main
from finqa_rag_lora.generate import Generation, RateLimiter, call_with_retry


class FakeModel:
    """Answers correctly when the gold page is in the prompt, otherwise answers 0."""

    def __init__(self, answers: dict[str, float], fail_on: set[str] = frozenset()):
        self.answers, self.fail_on, self.calls = answers, set(fail_on), 0

    def __call__(self, prompt: str) -> Generation:
        self.calls += 1
        for marker, value in self.answers.items():
            if marker in prompt:
                if marker in self.fail_on:
                    err = Exception("bad request")
                    err.code = 400
                    raise err
                return Generation(text=f"Reasoning...\nCALC: {value}", usage={"prompt_tokens": 10})
        return Generation(text="CALC: 0")


@pytest.fixture
def setup(tmp_path):
    corpus = pd.DataFrame({
        "doc_id": [f"d{i}" for i in range(6)],
        "text": [f"page-{i}" for i in range(6)],
        "company_name": "Co", "report_year": "2010", "file_name": "f",
    })
    corpus.to_parquet(tmp_path / "corpus.parquet")
    questions = pd.DataFrame({
        "id": ["q0", "q1", "q2"], "question": ["a?", "b?", "c?"],
        "answer": [5.0, 7.0, 9.0], "gold_doc_id": ["d0", "d1", "d2"],
    })
    questions.to_json(tmp_path / "eval_test.jsonl", orient="records", lines=True)
    # Retrieval found the gold page for q0 and q1 (in the top 3) but not for q2.
    retrieval = pd.DataFrame({"id": ["q0", "q1", "q2"],
                              "retrieved": [["d0", "d3", "d4"], ["d5", "d1", "d3"], ["d3", "d4", "d5"]]})
    retrieval.to_json(tmp_path / "retrieval_test.jsonl", orient="records", lines=True)
    cfg = {"run_name": "fake", "generator": "fake", "model": "m", "temperature": 0, "k": 3,
           "requests_per_minute": None, "concurrency": 2, "max_retries": 0,
           "corpus_path": str(tmp_path / "corpus.parquet"), "eval_dir": str(tmp_path),
           "retrieval_dir": str(tmp_path), "results_dir": str(tmp_path / "qa")}
    cfg_path = tmp_path / "qa.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    answers = {"page-0": 5.0, "page-1": 7.0, "page-2": 9.0}
    return tmp_path, cfg_path, answers


def run(cfg_path, model, mode="retrieved", overwrite=False):
    return main(str(cfg_path), mode, "test", None, overwrite, generator_factory=lambda cfg: model)


def test_retrieved_mode(setup):
    _, cfg_path, answers = setup
    m = run(cfg_path, FakeModel(answers))
    assert m["n"] == 3
    assert m["accuracy_strict"] == pytest.approx(2 / 3, abs=1e-3)  # q2's gold page wasn't retrieved
    assert m["gold_in_context_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert m["accuracy_strict_gold_in_context"] == 1.0


def test_oracle_mode(setup):
    _, cfg_path, answers = setup
    assert run(cfg_path, FakeModel(answers), mode="oracle")["accuracy_strict"] == 1.0


def test_resume_makes_no_new_calls(setup):
    _, cfg_path, answers = setup
    run(cfg_path, FakeModel(answers))
    second = FakeModel(answers)
    run(cfg_path, second)
    assert second.calls == 0


def test_changed_settings_refused(setup):
    tmp_path, cfg_path, answers = setup
    run(cfg_path, FakeModel(answers))
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["k"] = 2
    cfg_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="different settings"):
        run(cfg_path, FakeModel(answers))
    assert run(cfg_path, FakeModel(answers), overwrite=True)["n"] == 3


def test_failed_question_is_retried_on_rerun(setup):
    tmp_path, cfg_path, answers = setup
    assert run(cfg_path, FakeModel(answers, fail_on={"page-1"})) is None   # incomplete: no metrics
    assert not (tmp_path / "qa" / "fake_retrieved_test" / "metrics.json").exists()
    second = FakeModel(answers)
    m = run(cfg_path, second)
    assert second.calls == 1 and m["n"] == 3


def test_predictions_file_has_details(setup):
    tmp_path, cfg_path, answers = setup
    run(cfg_path, FakeModel(answers))
    rows = [json.loads(l) for l in (tmp_path / "qa" / "fake_retrieved_test" / "predictions.jsonl").open()]
    row = next(r for r in rows if r["id"] == "q0")
    assert row["expression"] == "5.0" and row["doc_ids"] == ["d0", "d3", "d4"] and row["prompt_tokens"] == 10


def test_call_with_retry_retries_rate_limits():
    attempts = []

    def flaky(prompt):
        attempts.append(prompt)
        if len(attempts) < 3:
            err = Exception("slow down")
            err.code = 429
            raise err
        return "ok"

    assert call_with_retry(flaky, "p", sleep=lambda s: None) == "ok" and len(attempts) == 3


def test_rate_limiter_spacing():
    now, waits = [0.0], []

    def sleep(s):
        waits.append(s)

    limiter = RateLimiter(per_minute=30, clock=lambda: now[0], sleep=sleep)
    for _ in range(3):
        limiter.acquire()
    assert waits == [2.0, 4.0]  # calls scheduled at t=0, 2, 4