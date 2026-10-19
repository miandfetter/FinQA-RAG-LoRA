"""Tests for building LoRA training data: train-only retrieval and SFT example construction."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

import finqa_rag_lora.retrieve_train as rt
from finqa_rag_lora.build_sft import build_sft, trim_after_calc
from finqa_rag_lora.data import questions_for_split
from finqa_rag_lora.prompts import SYSTEM_PROMPT, build_prompt


# ---------- data ----------

def test_questions_for_split_excludes_empty_and_other_splits():
    df = pd.DataFrame({"id": ["a", "b", "c"], "split": ["train", "train", "test"],
                       "question": ["q?", "  ", "q?"], "program_answer": ["1", "2", "3"],
                       "context_id": ["p1", "p2", "p3"], "company_name": "C", "report_year": 2010})
    out = questions_for_split(df, "train")
    assert list(out["id"]) == ["a"] and list(out.columns[:4]) == ["id", "question", "answer", "gold_doc_id"]


# ---------- train retrieval ----------

def test_ensure_gold_keeps_order_when_present():
    rng = np.random.default_rng(0)
    assert rt.ensure_gold(["a", "g", "b", "c"], "g", 3, rng) == ["a", "g", "b"]


def test_ensure_gold_inserts_when_missing():
    rng = np.random.default_rng(0)
    out = rt.ensure_gold(["a", "b", "c", "d"], "g", 3, rng)
    assert "g" in out and len(out) == 3 and "c" not in out  # lowest-ranked result dropped


def test_train_pages_exclude_held_out_pages():
    train_q = pd.DataFrame({"gold_doc_id": ["p1", "p2", "shared"]})
    test_q = pd.DataFrame({"gold_doc_id": ["shared", "p9"]})
    assert rt.train_page_ids(train_q, [test_q]) == {"p1", "p2"}


def test_train_retrieval_never_uses_test_pages(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    ids = [f"train_p{i}" for i in range(8)] + [f"test_p{i}" for i in range(4)]
    rng = np.random.default_rng(1)
    vecs = rng.normal(size=(len(ids), 8)).astype(np.float32)
    np.save(data / "doc.npy", vecs)
    (data / "doc_ids.json").write_text(json.dumps(ids))
    pd.DataFrame({"doc_id": ids, "text": "t", "company_name": "C", "report_year": "2010"}).to_parquet(data / "corpus.parquet")
    train_q = pd.DataFrame({"id": [f"t{i}" for i in range(8)], "question": [f"q {i}" for i in range(8)],
                            "answer": 1.0, "gold_doc_id": ids[:8], "company_name": "C", "report_year": "2010"})
    train_q.to_json(data / "questions_train.jsonl", orient="records", lines=True)
    for split in ("test", "dev"):
        pd.DataFrame({"id": ["x"], "question": ["q"], "answer": [1.0], "gold_doc_id": ["test_p0"],
                      "company_name": "C", "report_year": "2010"}).to_json(data / f"eval_{split}.jsonl", orient="records", lines=True)
    cfg = {"model": "fake", "query_task_type": None, "output_dimensionality": None, "batch_size": 1,
           "max_retries": 0, "requests_per_minute": None, "corpus_path": str(data / "corpus.parquet"),
           "vectors_path": str(data / "doc.npy"), "ids_path": str(data / "doc_ids.json")}
    (tmp_path / "e.yaml").write_text(yaml.safe_dump(cfg))
    # Fake query embeddings that point exactly at a TEST page, to tempt the retriever.
    monkeypatch.setattr(rt, "make_gemini_embed_fn", lambda *a: (lambda texts: [vecs[8].tolist() for _ in texts]))

    rt.main(str(tmp_path / "e.yaml"), k=3, seed=0)

    rows = pd.read_json(tmp_path / "results" / "retrieval_train.jsonl", lines=True)
    assert len(rows) == 8
    for r in rows.itertuples():
        assert r.gold_doc_id in r.retrieved and len(r.retrieved) == 3
        assert not any(d.startswith("test_") for d in r.retrieved)


# ---------- SFT examples ----------

def test_trim_after_calc():
    assert trim_after_calc("Reasoning.\nCALC: 1 + 2\nHope this helps!") == "Reasoning.\nCALC: 1 + 2"
    assert trim_after_calc("CALC: 5 then\nCALC: 6\n") == "CALC: 5 then\nCALC: 6"


@pytest.fixture
def corpus():
    return pd.DataFrame({"doc_id": ["d1", "d2"], "text": ["page one", "page two"],
                         "company_name": ["A", "B"], "report_year": ["2010", "2011"]}).set_index("doc_id")


def test_build_sft_keeps_only_strict_and_well_formatted(corpus):
    preds = pd.DataFrame({
        "id": ["a", "b", "c", "d"], "question": ["qa?", "qb?", "qc?", "qd?"],
        "strict": [True, False, True, True], "format_ok": [True, True, False, True],
        "doc_ids": [["d1", "d2"], ["d1"], ["d2"], ["d2"]],
        "response": ["Think.\nCALC: 1 + 1\nThanks", "CALC: 9", "answer is 3", "CALC: 4"],
    })
    train, val = build_sft(preds, corpus, val_fraction=0.0, seed=0)
    assert [e["id"] for e in train] == ["a", "d"] and val == []
    msgs = train[0]["messages"]
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    docs = [{"text": "page one", "company_name": "A", "report_year": "2010"},
            {"text": "page two", "company_name": "B", "report_year": "2011"}]
    assert msgs[1]["content"] == build_prompt("qa?", docs)   # identical to the eval prompt
    assert msgs[2]["content"] == "Think.\nCALC: 1 + 1"


def test_validation_split_is_deterministic_and_disjoint(corpus):
    preds = pd.DataFrame({"id": [f"q{i}" for i in range(40)], "question": "q?", "strict": True,
                          "format_ok": True, "doc_ids": [["d1"]] * 40, "response": "CALC: 1"})
    t1, v1 = build_sft(preds, corpus, 0.1, seed=7)
    t2, v2 = build_sft(preds, corpus, 0.1, seed=7)
    assert [e["id"] for e in v1] == [e["id"] for e in v2] and len(v1) == 4
    assert not {e["id"] for e in t1} & {e["id"] for e in v1}


def test_teacher_run_on_train_split_feeds_build_sft(tmp_path, corpus):
    from finqa_rag_lora.eval_qa import main as run_qa
    from finqa_rag_lora.generate import Generation

    corpus.reset_index().assign(file_name="f").to_parquet(tmp_path / "corpus.parquet")
    pd.DataFrame({"id": ["a", "b"], "question": ["qa?", "qb?"], "answer": [2.0, 5.0],
                  "gold_doc_id": ["d1", "d2"], "company_name": "A", "report_year": "2010"}
                 ).to_json(tmp_path / "questions_train.jsonl", orient="records", lines=True)
    pd.DataFrame({"id": ["a", "b"], "retrieved": [["d1", "d2"], ["d2", "d1"]]}
                 ).to_json(tmp_path / "retrieval_train.jsonl", orient="records", lines=True)
    cfg = {"run_name": "teacher", "generator": "fake", "model": "m", "k": 2, "concurrency": 1,
           "requests_per_minute": None, "max_retries": 0, "corpus_path": str(tmp_path / "corpus.parquet"),
           "eval_dir": str(tmp_path), "retrieval_dir": str(tmp_path), "results_dir": str(tmp_path / "qa")}
    (tmp_path / "t.yaml").write_text(yaml.safe_dump(cfg))
    fake = lambda prompt: Generation(text="Reasoning\nCALC: 1 + 1")   # right for "a" (2), wrong for "b" (5)

    m = run_qa(str(tmp_path / "t.yaml"), "retrieved", "train", None, False, generator_factory=lambda c: fake)
    assert m["n"] == 2 and m["accuracy_strict"] == 0.5

    preds = pd.read_json(tmp_path / "qa" / "teacher_retrieved_train" / "predictions.jsonl", lines=True)
    train, _ = build_sft(preds, corpus, 0.0, 0)
    assert [e["id"] for e in train] == ["a"]