"""End-to-end tests of the local-model path with a TINY random Qwen3-architecture model.

No download and no GPU: the model is built from a config, and a small tokenizer is trained on
the fly. Its answers are gibberish; the point is to exercise the real code (chat template,
batched generation, scoring, the GPU job's copy-in/copy-out) before spending GPU time.
Skipped automatically if torch/transformers aren't installed (they're GPU-only dependencies).
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

def test_generate_batch_shapes_and_usage(tiny_model_dir):
    from finqa_rag_lora.generate import HFGenerator

    gen = HFGenerator(str(tiny_model_dir), max_new_tokens=8, dtype="float32")
    assert gen.render("hi").endswith("<|im_start|>assistant\n")
    outs = gen.generate_batch(["short question", "a much longer question " * 20])
    assert len(outs) == 2
    for o in outs:
        assert isinstance(o.text, str) and o.finish_reason in {"STOP", "MAX_TOKENS"}
        assert 0 < o.usage["output_tokens"] <= 8 and o.usage["prompt_tokens"] > 0
    assert outs[1].usage["prompt_tokens"] > outs[0].usage["prompt_tokens"]  # padding not counted


def test_batched_matches_single(tiny_model_dir):
    """Left padding must not change greedy outputs: the same prompt alone or in a batch gives the same text."""
    from finqa_rag_lora.generate import HFGenerator

    gen = HFGenerator(str(tiny_model_dir), max_new_tokens=6, dtype="float32")
    prompts = ["revenue 2016", "revenue was 705.4 in 2016 and 696.3 in 2015, what changed?"]
    batched = [o.text for o in gen.generate_batch(prompts)]
    single = [gen.generate_batch([p])[0].text for p in prompts]
    assert batched == single


def _write_eval_inputs(root: Path) -> None:
    (root / "data").mkdir(parents=True)
    (root / "results").mkdir(parents=True)
    pd.DataFrame({"doc_id": ["d0", "d1", "d2"], "text": ["page zero", "page one", "page two"],
                  "company_name": "Co", "report_year": "2016", "file_name": "f"}
                 ).to_parquet(root / "data/corpus.parquet")
    q = pd.DataFrame({"id": [f"q{i}" for i in range(5)], "question": [f"question {i}?" for i in range(5)],
                      "answer": [1.0] * 5, "gold_doc_id": ["d0", "d1", "d2", "d0", "d1"],
                      "company_name": "Co", "report_year": "2016"})
    q.to_json(root / "data/eval_test.jsonl", orient="records", lines=True)
    pd.DataFrame({"id": q["id"], "retrieved": [["d0", "d1", "d2"]] * 5}
                 ).to_json(root / "results/retrieval_test.jsonl", orient="records", lines=True)


def test_gpu_entry_eval_end_to_end(tiny_model_dir, tmp_path):
    """The GPU job: copy inputs from the 'bucket', run both modes in batches, copy results back."""
    from finqa_rag_lora.gpu_entry import run_eval

    bucket = tmp_path / "bucket"
    _write_eval_inputs(bucket)
    cfg = {"run_name": "tiny", "generator": "hf", "model": str(tiny_model_dir), "adapter": None,
           "load_in_4bit": False, "dtype": "float32", "max_new_tokens": 8, "temperature": 0,
           "batch_size": 2, "k": 3, "require_gpu": False,
           "corpus_path": "x", "eval_dir": "x", "retrieval_dir": "x", "results_dir": "x"}
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    run_eval(str(cfg_path), ["retrieved", "oracle"], "test", None, bucket)

    for mode in ["retrieved", "oracle"]:
        run_dir = bucket / "results/qa" / f"tiny_{mode}_test"
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["n"] == 5
        rows = [json.loads(l) for l in (run_dir / "predictions.jsonl").open()]
        assert sorted(r["id"] for r in rows) == [f"q{i}" for i in range(5)]
        settings = json.loads((run_dir / "settings.json").read_text())
        assert settings["generator"] == "hf" and settings["dtype"] == "float32"
    oracle_rows = [json.loads(l) for l in (bucket / "results/qa/tiny_oracle_test/predictions.jsonl").open()]
    assert all(len(r["doc_ids"]) == 1 for r in oracle_rows)

    # Running the job again resumes from the bucket: nothing is regenerated.
    before = (bucket / "results/qa/tiny_retrieved_test/predictions.jsonl").read_text()
    run_eval(str(cfg_path), ["retrieved"], "test", None, bucket)
    assert (bucket / "results/qa/tiny_retrieved_test/predictions.jsonl").read_text() == before


def _tiny_cfg(tiny_model_dir, **overrides) -> dict:
    cfg = {"run_name": "tiny", "generator": "hf", "model": str(tiny_model_dir), "dtype": "float32",
           "max_new_tokens": 4, "temperature": 0, "batch_size": 1, "k": 3,
           "corpus_path": "x", "eval_dir": "x", "retrieval_dir": "x", "results_dir": "x"}
    cfg.update(overrides)
    return cfg


def test_job_refuses_to_run_on_cpu_by_default(tiny_model_dir, tmp_path):
    """The bug that cost hours: on a GPU machine with no visible GPU, the job must stop, not crawl on the CPU."""
    from finqa_rag_lora.gpu_entry import run_eval

    if torch.cuda.is_available():
        pytest.skip("this machine has a GPU")
    bucket = tmp_path / "bucket"
    _write_eval_inputs(bucket)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(_tiny_cfg(tiny_model_dir)))   # require_gpu not set -> defaults to True
    with pytest.raises(RuntimeError, match="No GPU visible"):
        run_eval(str(cfg_path), ["retrieved"], "test", None, bucket)
    assert not (bucket / "results/qa").exists()   # stopped before doing any work


def test_job_rejects_wrong_config_file(tmp_path):
    """E.g. the job file saved where the model config should be."""
    from finqa_rag_lora.gpu_entry import run_eval

    bucket = tmp_path / "bucket"
    _write_eval_inputs(bucket)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"workerPoolSpecs": [{"replicaCount": 1}]}))
    with pytest.raises(ValueError, match="missing required settings"):
        run_eval(str(cfg_path), ["retrieved"], "test", None, bucket)


def test_results_reach_the_bucket_during_the_run(tiny_model_dir, tmp_path, monkeypatch):
    """Finished answers are copied to the bucket every few batches, not only at the end."""
    from finqa_rag_lora import eval_qa, gpu_entry

    bucket = tmp_path / "bucket"
    _write_eval_inputs(bucket)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(_tiny_cfg(tiny_model_dir, require_gpu=False)))
    remote = bucket / "results/qa/tiny_retrieved_test/predictions.jsonl"
    seen = []

    original = eval_qa.run_batched

    def spy(*args, **kwargs):
        cp = kwargs["checkpoint"]

        def wrapped():
            cp()
            seen.append(sum(1 for _ in remote.open()) if remote.exists() else 0)
        kwargs["checkpoint"], kwargs["checkpoint_every"] = wrapped, 2
        return original(*args, **kwargs)

    monkeypatch.setattr(eval_qa, "run_batched", spy)
    gpu_entry.run_eval(str(cfg_path), ["retrieved"], "test", None, bucket)
    assert seen[:2] == [2, 4]   # 5 questions, batch 1, checkpoint every 2 batches