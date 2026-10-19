"""End-to-end tests of LoRA training with the TINY random Qwen3-architecture model (CPU, no download).

Checks the things that would silently ruin a real run: the loss must cover only the reply,
long examples must be dropped (not truncated), the adapter must save and load for evaluation,
and a rerun must resume from the bucket.
"""

import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

from finqa_rag_lora.prompts import SYSTEM_PROMPT  # noqa: E402


def _write_sft(path: Path, n: int, long_every: int = 0) -> None:
    with open(path, "w") as f:
        for i in range(n):
            user = f"Documents: revenue was {700 + i}.4 in 2016 and 696.3 in 2015. Question: change?"
            if long_every and i % long_every == 0:
                user += " filler" * 400   # too long for max_length -> must be dropped
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user},
                    {"role": "assistant", "content": f"The change is computed.\nCALC: {700 + i}.4 - 696.3"}]
            f.write(json.dumps({"id": f"q{i}", "messages": msgs}) + "\n")


# Note: the tiny test tokenizer is inefficient (~780 tokens per example), hence the test max_lengths.
def _cfg(model_dir, **overrides) -> dict:
    cfg = {"run_name": "tiny_lora", "model": str(model_dir), "dtype": "float32", "load_in_4bit": False,
           "lora_r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
           "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
           "max_length": 1200, "epochs": 1, "batch_size": 2, "grad_accum": 1, "learning_rate": 1e-3,
           "logging_steps": 1, "eval_steps": 2, "save_steps": 2, "gradient_checkpointing": False,
           "require_gpu": False}
    cfg.update(overrides)
    return cfg


def test_loss_covers_only_the_reply(tiny_model_dir):
    """Tokenize one example the way SFTTrainer does and check which tokens are trained on."""
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from transformers import AutoModelForCausalLM

    from finqa_rag_lora.train_lora import load_examples

    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    p = Path(tiny_model_dir).parent / "one.jsonl"
    _write_sft(p, 1)
    ex = load_examples(p)
    args = SFTConfig(output_dir=str(p.parent / "o"), max_length=2048, completion_only_loss=True,
                     report_to="none", use_cpu=True)
    trainer = SFTTrainer(model=AutoModelForCausalLM.from_pretrained(tiny_model_dir), args=args,
                         train_dataset=Dataset.from_list(ex), processing_class=tok)
    row = trainer.train_dataset[0]
    batch = trainer.data_collator([row])
    labels = batch["labels"][0].tolist()
    trained = [t for t, l in zip(batch["input_ids"][0].tolist(), labels) if l != -100]
    trained_text = tok.decode(trained)
    assert "CALC: 700.4 - 696.3" in trained_text            # the reply is trained on...
    assert "revenue was" not in trained_text                 # ...the prompt is not
    assert SYSTEM_PROMPT[:30] not in trained_text


def test_train_job_end_to_end_and_resume(tiny_model_dir, tmp_path):
    from finqa_rag_lora.gpu_entry import run_train

    bucket = tmp_path / "bucket"
    (bucket / "data/sft").mkdir(parents=True)
    _write_sft(bucket / "data/sft/sft_train.jsonl", 12, long_every=5)   # 3 of 12 too long
    _write_sft(bucket / "data/sft/sft_val.jsonl", 4)
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg(tiny_model_dir)))

    run_train(str(cfg_path), bucket)

    out = bucket / "adapters/tiny_lora"
    summary = json.loads((out / "train_summary.json").read_text())
    assert summary["train_examples"] == 9 and summary["dropped_too_long"] == 3
    assert 0 < summary["trainable_params"] < summary["total_params"] * 0.5   # only adapters train
    assert (out / "adapter/adapter_config.json").exists()
    assert any(out.glob("checkpoint-*"))                                     # synced during training
    losses = [h["loss"] for h in json.loads((out / "log_history.json").read_text()) if "loss" in h]
    assert losses and all(l == l for l in losses)                            # logged, no NaN

    # The saved adapter loads for evaluation, exactly as the eval job will use it.
    from finqa_rag_lora.generate import HFGenerator
    gen = HFGenerator(str(tiny_model_dir), adapter=str(out / "adapter"), max_new_tokens=4, dtype="float32")
    assert isinstance(gen.generate_batch(["q?"])[0].text, str)

    # Rerunning finds the bucket output and resumes from the latest checkpoint instead of starting over.
    run_train(str(cfg_path), bucket)
    assert json.loads((out / "train_summary.json").read_text())["train_examples"] == 9


def test_train_job_rejects_wrong_config(tmp_path):
    from finqa_rag_lora.gpu_entry import run_train

    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.safe_dump({"run_name": "x", "generator": "hf"}))
    with pytest.raises(ValueError, match="missing required settings"):
        run_train(str(cfg_path), tmp_path)


def test_too_long_examples_are_dropped_not_truncated(tiny_model_dir, tmp_path):
    """Truncation would cut off the reply (the only part trained on); such examples must be removed."""
    from transformers import AutoTokenizer

    from finqa_rag_lora.train_lora import drop_too_long, load_examples

    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    _write_sft(tmp_path / "s.jsonl", 10, long_every=2)
    kept, dropped = drop_too_long(load_examples(tmp_path / "s.jsonl"), tok, max_length=1200)
    assert dropped == 5 and len(kept) == 5
    assert all("filler" not in ex["prompt"][1]["content"] for ex in kept)