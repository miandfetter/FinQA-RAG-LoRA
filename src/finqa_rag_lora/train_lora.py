"""LoRA fine-tuning of the small model on the distilled training examples.

Each example is a conversation: system prompt + user prompt (pages + question) -> the teacher's
correct reasoning ending in a CALC line. The loss is computed ONLY on the reply, so the model
learns to answer, not to reproduce documents.

Run as a GPU job through gpu_entry (`train` task); can also be called directly for testing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import yaml


def load_examples(path: Path) -> list[dict]:
    """Chat-format examples from build_sft -> TRL's prompt/completion format."""
    out = []
    with open(path) as f:
        for line in f:
            msgs = json.loads(line)["messages"]
            assert [m["role"] for m in msgs] == ["system", "user", "assistant"], "unexpected example format"
            out.append({"prompt": msgs[:2], "completion": msgs[2:]})
    return out


def drop_too_long(examples: list[dict], tokenizer, max_length: int) -> tuple[list[dict], int]:
    """Remove examples that wouldn't fit: truncating them would cut off the answer we train on."""
    kept = []
    for ex in examples:
        n = len(tokenizer.apply_chat_template(ex["prompt"] + ex["completion"], tokenize=True))
        if n <= max_length:
            kept.append(ex)
    return kept, len(examples) - len(kept)


def train(cfg: dict, train_path: Path, val_path: Path, output_dir: Path,
          on_save: Callable[[], None] | None = None) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = cfg["max_length"]
    train_ex, dropped_train = drop_too_long(load_examples(train_path), tokenizer, max_length)
    val_ex, dropped_val = drop_too_long(load_examples(val_path), tokenizer, max_length)
    if cfg.get("max_train_examples"):
        train_ex = train_ex[: cfg["max_train_examples"]]
    print(f"train examples: {len(train_ex)} (dropped {dropped_train} longer than {max_length} tokens)")
    print(f"val examples:   {len(val_ex)} (dropped {dropped_val})", flush=True)

    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    model_kwargs = {"torch_dtype": dtype}
    if cfg.get("load_in_4bit"):  # QLoRA: frozen base weights in 4-bit (for 16 GB GPUs)
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], **model_kwargs)

    lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      target_modules=cfg["target_modules"], task_type="CAUSAL_LM")

    use_bf16 = dtype == torch.bfloat16 and torch.cuda.is_available()
    use_fp16 = dtype == torch.float16 and torch.cuda.is_available()
    args = SFTConfig(
        output_dir=str(output_dir),
        max_length=max_length,                 # default is 1024: far too short for ~3.5k-token prompts
        completion_only_loss=True,             # loss on the reply only
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=0.0,
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16, fp16=use_fp16,
        logging_steps=cfg.get("logging_steps", 10),
        eval_strategy="steps", eval_steps=cfg.get("eval_steps", 50),
        save_strategy="steps", save_steps=cfg.get("save_steps", 50), save_total_limit=2,
        report_to="none",
        seed=cfg.get("seed", 42),
        use_cpu=not torch.cuda.is_available(),
    )

    class SaveToBucket(TrainerCallback):
        def on_save(self, *_args, **_kwargs):
            if on_save:
                on_save()

    trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(train_ex),
                         eval_dataset=Dataset.from_list(val_ex), processing_class=tokenizer,
                         peft_config=lora, callbacks=[SaveToBucket()])
    trainable, total = trainer.model.get_nb_trainable_parameters()
    print(f"trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)", flush=True)

    has_checkpoint = any(output_dir.glob("checkpoint-*"))
    result = trainer.train(resume_from_checkpoint=True if has_checkpoint else None)

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))   # only the small LoRA weights + config
    tokenizer.save_pretrained(str(adapter_dir))
    final_eval = trainer.evaluate()
    summary = {
        "train_examples": len(train_ex), "val_examples": len(val_ex),
        "dropped_too_long": dropped_train + dropped_val,
        "trainable_params": trainable, "total_params": total,
        "train_loss": result.training_loss, "eval_loss": final_eval.get("eval_loss"),
        "train_runtime_s": result.metrics.get("train_runtime"),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "train_config.yaml").write_text(yaml.safe_dump(cfg))
    (output_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary