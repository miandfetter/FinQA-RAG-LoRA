"""Shared test fixtures."""

from pathlib import Path

import pytest

CHATML = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> Path:
    """A TINY random model with Qwen3's architecture plus a small trained tokenizer: no download."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    out = tmp_path_factory.mktemp("tiny_qwen")
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    corpus = ["Documents: Company revenue was 705.4 in 2016 and 696.3 in 2015. CALC: (1 + 2) / 3"] * 50
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=300, special_tokens=special,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    from tokenizers import decoders
    tok.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    fast.chat_template = CHATML
    fast.save_pretrained(out)

    cfg = Qwen3Config(vocab_size=len(fast), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=4096,
                      eos_token_id=fast.eos_token_id, pad_token_id=fast.pad_token_id)
    torch.manual_seed(0)
    Qwen3ForCausalLM(cfg).save_pretrained(out)
    return out