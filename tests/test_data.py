"""Tests for src/data.py. They use a small fake dataset, so no download is needed."""

import pandas as pd
import pytest

from finqa_rag_lora.data import build_corpus, sample_eval_set


@pytest.fixture
def fake_df() -> pd.DataFrame:
    rows = []
    for split, n_docs in [("train", 6), ("dev", 3), ("test", 4)]:
        for d in range(n_docs):
            for q in range(2):  # two questions per page, like the real data
                rows.append({
                    "id": f"{split}_{d}_{q}",
                    "context_id": f"{split}_ctx_{d}",
                    "split": split,
                    "question": f"question {q} about {split} page {d}",
                    "program_answer": str(d + q * 0.5),
                    "context": f"page text for {split} {d}",
                    "company_name": f"Company {d % 2}",
                    "report_year": 2010 + d,
                    "file_name": f"pdf/{split}/{d}.pdf",
                })
    return pd.DataFrame(rows)


def test_corpus_has_one_row_per_page(fake_df):
    corpus = build_corpus(fake_df)
    assert corpus["doc_id"].is_unique
    assert len(corpus) == fake_df["context_id"].nunique()


def test_corpus_includes_all_splits(fake_df):
    corpus = build_corpus(fake_df)
    assert set(corpus["doc_id"]) == set(fake_df["context_id"])


def test_corpus_rejects_inconsistent_text(fake_df):
    fake_df.loc[0, "context"] = "different text for the same page"
    with pytest.raises(AssertionError):
        build_corpus(fake_df)


def test_eval_gold_docs_exist_in_corpus(fake_df):
    corpus_ids = set(build_corpus(fake_df)["doc_id"])
    sample = sample_eval_set(fake_df, "test", n=5, seed=0)
    assert set(sample["gold_doc_id"]) <= corpus_ids


@pytest.mark.parametrize("split", ["test", "dev"])
def test_sample_only_contains_its_split(fake_df, split):
    sample = sample_eval_set(fake_df, split, n=4, seed=0)
    allowed = set(fake_df.loc[fake_df["split"] == split, "id"])
    assert set(sample["id"]) <= allowed


def test_sample_size(fake_df):
    assert len(sample_eval_set(fake_df, "test", n=5, seed=0)) == 5


def test_sample_is_deterministic(fake_df):
    a = sample_eval_set(fake_df, "test", n=5, seed=42)
    b = sample_eval_set(fake_df, "test", n=5, seed=42)
    assert list(a["id"]) == list(b["id"])


def test_sample_skips_empty_questions(fake_df):
    test_ids = fake_df.index[fake_df["split"] == "test"]
    fake_df.loc[test_ids[:3], "question"] = ["", "   ", None]
    sample = sample_eval_set(fake_df, "test", n=5, seed=0)  # 8 test questions, 3 empty -> 5 usable
    assert (sample["question"].str.strip() != "").all()


def test_sample_too_large_raises(fake_df):
    with pytest.raises(ValueError):
        sample_eval_set(fake_df, "test", n=1000, seed=0)