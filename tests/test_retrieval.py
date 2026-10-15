"""Tests for retrieval and its metrics, using tiny hand-made vectors."""

import numpy as np
import pytest

from finqa_rag_lora.eval_retrieval import classify_miss, gold_rank, mrr, recall_at_k
from finqa_rag_lora.retrieve import DenseRetriever, normalize


@pytest.fixture
def retriever():
    # Three pages pointing in clearly different directions.
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    return DenseRetriever(vectors, ["a", "b", "c"])


def test_nearest_page_comes_first(retriever):
    ids, scores = retriever.search(np.array([[0.1, 0.9, 0.2]]), k=3)
    assert ids[0][0] == "b"
    assert scores[0][0] >= scores[0][1] >= scores[0][2]  # best first


def test_length_does_not_matter(retriever):
    # Cosine similarity ignores vector length: a scaled query gives the same ranking.
    short, _ = retriever.search(np.array([[0.0, 0.0, 1.0]]), k=3)
    long, _ = retriever.search(np.array([[0.0, 0.0, 50.0]]), k=3)
    assert short == long


def test_k_larger_than_corpus(retriever):
    ids, _ = retriever.search(np.array([[1.0, 0.0, 0.0]]), k=10)
    assert len(ids[0]) == 3


def test_mismatched_ids_rejected():
    with pytest.raises(ValueError):
        DenseRetriever(np.eye(3, dtype=np.float32), ["a", "b"])


def test_normalize_gives_unit_length():
    out = normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0])


def test_gold_rank():
    assert gold_rank(["x", "gold", "y"], "gold") == 2
    assert gold_rank(["x", "y"], "gold") is None


def test_recall_and_mrr():
    ranks = [1, 3, None, 2]           # four questions
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 3) == 0.75
    assert mrr(ranks) == pytest.approx((1 + 1/3 + 0 + 1/2) / 4)


def test_classify_miss():
    gold = {"company_name": "Apple", "report_year": "2002"}
    assert classify_miss({"company_name": "IBM", "report_year": "2002"}, gold) == "wrong company"
    assert classify_miss({"company_name": "Apple", "report_year": "2003"}, gold) == "right company, wrong year"
    assert classify_miss({"company_name": "Apple", "report_year": "2002"}, gold) == "right company and year, wrong page"