"""Tests for embed.py. A fake embed function stands in for the Gemini API: no cost, no network."""

import numpy as np
import pandas as pd
import pytest

from finqa_rag_lora.embed import embed_corpus, embed_texts, embed_with_retry


class FakeEmbedder:
    """Returns a vector derived from each text's length, and records every call."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0, 0.0] for t in texts]


class FakeAPIError(Exception):
    def __init__(self, code):
        self.code = code


def no_sleep(_seconds):
    pass


@pytest.fixture
def corpus():
    return pd.DataFrame({"doc_id": [f"ctx_{i}" for i in range(10)],
                         "text": ["x" * (i + 1) for i in range(10)]})


def test_batches_and_order(corpus):
    fake = FakeEmbedder()
    vectors = embed_texts(corpus["text"].tolist(), fake, batch_size=4, sleep=no_sleep)
    assert [len(c) for c in fake.calls] == [4, 4, 2]
    assert vectors.shape == (10, 3)
    assert list(vectors[:, 0]) == [float(i + 1) for i in range(10)]  # rows stay in input order


def test_retries_rate_limit_then_succeeds():
    fake = FakeEmbedder()
    failures = iter([FakeAPIError(429), FakeAPIError(503)])

    def flaky(texts):
        err = next(failures, None)
        if err:
            raise err
        return fake(texts)

    assert embed_with_retry(flaky, ["a"], sleep=no_sleep) == [[1.0, 1.0, 0.0]]


def test_does_not_retry_bad_request():
    calls = []

    def bad(texts):
        calls.append(1)
        raise FakeAPIError(400)

    with pytest.raises(FakeAPIError):
        embed_with_retry(bad, ["a"], sleep=no_sleep)
    assert len(calls) == 1


def test_gives_up_after_max_retries():
    def always_429(texts):
        raise FakeAPIError(429)

    with pytest.raises(FakeAPIError):
        embed_with_retry(always_429, ["a"], max_retries=2, sleep=no_sleep)


def test_wrong_vector_count_raises():
    with pytest.raises(RuntimeError, match="batch_size: 1"):
        embed_texts(["a", "b"], lambda texts: [[1.0]], batch_size=2, sleep=no_sleep)


def test_cache_is_reused(corpus, tmp_path):
    vec_path, ids_path = tmp_path / "v.npy", tmp_path / "ids.json"
    fake = FakeEmbedder()
    first = embed_corpus(corpus, fake, vec_path, ids_path, batch_size=4, sleep=no_sleep)
    calls_after_first = len(fake.calls)
    second = embed_corpus(corpus, fake, vec_path, ids_path, batch_size=4, sleep=no_sleep)
    assert len(fake.calls) == calls_after_first  # no new API calls
    np.testing.assert_array_equal(first, second)


def test_empty_text_rejected_before_any_api_call():
    fake = FakeEmbedder()
    with pytest.raises(ValueError, match="empty"):
        embed_texts(["ok", "  ", "fine"], fake, batch_size=1, sleep=no_sleep)
    assert fake.calls == []


def test_pacing_waits_between_requests():
    fake, waits, now = FakeEmbedder(), [], [0.0]

    def fake_sleep(s):
        waits.append(s)
        now[0] += s

    embed_texts(["a", "b", "c"], fake, batch_size=1, sleep=fake_sleep,
                requests_per_minute=30, clock=lambda: now[0])
    assert waits == [2.0, 2.0]  # 30/min = one request every 2s; no wait before the first


def test_resumes_after_crash(corpus, tmp_path):
    vec_path, ids_path = tmp_path / "v.npy", tmp_path / "ids.json"
    fake = FakeEmbedder()

    def crash_on_third_batch(texts):
        if len(fake.calls) == 2:
            raise FakeAPIError(400)  # not retryable: the run stops here
        return fake(texts)

    with pytest.raises(FakeAPIError):
        embed_corpus(corpus, crash_on_third_batch, vec_path, ids_path, batch_size=4, sleep=no_sleep)
    assert not vec_path.exists()

    fake2 = FakeEmbedder()
    vectors = embed_corpus(corpus, fake2, vec_path, ids_path, batch_size=4, sleep=no_sleep)
    assert [len(c) for c in fake2.calls] == [2]  # only the last 2 pages were embedded again
    assert list(vectors[:, 0]) == [float(i + 1) for i in range(10)]
    assert not (tmp_path / "v_progress").exists()  # cleaned up after success


def test_stale_progress_is_discarded_and_run_completes(corpus, tmp_path):
    """Progress left over from a DIFFERENT text set (e.g. an old question sample) must not break a new run."""
    vec_path, ids_path = tmp_path / "v.npy", tmp_path / "ids.json"
    fake = FakeEmbedder()

    def crash_on_second_batch(texts):
        if len(fake.calls) == 1:
            raise FakeAPIError(400)
        return fake(texts)

    with pytest.raises(FakeAPIError):
        embed_corpus(corpus, crash_on_second_batch, vec_path, ids_path, batch_size=4, sleep=no_sleep)

    other = pd.DataFrame({"doc_id": [f"new_{i}" for i in range(6)], "text": ["y" * (i + 1) for i in range(6)]})
    vectors = embed_corpus(other, FakeEmbedder(), vec_path, ids_path, batch_size=4, sleep=no_sleep)
    assert vectors.shape == (6, 3)


def test_cache_invalidated_when_corpus_changes(corpus, tmp_path):
    vec_path, ids_path = tmp_path / "v.npy", tmp_path / "ids.json"
    fake = FakeEmbedder()
    embed_corpus(corpus, fake, vec_path, ids_path, batch_size=4, sleep=no_sleep)
    calls_after_first = len(fake.calls)
    embed_corpus(corpus.iloc[::-1], fake, vec_path, ids_path, batch_size=4, sleep=no_sleep)
    assert len(fake.calls) > calls_after_first  # different order -> re-embedded