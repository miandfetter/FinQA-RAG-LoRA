import pytest

from finqa_rag_lora.prompts import build_prompt, parse_answer
from finqa_rag_lora.scoring import score


def test_prompt_lists_docs_in_order_then_question():
    docs = [{"text": "first page", "company_name": "A", "report_year": "2009"},
            {"text": "second page", "company_name": "B", "report_year": "2010"}]
    p = build_prompt("What was revenue?", docs)
    assert p.index("[Document 1] Company: A") < p.index("[Document 2] Company: B") < p.index("Question: What was revenue?")


def test_parse_uses_last_calc_line():
    text = "First guess CALC: 1 + 1 was wrong.\nNet change is 9.1\nCALC: 705.4 - 696.3"
    a = parse_answer(text)
    assert a.format_ok and a.value == pytest.approx(9.1) and a.expression == "705.4 - 696.3"


def test_parse_tolerates_markdown_formatting():
    assert parse_answer("**CALC:** (10 - 5) / 5").value == pytest.approx(1.0)


def test_parse_falls_back_to_last_number_without_calc():
    a = parse_answer("The answer is 1,234.5")
    assert a.value == 1234.5 and not a.format_ok


def test_parse_bad_expression():
    a = parse_answer("CALC: revenue - costs")
    assert a.value is None and not a.format_ok and a.error


def test_parse_empty_response():
    a = parse_answer("")
    assert a.value is None and not a.format_ok


@pytest.mark.parametrize("pred, gold, strict, lenient, sign_flip", [
    (9.1, 9.100000000000023, True, True, False),     # float noise in labels
    (0.1505, 0.15, True, True, False),               # within 1%
    (0.16, 0.15, False, False, False),
    (15.0, 0.15, False, True, False),                # percent vs fraction
    (0.15, 15.0, False, True, False),
    (-0.1657, 0.1657, False, False, True),           # sign only
    (16.57, -0.1657, False, False, True),
    (0.0, 0.0, True, True, False),                   # zero label
    (0.5, 0.0, False, False, False),
    (None, 3.0, False, False, False),                # unreadable answer
])
def test_score(pred, gold, strict, lenient, sign_flip):
    assert score(pred, gold) == {"strict": strict, "lenient": lenient, "sign_flip": sign_flip}