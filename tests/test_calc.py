import pytest

from finqa_rag_lora.calc import CalcError, safe_eval


@pytest.mark.parametrize("expr, expected", [
    ("705.4 - 696.3", 9.1),
    ("(4535 - 4180) / 4180", (4535 - 4180) / 4180),
    ("2 + 3 * 4", 14),                 # precedence
    ("-31", -31),
    ("18,178", 18178),                 # thousands separators
    ("$ 1,234.5 + 10", 1244.5),
    ("15%", 0.15),
    ("−2.4 × 3", -7.2),                # unicode minus and times
    ("(1.5 ** 0.5)", 1.5 ** 0.5),
    ("613 / 1178.", 613 / 1178),       # trailing period
    ("`100 / 4`", 25),
])
def test_valid_arithmetic(expr, expected):
    assert safe_eval(expr) == pytest.approx(expected)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('ls')",   # code, not arithmetic
    "open('x')",
    "x + 1",
    "1 / 0",
    "2 ** 1000",
    "",
    "1 +",
    "1" * 400,
    "(-8) ** 0.5",                     # complex result
])
def test_rejected(expr):
    with pytest.raises(CalcError):
        safe_eval(expr)