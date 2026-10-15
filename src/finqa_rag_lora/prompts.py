"""The prompt every model gets, and how its answer is read back.

Gemini, the base small model and the LoRA model all use these same functions,
so differences in results come from the model, not the prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from finqa_rag_lora.calc import CalcError, safe_eval

SYSTEM_PROMPT = """You answer questions about company financial reports using only the documents provided.

Rules:
- Find the numbers you need in the documents' tables and text. Check the company, the year and the row before using a number.
- Tables write negative numbers like "-2.4 ( 2.4 )"; both parts mean -2.4.
- Write percentages and ratios as fractions: 0.15 means 15%.
- A decrease, decline or loss is a negative number.
- Keep the units used in the documents (if a table is in millions, answer in millions).
- Reason briefly, then end with exactly one final line in this form:
CALC: <arithmetic expression>
The expression may contain only numbers, + - * / and parentheses, for example:
CALC: (705.4 - 696.3) / 696.3
If the answer is a number read directly from a document, write it alone, for example:
CALC: 18178"""

# Tolerates markdown around the label, e.g. "**CALC:** 1 + 2" or "`CALC: 1 + 2`".
_CALC_LINE = re.compile(r"^\W*CALC\W*?:[\s*_`]*(.+?)[\s*_`]*$", re.IGNORECASE | re.MULTILINE)
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def format_document(i: int, doc: dict) -> str:
    return (f"[Document {i}] Company: {doc['company_name']} | Report year: {doc['report_year']}\n"
            f"{doc['text']}")


def build_prompt(question: str, docs: list[dict]) -> str:
    """The user message: the documents (best match first), then the question."""
    parts = [format_document(i, d) for i, d in enumerate(docs, start=1)]
    return "Documents:\n\n" + "\n\n".join(parts) + f"\n\nQuestion: {question}"


@dataclass
class ParsedAnswer:
    value: float | None      # the number the model's answer evaluates to (None if unreadable)
    expression: str | None   # what followed CALC:
    format_ok: bool          # True if a CALC line was found AND evaluated
    error: str | None = None


def parse_answer(response: str) -> ParsedAnswer:
    """Read the final CALC line; if there isn't one, fall back to the last number in the text."""
    response = response or ""
    matches = _CALC_LINE.findall(response)
    if matches:
        expr = matches[-1]
        try:
            return ParsedAnswer(safe_eval(expr), expr, True)
        except CalcError as err:
            return ParsedAnswer(None, expr, False, str(err))

    numbers = _NUMBER.findall(response)
    if numbers:
        try:
            return ParsedAnswer(float(numbers[-1].replace(",", "")), None, False, "no CALC line; used last number")
        except ValueError:
            pass
    return ParsedAnswer(None, None, False, "no CALC line and no number")