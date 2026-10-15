"""Safely evaluate the arithmetic expression a model writes after `CALC:`.

Never use Python's eval() on model output: it can run arbitrary code. This parses the
expression and allows only numbers, + - * / ** and parentheses.
"""

from __future__ import annotations

import ast
import math
import operator
import re

MAX_LENGTH = 300

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class CalcError(ValueError):
    """The expression is not valid, safe arithmetic."""


def clean_expression(expr: str) -> str:
    """Normalize common ways models write numbers so they parse as arithmetic."""
    s = expr.strip().rstrip(".").strip()
    s = s.strip("`").strip()
    s = s.replace("−", "-").replace("–", "-").replace("×", "*").replace("÷", "/")
    s = s.replace("$", "")
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)            # 1,234,567 -> 1234567
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", s)        # 15% -> (15/100)
    return s.strip()


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise CalcError("exponent too large")
        try:
            return _BINARY[type(node.op)](left, right)
        except ZeroDivisionError as err:
            raise CalcError("division by zero") from err
        except OverflowError as err:
            raise CalcError("result too large") from err
    raise CalcError(f"not allowed in an arithmetic expression: {type(node).__name__}")


def safe_eval(expr: str) -> float:
    """Evaluate an arithmetic expression, or raise CalcError."""
    s = clean_expression(expr)
    if not s:
        raise CalcError("empty expression")
    if len(s) > MAX_LENGTH:
        raise CalcError("expression too long")
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as err:
        raise CalcError(f"not valid arithmetic: {s!r}") from err
    value = _eval(tree)
    if isinstance(value, complex) or not math.isfinite(value):
        raise CalcError("result is not a finite real number")
    return float(value)