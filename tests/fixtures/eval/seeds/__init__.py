"""Seed corpus for the Pod 4.4c-3 planting pipeline (TEST-ONLY — python-expert ruled this is not
in-package). Each :class:`~cogworx.eval.planting.Seed` is a known-correct, code-domain solution + a
frozen ``test_code`` importing ``from solution import <name>``. The shapes span the three
O-derivable regime classes so the operator mix yields all of them:

  * arithmetic kernels        -> ``arithmetic-swap`` (logic-wrong), ``constant-replacement`` /
                                 ``sign-flip`` / ``unit`` (off-by-semantics)
  * boundary / loop / compare -> ``relational-swap`` (logic-wrong), ``boundary`` /
                                 ``statement-deletion`` (edge-case-miss)
  * unit / sign-sensitive     -> ``unit`` / ``sign-flip`` (off-by-semantics)

BUILD REQUIREMENT (red-team bait #5): every seed's correct solution MUST pass its own frozen test
under the executable oracle BEFORE it enters the set — a seed whose "correct" solution fails its own
test poisons both the O-mutation base AND the clean pool. ``tests/eval/test_planting.py`` wires this
self-check via :func:`cogworx.eval._authoring.run_frozen_check` (journal-free, S1-clean authoring).
"""

from __future__ import annotations

from textwrap import dedent

from cogworx.eval.planting import Seed
from cogworx.verification.contracts import OracleFrame, Thesis


def _seed(seed_id: int, statement: str, solution: str, test_code: str) -> Seed:
    return Seed(
        seed_id=seed_id,
        frame=OracleFrame(
            completion_criterion="tests_pass",
            problem_type="code",
            problem_statement=statement,
        ),
        thesis=Thesis(
            proposed_solution=dedent(solution).strip() + "\n",
            experiment_design="run the frozen tests",
        ),
        test_code=dedent(test_code).strip() + "\n",
    )


# Each entry: (statement, solution, frozen test). Solutions go to solution.py; tests import
# `from solution import <name>`. Kept small + deterministic (the oracle runs them per author check).
_RAW: list[tuple[str, str, str]] = [
    # --- arithmetic kernels (arithmetic-swap / constant-replacement / sign-flip / unit) ---
    (
        "add two ints",
        "def add(a, b):\n    return a + b",
        "from solution import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
        "    assert add(-1, 1) == 0",
    ),
    (
        "subtract two ints",
        "def sub(a, b):\n    return a - b",
        "from solution import sub\n\n\ndef test_sub():\n    assert sub(5, 3) == 2\n"
        "    assert sub(0, 4) == -4",
    ),
    (
        "multiply two ints",
        "def mul(a, b):\n    return a * b",
        "from solution import mul\n\n\ndef test_mul():\n    assert mul(3, 4) == 12\n"
        "    assert mul(-2, 5) == -10",
    ),
    (
        "scale by a constant factor of 2",
        "def scale(x):\n    return x * 2",
        "from solution import scale\n\n\ndef test_scale():\n    assert scale(3) == 6\n"
        "    assert scale(0) == 0",
    ),
    (
        "add a constant offset of 10",
        "def offset(x):\n    return x + 10",
        "from solution import offset\n\n\ndef test_offset():\n    assert offset(0) == 10\n"
        "    assert offset(5) == 15",
    ),
    (
        "integer average of two ints",
        "def avg(a, b):\n    return (a + b) // 2",
        "from solution import avg\n\n\ndef test_avg():\n    assert avg(2, 4) == 3\n"
        "    assert avg(10, 10) == 10",
    ),
    (
        "square a number",
        "def square(x):\n    return x * x",
        "from solution import square\n\n\ndef test_square():\n    assert square(4) == 16\n"
        "    assert square(-3) == 9",
    ),
    (
        "remainder modulo 3",
        "def mod3(x):\n    return x % 3",
        "from solution import mod3\n\n\ndef test_mod3():\n    assert mod3(7) == 1\n"
        "    assert mod3(9) == 0",
    ),
    (
        "negate a number",
        "def negate(x):\n    return -x",
        "from solution import negate\n\n\ndef test_negate():\n    assert negate(5) == -5\n"
        "    assert negate(-2) == 2",
    ),
    (
        "absolute difference of two ints",
        "def absdiff(a, b):\n    d = a - b\n    if d < 0:\n        d = -d\n    return d",
        "from solution import absdiff\n\n\ndef test_absdiff():\n    assert absdiff(3, 7) == 4\n"
        "    assert absdiff(7, 3) == 4",
    ),
    # --- boundary / loop / compare (relational-swap / boundary / statement-deletion) ---
    (
        "return True iff x is positive",
        "def is_positive(x):\n    return x > 0",
        "from solution import is_positive\n\n\ndef test_is_positive():\n"
        "    assert is_positive(1) is True\n    assert is_positive(0) is False\n"
        "    assert is_positive(-1) is False",
    ),
    (
        "return True iff x is non-negative",
        "def non_negative(x):\n    return x >= 0",
        "from solution import non_negative\n\n\ndef test_non_negative():\n"
        "    assert non_negative(0) is True\n    assert non_negative(-1) is False",
    ),
    (
        "max of two ints",
        "def maximum(a, b):\n    if a > b:\n        return a\n    return b",
        "from solution import maximum\n\n\ndef test_maximum():\n    assert maximum(3, 7) == 7\n"
        "    assert maximum(9, 2) == 9\n    assert maximum(4, 4) == 4",
    ),
    (
        "min of two ints",
        "def minimum(a, b):\n    if a < b:\n        return a\n    return b",
        "from solution import minimum\n\n\ndef test_minimum():\n    assert minimum(3, 7) == 3\n"
        "    assert minimum(9, 2) == 2",
    ),
    (
        "clamp x into [0, 10]",
        "def clamp(x):\n    if x < 0:\n        return 0\n    if x > 10:\n"
        "        return 10\n    return x",
        "from solution import clamp\n\n\ndef test_clamp():\n    assert clamp(-5) == 0\n"
        "    assert clamp(20) == 10\n    assert clamp(5) == 5\n    assert clamp(0) == 0\n"
        "    assert clamp(10) == 10",
    ),
    (
        "sum the first n natural numbers",
        "def sum_to(n):\n    total = 0\n    for i in range(1, n + 1):\n"
        "        total = total + i\n    return total",
        "from solution import sum_to\n\n\ndef test_sum_to():\n    assert sum_to(5) == 15\n"
        "    assert sum_to(0) == 0\n    assert sum_to(1) == 1",
    ),
    (
        "count elements strictly greater than a threshold",
        "def count_above(xs, t):\n    c = 0\n    for x in xs:\n        if x > t:\n"
        "            c = c + 1\n    return c",
        "from solution import count_above\n\n\ndef test_count_above():\n"
        "    assert count_above([1, 5, 3, 9], 4) == 2\n    assert count_above([], 0) == 0",
    ),
    (
        "last index of a list (length-aware boundary)",
        "def last_index(xs):\n    if len(xs) == 0:\n        return -1\n    return len(xs) - 1",
        "from solution import last_index\n\n\ndef test_last_index():\n"
        "    assert last_index([10, 20, 30]) == 2\n    assert last_index([]) == -1",
    ),
    (
        "factorial via a loop",
        "def fact(n):\n    result = 1\n    for i in range(2, n + 1):\n"
        "        result = result * i\n    return result",
        "from solution import fact\n\n\ndef test_fact():\n    assert fact(0) == 1\n"
        "    assert fact(1) == 1\n    assert fact(5) == 120",
    ),
    (
        "linear search returns index or -1",
        "def find(xs, target):\n    for i in range(len(xs)):\n        if xs[i] == target:\n"
        "            return i\n    return -1",
        "from solution import find\n\n\ndef test_find():\n    assert find([4, 8, 15], 8) == 1\n"
        "    assert find([4, 8, 15], 99) == -1",
    ),
    # --- unit / sign-sensitive (unit / sign-flip / constant-replacement) ---
    (
        "minutes to seconds",
        "def to_seconds(minutes):\n    return minutes * 60",
        "from solution import to_seconds\n\n\ndef test_to_seconds():\n"
        "    assert to_seconds(2) == 120\n    assert to_seconds(0) == 0",
    ),
    (
        "hours to minutes",
        "def to_minutes(hours):\n    return hours * 60",
        "from solution import to_minutes\n\n\ndef test_to_minutes():\n"
        "    assert to_minutes(3) == 180\n    assert to_minutes(1) == 60",
    ),
    (
        "kilometres to metres",
        "def km_to_m(km):\n    return km * 1000",
        "from solution import km_to_m\n\n\ndef test_km_to_m():\n    assert km_to_m(2) == 2000\n"
        "    assert km_to_m(0) == 0",
    ),
    (
        "celsius to fahrenheit (linear unit transform)",
        "def c_to_f(c):\n    return c * 9 // 5 + 32",
        "from solution import c_to_f\n\n\ndef test_c_to_f():\n    assert c_to_f(0) == 32\n"
        "    assert c_to_f(100) == 212",
    ),
    (
        "apply a discount of 5 off a price",
        "def discount(price):\n    return price - 5",
        "from solution import discount\n\n\ndef test_discount():\n    assert discount(20) == 15\n"
        "    assert discount(5) == 0",
    ),
    (
        "signed step toward zero by 1",
        "def step_to_zero(x):\n    if x > 0:\n        return x - 1\n    if x < 0:\n"
        "        return x + 1\n    return 0",
        "from solution import step_to_zero\n\n\ndef test_step_to_zero():\n"
        "    assert step_to_zero(3) == 2\n    assert step_to_zero(-3) == -2\n"
        "    assert step_to_zero(0) == 0",
    ),
    (
        "double then add one",
        "def double_plus_one(x):\n    return x * 2 + 1",
        "from solution import double_plus_one\n\n\ndef test_double_plus_one():\n"
        "    assert double_plus_one(3) == 7\n    assert double_plus_one(0) == 1",
    ),
    (
        "balance after a withdrawal of 100",
        "def balance_after(balance):\n    return balance - 100",
        "from solution import balance_after\n\n\ndef test_balance_after():\n"
        "    assert balance_after(250) == 150\n    assert balance_after(100) == 0",
    ),
]

SEEDS: tuple[Seed, ...] = tuple(
    _seed(i, statement, solution, test_code)
    for i, (statement, solution, test_code) in enumerate(_RAW)
)
"""The frozen seed corpus (>=25). ``seed_id`` is the 0-based index into ``_RAW``."""


__all__ = ["SEEDS"]
