"""Cryptarithm reasoning generator.

Two regimes are handled:

* **Concatenation operators** (forward / reverse) — the result is just the
  operands glued together. Solved by direct symbol manipulation.
* **Arithmetic operators** (add / abs_diff / mul) — each symbol hides a digit
  and each operator hides an arithmetic operation. A backtracking search
  (ported from ``investigators/cryptarithm_deduce.py``) deduces a consistent
  symbol->digit and operator->operation mapping; a natural-language CoT then
  explains the mapping and the arithmetic step by step.

The arithmetic path is tried first whenever the question operator is not a pure
concatenation. If the search cannot find a consistent assignment within its
budget, we fall back to the concatenation reasoning (the prior behaviour), so a
trace is always produced and nothing regresses.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from reasoners.store_types import Problem


def _extract_boxed(text: str) -> str:
    """Mirror reasoning.extract_answer (the downstream grader extraction)."""
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    return ""


def _verify(stored: str, predicted: str) -> bool:
    """Mirror reasoning.compare_answer (inlined to avoid a circular import)."""
    stored = stored.strip()
    predicted = predicted.strip()
    if re.fullmatch(r"[01]+", stored):
        return predicted.lower() == stored.lower()
    try:
        return math.isclose(float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored.lower()


# ---------------------------------------------------------------------------
# Solver (ported from investigators/cryptarithm_deduce.py, with a node budget)
# ---------------------------------------------------------------------------

# index matches OP_NAMES below
OPS = [
    lambda a, b: a + b,  # 0: add
    lambda a, b: abs(a - b),  # 1: abs_diff
    lambda a, b: a * b,  # 2: mul
    lambda a, b: a * 100 + b,  # 3: concat
    lambda a, b: b * 100 + a,  # 4: rev_concat
]
OP_NAMES = ["add", "abs_diff", "mul", "concat", "rev_concat"]

# Search budgets (number of recursion nodes). Solvable instances finish well
# within these; hard/unsolvable instances bail out fast instead of hanging.
_UNIQUE_BUDGET = 1_500_000
_NONUNIQUE_BUDGET = 400_000


def _num_to_digits(n: int) -> tuple[int, ...]:
    if n == 0:
        return (0,)
    d = []
    while n > 0:
        d.append(n % 10)
        n //= 10
    return tuple(reversed(d))


def _is_concat(ex: tuple) -> bool:
    """True if the result is s0s1s3s4 or s3s4s0s1 (trivial concatenation)."""
    s0, s1, _, s3, s4, rsyms = ex
    return rsyms == (s0, s1, s3, s4) or rsyms == (s3, s4, s0, s1)


class _Solver:
    """Backtracking search for a symbol->digit / operator->operation mapping."""

    def __init__(self, examples, query, unique=True, budget=_UNIQUE_BUDGET):
        self.examples = examples
        self.query = query
        self.unique = unique
        self.mapping: dict[str, int] = {}
        self.used: set[int] = set()
        self.op_assign: dict[str, int] = {}
        self.answers: dict[str, int] = {}
        self.answer_info: dict[str, tuple[dict, dict]] = {}
        self.max_solutions = 200
        self.budget = budget
        self.nodes = 0
        self.aborted = False

    def solve(self):
        self._process(0)
        if self.answers:
            best = max(self.answers, key=lambda k: self.answers[k])
            best_count = self.answers[best]
            total = sum(self.answers.values())
            # For non-unique mappings, require a strong consensus.
            if not self.unique and total > 1 and best_count < total * 0.3:
                return None, ({}, {})
            return best, self.answer_info.get(best, ({}, {}))
        return None, ({}, {})

    def _process(self, idx: int) -> None:
        if self.aborted or len(self.answers) >= self.max_solutions:
            return
        self.nodes += 1
        if self.nodes > self.budget:
            self.aborted = True
            return
        if idx == len(self.examples):
            self._compute_query()
            return

        s0, s1, op_sym, s3, s4, rsyms = self.examples[idx]
        rlen = len(rsyms)

        # Restrict operations by feasible result length.
        feasible_ops = []
        if rlen <= 3:
            feasible_ops.append(0)  # add
        if rlen <= 2:
            feasible_ops.append(1)  # abs_diff
        if rlen <= 4:
            feasible_ops.append(2)  # mul
        if rlen == 4:
            feasible_ops.extend([3, 4])  # concat, rev_concat

        for d0 in self._vals(s0):
            n0 = self._assign(s0, d0)
            if n0 is None:
                continue
            for d1 in self._vals(s1):
                n1 = self._assign(s1, d1)
                if n1 is None:
                    continue
                lv = d0 * 10 + d1
                for d3 in self._vals(s3):
                    n3 = self._assign(s3, d3)
                    if n3 is None:
                        continue
                    for d4 in self._vals(s4):
                        n4 = self._assign(s4, d4)
                        if n4 is None:
                            continue
                        rv = d3 * 10 + d4

                        ops_to_try = (
                            [self.op_assign[op_sym]]
                            if op_sym in self.op_assign
                            else feasible_ops
                        )

                        for op_id in ops_to_try:
                            result_val = OPS[op_id](lv, rv)
                            if op_id >= 3:
                                if result_val < 0 or result_val >= 10000:
                                    continue
                                rd = (
                                    result_val // 1000,
                                    (result_val // 100) % 10,
                                    (result_val // 10) % 10,
                                    result_val % 10,
                                )
                            else:
                                rd = _num_to_digits(result_val)
                            if len(rd) != rlen:
                                continue

                            assigns = []
                            ok = True
                            for rs, rdig in zip(rsyms, rd):
                                ns = self._assign(rs, rdig)
                                if ns is None:
                                    ok = False
                                    break
                                assigns.append((rs, ns))

                            if ok:
                                op_new = op_sym not in self.op_assign
                                if op_new:
                                    self.op_assign[op_sym] = op_id
                                self._process(idx + 1)
                                if op_new:
                                    del self.op_assign[op_sym]

                            for rs, ns in reversed(assigns):
                                self._undo(rs, ns)

                            if self.aborted or len(self.answers) >= self.max_solutions:
                                self._undo(s4, n4)
                                self._undo(s3, n3)
                                self._undo(s1, n1)
                                self._undo(s0, n0)
                                return

                        self._undo(s4, n4)
                    self._undo(s3, n3)
                self._undo(s1, n1)
            self._undo(s0, n0)

    def _vals(self, sym):
        if sym in self.mapping:
            return (self.mapping[sym],)
        if self.unique:
            return tuple(d for d in range(10) if d not in self.used)
        return range(10)

    def _assign(self, sym, dig):
        if sym in self.mapping:
            return False if self.mapping[sym] == dig else None
        if self.unique and dig in self.used:
            return None
        self.mapping[sym] = dig
        if self.unique:
            self.used.add(dig)
        return True

    def _undo(self, sym, was_new):
        if was_new is True:
            if self.unique:
                self.used.discard(self.mapping[sym])
            del self.mapping[sym]

    def _compute_query(self) -> None:
        qs0, qs1, qop, qs3, qs4 = self.query
        for s in (qs0, qs1, qs3, qs4):
            if s not in self.mapping:
                return

        ql = self.mapping[qs0] * 10 + self.mapping[qs1]
        qr = self.mapping[qs3] * 10 + self.mapping[qs4]
        if qop in self.op_assign:
            op_candidates = [self.op_assign[qop]]
        else:
            op_candidates = range(len(OP_NAMES))

        d2s: dict[int, str] = {}
        for s, d in self.mapping.items():
            if d not in d2s:
                d2s[d] = s

        for op_id in op_candidates:
            result_val = OPS[op_id](ql, qr)
            if op_id >= 3:
                if result_val < 0 or result_val >= 10000:
                    continue
                rd = (
                    result_val // 1000,
                    (result_val // 100) % 10,
                    (result_val // 10) % 10,
                    result_val % 10,
                )
            else:
                rd = _num_to_digits(result_val)

            parts = []
            ok = True
            for d in rd:
                if d not in d2s:
                    ok = False
                    break
                parts.append(d2s[d])
            if not ok:
                continue

            ans = "".join(parts)
            self.answers[ans] = self.answers.get(ans, 0) + 1
            if ans not in self.answer_info:
                op_info = {k: OP_NAMES[v] for k, v in self.op_assign.items()}
                op_info[qop] = OP_NAMES[op_id]
                self.answer_info[ans] = (dict(self.mapping), op_info)


def _solve_arith(examples, query):
    """Return (answer, mapping, op_info) or (None, {}, {})."""
    arith_examples = [ex for ex in examples if not _is_concat(ex)]
    if not arith_examples:
        return None, {}, {}

    solver = _Solver(arith_examples, query, unique=True, budget=_UNIQUE_BUDGET)
    ans, (mapping, op_info) = solver.solve()
    if ans is not None:
        return ans, mapping, op_info

    solver2 = _Solver(arith_examples, query, unique=False, budget=_NONUNIQUE_BUDGET)
    ans, (mapping, op_info) = solver2.solve()
    return (ans, mapping, op_info) if ans is not None else (None, {}, {})


# ---------------------------------------------------------------------------
# Reasoning text generators
# ---------------------------------------------------------------------------

_OP_VERB = {
    "add": "addition",
    "abs_diff": "absolute difference",
    "mul": "multiplication",
    "concat": "concatenation",
    "rev_concat": "reverse concatenation",
}


def _box(s: str) -> str:
    """Wrap each character in 【】 brackets."""
    return "".join(f"【{c}】" for c in s)


def _quote(s: str) -> str:
    return f"【{s}】"


def _op_calc(op_name: str, a: int, b: int) -> str:
    if op_name == "add":
        return f"{a} + {b} = {a + b}"
    if op_name == "abs_diff":
        return f"|{a} - {b}| = {abs(a - b)}"
    if op_name == "mul":
        return f"{a} * {b} = {a * b}"
    if op_name == "concat":
        return f"concat({a}, {b}) = {a * 100 + b}"
    if op_name == "rev_concat":
        return f"rev_concat({a}, {b}) = {b * 100 + a}"
    return "?"


def _reasoning_arith(
    problem: Problem,
    examples: list[tuple],
    query: tuple,
    answer: str,
    mapping: dict[str, int],
    op_info: dict[str, str],
) -> str:
    """Explain the deduced digit/operation mapping and the query arithmetic."""
    lines: list[str] = []
    lines.append(
        "Each symbol stands for a single digit and each operator stands for "
        "an arithmetic operation. I will deduce both from the examples."
    )
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    lines.append("Examples:")
    for ex in examples:
        s0, s1, op_sym, s3, s4, rsyms = ex
        inp = s0 + s1 + op_sym + s3 + s4
        out = "".join(rsyms)
        lines.append(f"  {_box(inp)} = {_box(out)}")
    lines.append("")

    lines.append("Deduced symbol-to-digit mapping:")
    for s, d in sorted(mapping.items()):
        lines.append(f"  {_quote(s)} = {d}")
    lines.append("")

    lines.append("Deduced operator-to-operation mapping:")
    for s, name in sorted(op_info.items()):
        verb = _OP_VERB.get(name, name)
        lines.append(f"  {_quote(s)} = {name} ({verb})")
    lines.append("")

    # Verify each example numerically where the mapping covers it.
    lines.append("Verifying the examples with this mapping:")
    for ex in examples:
        s0, s1, op_sym, s3, s4, rsyms = ex
        inp = s0 + s1 + op_sym + s3 + s4
        out = "".join(rsyms)
        op_name = op_info.get(op_sym)
        if (
            op_name is not None
            and s0 in mapping
            and s1 in mapping
            and s3 in mapping
            and s4 in mapping
        ):
            lv = mapping[s0] * 10 + mapping[s1]
            rv = mapping[s3] * 10 + mapping[s4]
            lines.append(f"  {_box(inp)} -> {_op_calc(op_name, lv, rv)} -> {_box(out)}")
        else:
            lines.append(f"  {_box(inp)} = {_box(out)}")
    lines.append("")

    # Apply to the question.
    qs0, qs1, qop, qs3, qs4 = query
    q_str = qs0 + qs1 + qop + qs3 + qs4
    op_name = op_info.get(qop, "?")
    verb = _OP_VERB.get(op_name, op_name)
    lines.append(f"Question {_box(q_str)}")
    lines.append(
        f"  left {_quote(qs0)}{_quote(qs1)} = {mapping[qs0]}{mapping[qs1]}"
        f" = {mapping[qs0] * 10 + mapping[qs1]}"
    )
    lines.append(
        f"  right {_quote(qs3)}{_quote(qs4)} = {mapping[qs3]}{mapping[qs4]}"
        f" = {mapping[qs3] * 10 + mapping[qs4]}"
    )
    lines.append(f"  operator {_quote(qop)} is {op_name} ({verb})")

    lv = mapping[qs0] * 10 + mapping[qs1]
    rv = mapping[qs3] * 10 + mapping[qs4]
    lines.append(f"  {_op_calc(op_name, lv, rv)}")
    lines.append(f"  mapping the result digits back to symbols gives {_box(answer)}")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{answer}}}")
    return "\n".join(lines)


@dataclass
class _Ex:
    a: tuple[str, str]
    op: str
    b: tuple[str, str]
    out: str


def _concat_type(exs: list[_Ex]) -> str | None:
    """Return 'fwd' if A1A2B1B2, 'rev' if B1B2A1A2, else None."""
    if all(ex.out == ex.a[0] + ex.a[1] + ex.b[0] + ex.b[1] for ex in exs):
        return "fwd"
    if all(ex.out == ex.b[0] + ex.b[1] + ex.a[0] + ex.a[1] for ex in exs):
        return "rev"
    return None


def _reasoning_concat(problem: Problem) -> str | None:
    """Concatenation-only reasoning (the original behaviour)."""

    def quote(s: str) -> str:
        return f"【{s}】"

    exs: list[_Ex] = []
    for ex in problem.examples:
        inp = str(ex.input_value)
        if len(inp) != 5:
            return None
        exs.append(
            _Ex(
                a=(inp[0], inp[1]),
                op=inp[2],
                b=(inp[3], inp[4]),
                out=str(ex.output_value),
            )
        )

    q = str(problem.question)
    if len(q) != 5:
        return None
    q_a = (q[0], q[1])
    q_op = q[2]
    q_b = (q[3], q[4])

    by_op: dict[str, list[_Ex]] = {}
    for parsed_ex in exs:
        by_op.setdefault(parsed_ex.op, []).append(parsed_ex)

    concat_types: dict[str, str] = {}
    for op, op_exs in by_op.items():
        ct = _concat_type(op_exs)
        if ct is not None:
            concat_types[op] = ct

    if q_op in by_op:
        q_ct = _concat_type(by_op[q_op])
        if q_ct is None:
            q_ct = "fwd"
    else:
        q_ct = "fwd"

    if q_ct == "fwd":
        answer = q_a[0] + q_a[1] + q_b[0] + q_b[1]
    else:
        answer = q_b[0] + q_b[1] + q_a[0] + q_a[1]

    lines: list[str] = []
    lines.append("We need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    for ex, ex_parsed in zip(problem.examples, exs):
        orig_inp = str(ex.input_value)
        orig_out = str(ex.output_value)
        lines.append(f"{quote(orig_inp)} = {quote(orig_out)}")
        a0, a1 = quote(ex_parsed.a[0]), quote(ex_parsed.a[1])
        b0, b1 = quote(ex_parsed.b[0]), quote(ex_parsed.b[1])
        op_q = quote(ex_parsed.op)
        out_boxed = _box(orig_out)
        lines.append(f"  input: {a0}{a1}{op_q}{b0}{b1}")
        lines.append(f"  left:{a0}{a1}")
        lines.append(f"  operator: {op_q}")
        lines.append(f"  right:{b0}{b1}")
        lines.append(f"  output: {out_boxed}")

        fwd = ex_parsed.a[0] + ex_parsed.a[1] + ex_parsed.b[0] + ex_parsed.b[1]
        rev = ex_parsed.b[0] + ex_parsed.b[1] + ex_parsed.a[0] + ex_parsed.a[1]
        is_fwd = orig_out == fwd
        is_rev = orig_out == rev

        lines.append(
            f"  concatenation: {_box(fwd)} {'match' if is_fwd else 'mismatch'}"
        )
        lines.append(
            f"  reverse concatenation: {_box(rev)} {'match' if is_rev else 'mismatch'}"
        )

        ct = concat_types.get(ex_parsed.op)
        if ct == "fwd":
            op_type = "concatenation"
        elif ct == "rev":
            op_type = "reverse concatenation"
        else:
            op_type = "unknown"
        lines.append(f"  operator: {quote(ex_parsed.op)}{op_type}")
        lines.append("")

    q_op_known = q_op in concat_types
    op_label = "concatenation" if q_ct == "fwd" else "reverse concatenation"

    qa0, qa1 = quote(q_a[0]), quote(q_a[1])
    qb0, qb1 = quote(q_b[0]), quote(q_b[1])
    q_orig = str(problem.question)
    lines.append(f"Question{quote(q_orig)}")
    lines.append(f"  input: {qa0}{qa1}{quote(q_op)}{qb0}{qb1}")
    lines.append(f"  left:{qa0}{qa1}")
    lines.append(f"  operator:{quote(q_op)}")
    lines.append(f"  right:{qb0}{qb1}")
    lines.append("")

    if q_op_known:
        lines.append(f"The question operator is {quote(q_op)}, which is {op_label}.")
    else:
        lines.append(f"The question operator is {quote(q_op)}, which is unknown.")
        lines.append(
            "As the question operator is unknown, we default to concatenation."
        )
    lines.append("")

    lines.append(f"  {op_label}({qa0}{qa1}, {qb0}{qb1}) = {_box(answer)}")
    lines.append(f"  output: {quote(answer)}-> {quote('{' + answer + '}')}")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{answer}}}")
    return "\n".join(lines)


def reasoning_cryptarithm(problem: Problem) -> str | None:
    """Generate reasoning for cryptarithm problems.

    Tries the arithmetic solver first (digit/operation deduction); if the
    question operator is a pure concatenation, or the search fails, falls back
    to concatenation reasoning.
    """
    # Parse examples and question into the solver's tuple form.
    examples: list[tuple] = []
    for ex in problem.examples:
        inp = str(ex.input_value)
        out = str(ex.output_value)
        if len(inp) != 5:
            return _reasoning_concat(problem)
        examples.append((inp[0], inp[1], inp[2], inp[3], inp[4], tuple(out)))

    q = str(problem.question)
    if len(q) != 5:
        return _reasoning_concat(problem)
    query = (q[0], q[1], q[2], q[3], q[4])
    q_op = query[2]

    # Classify operators as concatenation vs. non-concatenation.
    concat_ops: set[str] = set()
    nonconcat_ops: set[str] = set()
    for ex in examples:
        if _is_concat(ex):
            concat_ops.add(ex[2])
        else:
            nonconcat_ops.add(ex[2])

    # Pure concatenation question -> concat reasoning is exact and fast.
    if q_op in concat_ops and q_op not in nonconcat_ops:
        return _reasoning_concat(problem)

    # Otherwise attempt arithmetic deduction. The search is heuristic (it can
    # settle on a self-consistent but wrong mapping), so only emit the
    # arithmetic trace when its answer matches the ground truth; this keeps
    # incorrect arithmetic traces out of the corpus.
    answer, mapping, op_info = _solve_arith(examples, query)
    if answer is not None:
        trace = _reasoning_arith(problem, examples, query, answer, mapping, op_info)
        # Gate on the *rendered* trace: the boxed answer must extract cleanly
        # and match the ground truth. This rejects both wrong mappings and
        # answers whose symbols break \boxed{} extraction (e.g. a literal '}').
        if _verify(str(problem.answer), _extract_boxed(trace)):
            return trace

    # Fall back to concatenation reasoning (prior behaviour, no regression).
    return _reasoning_concat(problem)
