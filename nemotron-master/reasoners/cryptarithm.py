"""Equation symbolic reasoning generator.

Currently handles concatenation operators only (forward and reverse).
Operates directly on the original symbols without letter assignment.
"""

from __future__ import annotations

from dataclasses import dataclass

from reasoners.store_types import Problem


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


def _box(s: str) -> str:
    """Wrap each character in 【】 brackets."""
    return "".join(f"【{c}】" for c in s)


# ── Arithmetic solver (promoted from investigators/cryptarithm_deduce.py) ──────
# Each symbol is a digit 0-9; the operator symbol is one arithmetic operation.
# Backtracking CSP infers the symbol->digit map + per-operator op from the
# examples, then applies it to the query. This recovers the ~30% of problems
# whose query operator is NOT plain concatenation. A node budget bounds runtime
# (reasoning.py loops every problem); reasoning.py re-verifies the boxed answer
# against the ground truth, so a wrong deduction is gated out, not trusted.
_ARITH_OPS = [
    lambda a, b: a + b,  # 0 add
    lambda a, b: abs(a - b),  # 1 abs_diff
    lambda a, b: a * b,  # 2 mul
    lambda a, b: a * 100 + b,  # 3 concat
    lambda a, b: b * 100 + a,  # 4 rev_concat
]
_ARITH_OP_NAMES = ["add", "abs_diff", "mul", "concat", "rev_concat"]
_ARITH_NODE_BUDGET = 300_000


def _num_to_digits(n: int) -> tuple[int, ...]:
    if n == 0:
        return (0,)
    d = []
    while n > 0:
        d.append(n % 10)
        n //= 10
    return tuple(reversed(d))


def _is_concat_tuple(s0: str, s1: str, s3: str, s4: str, rsyms: tuple) -> bool:
    return rsyms == (s0, s1, s3, s4) or rsyms == (s3, s4, s0, s1)


class _ArithSolver:
    """Backtracking digit/operator assignment; collects consistent query answers."""

    def __init__(self, examples, query, unique=True):
        self.examples = examples  # list[(s0,s1,op,s3,s4,rsyms)]
        self.query = query  # (q0,q1,qop,q3,q4)
        self.unique = unique
        self.mapping: dict = {}
        self.used: set = set()
        self.op_assign: dict = {}
        self.answers: dict = {}  # answer_str -> count
        self.answer_info: dict = {}  # answer_str -> (mapping, op_info)
        self.max_solutions = 200
        self.nodes = 0

    def solve(self):
        self._process(0)
        if not self.answers:
            return None
        best = max(self.answers, key=lambda k: self.answers[k])
        total = sum(self.answers.values())
        # Non-unique mapping is permissive; require a clear consensus answer.
        if not self.unique and total > 1 and self.answers[best] < total * 0.3:
            return None
        return best, self.answer_info.get(best, ({}, {}))

    def _process(self, idx):
        self.nodes += 1
        if self.nodes > _ARITH_NODE_BUDGET or len(self.answers) >= self.max_solutions:
            return
        if idx == len(self.examples):
            self._compute_query()
            return
        s0, s1, op_sym, s3, s4, rsyms = self.examples[idx]
        rlen = len(rsyms)
        feasible = []
        if rlen <= 3:
            feasible.append(0)
        if rlen <= 2:
            feasible.append(1)
        if rlen <= 4:
            feasible.append(2)
        if rlen == 4:
            feasible.extend([3, 4])

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
                            else feasible
                        )
                        for op_id in ops_to_try:
                            result_val = _ARITH_OPS[op_id](lv, rv)
                            if op_id >= 3:
                                if not (0 <= result_val < 10000):
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
                            if len(self.answers) >= self.max_solutions:
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

    def _compute_query(self):
        qs0, qs1, qop, qs3, qs4 = self.query
        for s in (qs0, qs1, qs3, qs4):
            if s not in self.mapping:
                return
        ql = self.mapping[qs0] * 10 + self.mapping[qs1]
        qr = self.mapping[qs3] * 10 + self.mapping[qs4]
        op_candidates = (
            [self.op_assign[qop]] if qop in self.op_assign else range(len(_ARITH_OPS))
        )
        d2s: dict = {}
        for s, d in self.mapping.items():
            d2s.setdefault(d, s)
        for op_id in op_candidates:
            result_val = _ARITH_OPS[op_id](ql, qr)
            if op_id >= 3:
                if not (0 <= result_val < 10000):
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
                op_info = {k: _ARITH_OP_NAMES[v] for k, v in self.op_assign.items()}
                op_info[qop] = _ARITH_OP_NAMES[op_id]
                self.answer_info[ans] = (dict(self.mapping), op_info)


def _solve_arithmetic(exs: list[_Ex], q_a, q_op, q_b):
    """Return (answer, mapping, op_info) or None for a non-concat query."""
    tuples = [
        (ex.a[0], ex.a[1], ex.op, ex.b[0], ex.b[1], tuple(ex.out)) for ex in exs
    ]
    # Only arithmetic (non-concat) examples constrain the digit map.
    arith = [t for t in tuples if not _is_concat_tuple(t[0], t[1], t[3], t[4], t[5])]
    if not arith:
        return None
    query = (q_a[0], q_a[1], q_op, q_b[0], q_b[1])
    res = _ArithSolver(arith, query, unique=True).solve()
    if res is None:
        res = _ArithSolver(arith, query, unique=False).solve()
    if res is None:
        return None
    answer, (mapping, op_info) = res
    return answer, mapping, op_info


def _arithmetic_cot(problem: Problem, exs, q_a, q_op, q_b, mapping, op_info, answer):
    """Deterministic CoT for an arithmetic (non-concat) cryptarithm deduction."""

    def q(s):
        return f"【{s}】"

    op_fmt = {
        "add": lambda a, b: f"{a} + {b} = {a + b}",
        "abs_diff": lambda a, b: f"|{a} - {b}| = {abs(a - b)}",
        "mul": lambda a, b: f"{a} * {b} = {a * b}",
        "concat": lambda a, b: f"concat({a}, {b}) = {a * 100 + b}",
        "rev_concat": lambda a, b: f"rev_concat({a}, {b}) = {b * 100 + a}",
    }
    lines = [
        "We need to infer the transformation rule from the examples.",
        "Each symbol stands for a digit (0-9); the operator symbol stands for an "
        "arithmetic operation on the two-digit numbers.",
        "I will put my final answer inside \\boxed{}.",
        "",
        "Deduced symbol-to-digit mapping:",
    ]
    for s, d in sorted(mapping.items()):
        lines.append(f"  {q(s)} = {d}")
    lines.append("")
    lines.append("Deduced operator meaning:")
    for s, name in sorted(op_info.items()):
        lines.append(f"  {q(s)} = {name}")
    lines.append("")
    lines.append("Verify on each example:")
    for ex in exs:
        inp = ex.a[0] + ex.a[1] + ex.op + ex.b[0] + ex.b[1]
        name = op_info.get(ex.op, "?")
        lv = mapping.get(ex.a[0]), mapping.get(ex.a[1])
        rv = mapping.get(ex.b[0]), mapping.get(ex.b[1])
        if None not in lv and None not in rv and name in op_fmt:
            left, right = lv[0] * 10 + lv[1], rv[0] * 10 + rv[1]
            lines.append(f"  {q(inp)} = {q(ex.out)}  =>  {op_fmt[name](left, right)}")
        else:
            lines.append(f"  {q(inp)} = {q(ex.out)}")
    lines.append("")
    q_inp = q_a[0] + q_a[1] + q_op + q_b[0] + q_b[1]
    name = op_info.get(q_op, "?")
    lv = mapping.get(q_a[0]), mapping.get(q_a[1])
    rv = mapping.get(q_b[0]), mapping.get(q_b[1])
    lines.append(f"Question {q(q_inp)}")
    if None not in lv and None not in rv and name in op_fmt:
        left, right = lv[0] * 10 + lv[1], rv[0] * 10 + rv[1]
        lines.append(f"  apply {name}: {op_fmt[name](left, right)}")
    lines.append(f"  encode the result back to symbols: {q(answer)}")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{answer}}}")
    return "\n".join(lines)


def reasoning_cryptarithm(problem: Problem) -> str | None:
    """Generate reasoning for cryptarithm problems."""

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

    # Group by operator
    by_op: dict[str, list[_Ex]] = {}
    for parsed_ex in exs:
        by_op.setdefault(parsed_ex.op, []).append(parsed_ex)

    # Detect concat types for each operator
    concat_types: dict[str, str] = {}
    for op, op_exs in by_op.items():
        ct = _concat_type(op_exs)
        if ct is not None:
            concat_types[op] = ct

    # Check question operator for concatenation type (default to fwd if unknown)
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

    # If the query operator is NOT a confirmed concatenation, the concat answer
    # above is just a guess. Try the arithmetic CSP solver instead; if it deduces
    # a consistent answer, emit that reasoning. Fall through to the concat trace
    # only when arithmetic can't solve it (preserves prior behaviour/coverage).
    if q_op not in concat_types:
        arith = _solve_arithmetic(exs, q_a, q_op, q_b)
        if arith is not None:
            arith_answer, mapping, op_info = arith
            return _arithmetic_cot(
                problem, exs, q_a, q_op, q_b, mapping, op_info, arith_answer
            )

    # Generate trace
    lines: list[str] = []
    lines.append("We need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    # Show each example with concatenation check
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

        # Operator line with type
        ct = concat_types.get(ex_parsed.op)
        if ct == "fwd":
            op_type = "concatenation"
        elif ct == "rev":
            op_type = "reverse concatenation"
        else:
            op_type = "unknown"
        lines.append(f"  operator: {quote(ex_parsed.op)}{op_type}")
        lines.append("")

    # Apply to question
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
        lines.append(
            f"The question operator is {quote(q_op)}, which is {op_label}."
        )
    else:
        lines.append(f"The question operator is {quote(q_op)}, which is unknown.")
        lines.append(
            "As the question operator is unknown, we default to concatenation."
        )
    lines.append("")

    lines.append(
        f"  {op_label}({qa0}{qa1}, {qb0}{qb1}) = {_box(answer)}"
    )
    lines.append(f"  output: {quote(answer)}-> {quote('{' + answer + '}')}")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{answer}}}")
    return "\n".join(lines)
