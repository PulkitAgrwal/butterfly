"""Paired comparison of two methods on the same runs."""

from math import comb


def mcnemar(rows_a: list[dict], rows_b: list[dict], key: str) -> dict:
    """Exact two-sided McNemar test on per-run hits (e.g. key='step_hit')."""
    a = {r["run_id"]: r[key] for r in rows_a}
    b = {r["run_id"]: r[key] for r in rows_b}
    only_a = sum(1 for k in a if a[k] and not b.get(k))
    only_b = sum(1 for k in a if b.get(k) and not a[k])
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"only_a": only_a, "only_b": only_b, "p": p}
