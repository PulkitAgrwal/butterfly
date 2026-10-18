"""Score attribution methods on Who&When with strict matching (exact step, case-insensitive agent)."""

import argparse
import json
import random
from pathlib import Path

from butterfly import store
from butterfly.dataset import Run, load_runs
from butterfly.rank import Priors, rank_suspects

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def same_agent(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower()


def bootstrap_ci(hits: list[bool], iters: int = 2000, seed: int = 7) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(hits)
    means = sorted(sum(rng.choice(hits) for _ in range(n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    step_hits = [r["step_hit"] for r in rows]
    lo, hi = bootstrap_ci(step_hits)
    return {
        "n": n,
        "agent_acc": sum(r["agent_hit"] for r in rows) / n,
        "step_acc": sum(step_hits) / n,
        "step_acc_ci95": [lo, hi],
        "step_top3": sum(r["top3_hit"] for r in rows) / n,
        "step_within1": sum(r["within1_hit"] for r in rows) / n,
    }


def score_prediction(run: Run, agent: str, step: int, ranked_steps: list[int]) -> dict:
    return {
        "run_id": run.run_id,
        "gold_agent": run.mistake_agent,
        "gold_step": run.mistake_step,
        "pred_agent": agent,
        "pred_step": step,
        "agent_hit": same_agent(agent, run.mistake_agent),
        "step_hit": step == run.mistake_step,
        "top3_hit": run.mistake_step in ranked_steps[:3],
        "within1_hit": abs(step - run.mistake_step) <= 1,
    }


def eval_graph(mode: str) -> dict:
    db = store.connect()
    runs = load_runs()
    priors = Priors(runs)
    rows = []
    for run in runs:
        ranked = rank_suspects(db, run, priors, mode=mode)
        top = ranked[0]
        rows.append({"split": run.split, **score_prediction(run, top.agent, top.idx, [s.idx for s in ranked])})
    report = {split: summarize([r for r in rows if r["split"] == split]) for split in ("algorithm_generated", "hand_crafted")}
    report["all"] = summarize(rows)
    return {"method": f"graph-only ({mode})", "report": report, "rows": rows}


def eval_llm(method: str, limit: int | None = None, workers: int = 4) -> dict:
    """method: 'butterfly' (graph + one LLM call) or 'baseline' (official all-at-once, same model)."""
    from concurrent.futures import ThreadPoolExecutor

    from butterfly import llm
    from butterfly.attribute import baseline, diagnose

    db = store.connect()
    runs = load_runs()
    priors = Priors(runs)
    if limit:  # pilot: a few runs from each split
        runs = [r for r in runs if r.split == "algorithm_generated"][:limit] + [
            r for r in runs if r.split == "hand_crafted"
        ][:limit]

    def one(run: Run) -> dict:
        if method == "butterfly":
            d = diagnose(db, run, priors)
        else:
            d = baseline(run, numbered=method == "baseline_numbered")
        row = score_prediction(run, d.agent, d.step, d.ranked_steps)
        return {"split": run.split, **row, "prompt_tokens": d.prompt_tokens, "completion_tokens": d.completion_tokens}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, runs))
    splits = [s for s in ("algorithm_generated", "hand_crafted") if any(r["split"] == s for r in rows)]
    report = {split: summarize([r for r in rows if r["split"] == split]) for split in splits}
    report["all"] = summarize(rows)
    for split, m in report.items():
        subset = rows if split == "all" else [r for r in rows if r["split"] == split]
        m["prompt_tokens_mean"] = sum(r["prompt_tokens"] for r in subset) / len(subset)
        m["completion_tokens_mean"] = sum(r["completion_tokens"] for r in subset) / len(subset)
    return {"method": f"{method} ({llm.MODEL})", "model": llm.MODEL, "report": report, "rows": rows}


def print_table(result: dict) -> None:
    print(f"\n{result['method']}")
    print(f"{'split':22} {'n':>4} {'agent':>7} {'step':>7} {'step 95% CI':>15} {'top-3':>7} {'±1':>7}")
    for split, m in result["report"].items():
        lo, hi = m["step_acc_ci95"]
        tokens = f" {m['prompt_tokens_mean']:>9,.0f} in" if "prompt_tokens_mean" in m else ""
        print(
            f"{split:22} {m['n']:>4} {m['agent_acc']:>7.1%} {m['step_acc']:>7.1%} "
            f"{f'{lo:.0%}-{hi:.0%}':>15} {m['step_top3']:>7.1%} {m['step_within1']:>7.1%}{tokens}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        default="graph",
        choices=["graph", "prior", "position", "butterfly", "baseline", "baseline_numbered"],
    )
    parser.add_argument("--limit", type=int, default=None, help="pilot: only the first N runs of each split")
    args = parser.parse_args()
    if args.method in ("butterfly", "baseline", "baseline_numbered"):
        result = eval_llm(args.method, args.limit)
        name = f"{args.method}_{result['model']}{'_pilot' if args.limit else ''}"
    else:
        result = eval_graph(args.method)
        name = f"graph_only_{args.method}"
    print_table(result)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
