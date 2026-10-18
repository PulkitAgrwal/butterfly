"""Render results/*.json into the README results block, with paired significance tests."""

import json
from pathlib import Path

from butterfly.stats import mcnemar as _mcnemar

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
README = ROOT / "README.md"
MODEL = "gpt-5.4-mini"

ROWS = [
    ("graph_only_position", "Earliest plausible step (no graph, no LLM)"),
    ("graph_only_prior", "Step-type prior only (no graph, no LLM)"),
    (f"baseline_{MODEL}", f"{MODEL} reads the whole log (official Who&When prompt)"),
    (f"baseline_numbered_{MODEL}", f"{MODEL} reads the whole log, steps numbered (strongest baseline)"),
    ("graph_only_graph", "**Butterfly, graph only (no LLM)**"),
    (f"butterfly_{MODEL}", f"**Butterfly, graph + one {MODEL} call**"),
]


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def mcnemar(a: dict, b: dict, key: str) -> tuple[int, int, float]:
    r = _mcnemar(a["rows"], b["rows"], key)
    return r["only_a"], r["only_b"], r["p"]


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render() -> str:
    lines = [
        "## Results on all 184 Who&When failures",
        "",
        f"Same model ({MODEL}) for every LLM method, no ground truth in any prompt, exact-step scoring, "
        "95% bootstrap confidence intervals. Reproduce with the commands under *Quickstart*.",
        "",
        "| Method | Right step | 95% CI | Right agent | LLM input per run |",
        "|---|---|---|---|---|",
    ]
    for name, label in ROWS:
        data = load(name)
        if not data:
            continue
        m = data["report"]["all"]
        lo, hi = m["step_acc_ci95"]
        tokens = f"{m['prompt_tokens_mean'] / 1000:.1f}k tokens" if "prompt_tokens_mean" in m else "none"
        lines.append(f"| {label} | {pct(m['step_acc'])} | {pct(lo)}–{pct(hi)} | {pct(m['agent_acc'])} | {tokens} |")

    ours, strong, official = load(f"butterfly_{MODEL}"), load(f"baseline_numbered_{MODEL}"), load(f"baseline_{MODEL}")
    if ours and strong:
        s_a, s_b, s_p = mcnemar(ours, strong, "step_hit")
        a_a, a_b, a_p = mcnemar(ours, strong, "agent_hit")
        saved = 1 - ours["report"]["all"]["prompt_tokens_mean"] / strong["report"]["all"]["prompt_tokens_mean"]
        hc_saved = 1 - ours["report"]["hand_crafted"]["prompt_tokens_mean"] / strong["report"]["hand_crafted"]["prompt_tokens_mean"]
        lines += [
            "",
            "What this shows, against the strongest baseline (paired exact McNemar tests on the same 184 runs):",
            "",
            f"- **Finding the step: a tie.** Butterfly {pct(ours['report']['all']['step_acc'])} vs "
            f"{pct(strong['report']['all']['step_acc'])} (right only for Butterfly {s_a}, only for the baseline {s_b}, p = {s_p:.2f}).",
            f"- **Naming the responsible agent: Butterfly is better.** {pct(ours['report']['all']['agent_acc'])} vs "
            f"{pct(strong['report']['all']['agent_acc'])} ({a_a} vs {a_b}, p = {a_p:.2f}).",
            f"- **Cost: {saved:.0%} fewer LLM input tokens**, and {hc_saved:.0%} fewer on the long hand-crafted runs.",
            "- **Proof:** every Butterfly answer comes with the dependency path from the failure back to the mistake. "
            "The baseline gives prose.",
            "- **Without any LLM**, the graph alone finds the step "
            f"{pct(load('graph_only_graph')['report']['all']['step_acc'])} of the time.",
        ]
    if ours and official:
        o_a, o_b, o_p = mcnemar(ours, official, "step_hit")
        lines += [
            "",
            f"Against the benchmark's official prompt, Butterfly roughly doubles step accuracy ({o_a} vs {o_b}, p = {o_p:.1g}). "
            "Most of that gap comes from the official prompt asking the model to count steps itself: the same model is right "
            "within one step far more often than exactly. That is why the numbered baseline above is the fair comparison.",
        ]
    lines += ["", "Per split:", "", "| Split | Method | Right step | Right agent | LLM input per run |", "|---|---|---|---|---|"]
    for split, split_label in (("algorithm_generated", "Generated (126 runs, ≤10 steps)"), ("hand_crafted", "Hand-crafted (58 runs, up to 130 steps)")):
        for name, label in ((f"baseline_numbered_{MODEL}", "Strongest baseline"), ("graph_only_graph", "Butterfly, graph only"), (f"butterfly_{MODEL}", "Butterfly")):
            data = load(name)
            if not data:
                continue
            m = data["report"][split]
            tokens = f"{m['prompt_tokens_mean'] / 1000:.1f}k" if "prompt_tokens_mean" in m else "none"
            lines.append(f"| {split_label} | {label} | {pct(m['step_acc'])} | {pct(m['agent_acc'])} | {tokens} |")
    return "\n".join(lines)


def main() -> None:
    text = README.read_text()
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    README.write_text(f"{before}{start}\n{render()}\n{end}{after}")
    print(render())


if __name__ == "__main__":
    main()
