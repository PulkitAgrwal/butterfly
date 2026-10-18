"""The Butterfly pipeline: graph ranks suspects, one LLM call adjudicates, the graph supplies the proof path."""

import json
import re
from dataclasses import dataclass, field

from falkordb import FalkorDB

from butterfly import llm, prompts, store
from butterfly import queries as q
from butterfly.dataset import Run
from butterfly.rank import Priors, Suspect, rank_suspects

TOP_K = {"algorithm_generated": 5, "hand_crafted": 10}


@dataclass
class Diagnosis:
    run_id: str
    method: str
    agent: str
    step: int
    reason: str
    ranked_steps: list[int]
    suspects: list[Suspect] = field(default_factory=list)
    blame_path: list[int] = field(default_factory=list)
    blame_via: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


def blame_path(db: FalkorDB, run_id: str, idx: int) -> tuple[list[int], list[str]]:
    rows = store.read(db, run_id, q.BLAME_PATH, {"idx": idx})
    if not rows or not rows[0]["steps"]:
        return [], []
    return list(rows[0]["steps"]), list(rows[0]["via"])


def _known_agent(run: Run, name: str) -> str | None:
    name = re.sub(r"\s*\(.*\)\s*$", "", name or "").strip().lower()
    for step in run.steps:
        if step.agent.lower() == name:
            return step.agent
    return None


def _parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def diagnose(db: FalkorDB, run: Run, priors: Priors, use_llm: bool = True) -> Diagnosis:
    ranked = rank_suspects(db, run, priors)
    top = ranked[: TOP_K[run.split]]
    paths = {s.idx: blame_path(db, run.run_id, s.idx)[0] for s in top}
    agent, step, reason = top[0].agent, top[0].idx, "; ".join(top[0].reasons)
    prompt_tokens = completion_tokens = 0
    method = "butterfly-graph-only"

    if use_llm:
        method = "butterfly"
        out = llm.complete(prompts.SYSTEM, prompts.butterfly_prompt(run, top, paths), tag=f"butterfly-{run.run_id}")
        prompt_tokens, completion_tokens = out.prompt_tokens, out.completion_tokens
        parsed = _parse_json(out.text) or {}
        if isinstance(parsed.get("step"), int) and 0 <= parsed["step"] < len(run.steps):
            step = parsed["step"]
            agent = _known_agent(run, parsed.get("agent", "")) or run.steps[step].agent
            reason = str(parsed.get("reason", ""))

    path, via = blame_path(db, run.run_id, step)
    return Diagnosis(
        run_id=run.run_id,
        method=method,
        agent=agent,
        step=step,
        reason=reason,
        ranked_steps=[s.idx for s in ranked],
        suspects=top,
        blame_path=path,
        blame_via=via,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


_AGENT_RE = re.compile(r"Agent Name:\s*([\w_]+)")
_STEP_RE = re.compile(r"Step Number:\s*(\d+)")


def baseline(run: Run, numbered: bool = False) -> Diagnosis:
    """Official all-at-once method: the whole log in one prompt, same model, no ground truth."""
    tag = f"baseline{'-numbered' if numbered else ''}-{run.run_id}"
    out = llm.complete(prompts.SYSTEM, prompts.baseline_prompt(run, numbered), tag=tag)
    text = out.text.replace("*", "")
    agent_m, step_m = _AGENT_RE.search(text), _STEP_RE.search(text)
    step = int(step_m.group(1)) if step_m else -1
    return Diagnosis(
        run_id=run.run_id,
        method="all-at-once-numbered" if numbered else "all-at-once",
        agent=agent_m.group(1) if agent_m else "",
        step=step,
        reason=text[-400:],
        ranked_steps=[step],
        prompt_tokens=out.prompt_tokens,
        completion_tokens=out.completion_tokens,
    )
