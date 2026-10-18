"""Turn a Run into a dependency graph: which step relied on which earlier step, and why."""

import re
from dataclasses import dataclass, field

from butterfly.dataset import Run, Step, is_candidate
from butterfly.entities import extract, given_entities

_TOOL_ERROR = re.compile(r"exitcode: [1-9]|Traceback \(most recent call last\)|Error:", re.IGNORECASE)
MAX_VIA = 8  # entities kept per edge, for display


@dataclass
class StepNode:
    idx: int
    agent: str
    speaker: str
    kind: str
    candidate: bool
    rel_pos: float
    chars: int
    preview: str
    new_entities: int
    tool_error_next: bool
    is_final: bool


@dataclass
class DepEdge:
    src: int  # the later step that relied on ...
    dst: int  # ... this earlier step
    via: str  # entity | instruction | plan | observes | executes | tool_result
    weight: int = 1
    entities: list[str] = field(default_factory=list)


@dataclass
class RunGraph:
    run: Run
    nodes: list[StepNode]
    edges: list[DepEdge]
    introduced: dict[str, tuple[int, str]]  # entity -> (step that introduced it, kind)


def _control_edges(steps: list[Step]) -> list[DepEdge]:
    """Edges implied by who-talks-to-whom, independent of content."""
    edges: list[DepEdge] = []
    last_plan = last_action = last_tool = None
    pending_instruction: dict[str, int] = {}
    for step in steps:
        i = step.idx
        if step.kind == "instruction":
            if last_plan is not None:
                edges.append(DepEdge(i, last_plan, "plan"))
            pending_instruction[step.target or ""] = i
        elif step.kind in {"ledger", "thought"}:
            if last_action is not None:
                edges.append(DepEdge(i, last_action, "observes"))
            last_plan = i
        elif step.kind == "action":
            if step.agent in pending_instruction:
                edges.append(DepEdge(i, pending_instruction.pop(step.agent), "instruction"))
            if last_tool is not None and last_tool == i - 1:
                edges.append(DepEdge(i, last_tool, "tool_result"))
            last_action = i
        elif step.kind == "tool_output":
            if last_action is not None:
                edges.append(DepEdge(i, last_action, "executes"))
            last_tool = i
    return edges


def build(run: Run) -> RunGraph:
    given = given_entities(run.question)
    introduced: dict[str, tuple[int, str]] = {}
    new_counts: dict[int, int] = {}
    ref: dict[tuple[int, int], list[str]] = {}

    for step in run.steps:
        if step.kind == "stub":
            continue
        ents = extract(step.text)
        new_counts[step.idx] = 0
        for ent, kind in ents.items():
            if ent in given:
                continue
            if ent not in introduced:
                introduced[ent] = (step.idx, kind)
                new_counts[step.idx] += 1
            elif introduced[ent][0] != step.idx:
                ref.setdefault((step.idx, introduced[ent][0]), []).append(ent)

    edges = [DepEdge(s, d, "entity", len(v), v[:MAX_VIA]) for (s, d), v in ref.items()]
    edges += _control_edges(run.steps)

    last = len(run.steps) - 1
    nodes = []
    for step in run.steps:
        nxt = run.steps[step.idx + 1] if step.idx < last else None
        nodes.append(
            StepNode(
                idx=step.idx,
                agent=step.agent,
                speaker=step.speaker,
                kind=step.kind,
                candidate=is_candidate(step),
                rel_pos=step.idx / last if last else 0.0,
                chars=len(step.text),
                preview=step.text[:400],
                new_entities=new_counts.get(step.idx, 0),
                tool_error_next=bool(nxt and nxt.kind == "tool_output" and _TOOL_ERROR.search(nxt.text)),
                is_final=step.idx == last,
            )
        )
    return RunGraph(run, nodes, edges, introduced)
