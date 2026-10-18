"""Rank suspect steps from graph features computed in FalkorDB (no LLM involved)."""

import math
from collections import Counter
from dataclasses import dataclass, field

from falkordb import FalkorDB

from butterfly import queries as q
from butterfly import store
from butterfly.dataset import Run, is_candidate


@dataclass
class Suspect:
    idx: int
    agent: str
    kind: str
    score: float
    downstream: int
    reaches_final: bool
    betweenness: float
    new_entities: int
    rel_pos: float
    first_turn: bool
    tool_error_next: bool
    reasons: list[str] = field(default_factory=list)


def prior_key(run: Run, idx: int) -> str:
    """What kind of step this is, in terms that generalize across runs of the same system."""
    step = run.steps[idx]
    first_turn = all(s.agent != step.agent for s in run.steps[:idx])
    role = f"{step.agent}:{step.kind}" if run.split == "hand_crafted" else step.kind
    return f"{role}|first={first_turn}"


class Priors:
    """P(step is the decisive mistake | step type), estimated leave-one-out over the other runs."""

    def __init__(self, runs: list[Run]):
        self.gold: dict[str, Counter] = {}
        self.seen: dict[str, Counter] = {}
        self.per_run: dict[str, tuple[Counter, Counter]] = {}
        for run in runs:
            gold, seen = Counter(), Counter()
            for step in run.steps:
                if is_candidate(step):
                    seen[prior_key(run, step.idx)] += 1
            if run.mistake_step < len(run.steps):
                gold[prior_key(run, run.mistake_step)] += 1
            self.per_run[run.run_id] = (gold, seen)
            self.gold.setdefault(run.split, Counter()).update(gold)
            self.seen.setdefault(run.split, Counter()).update(seen)

    def get(self, run: Run, key: str) -> float:
        gold_self, seen_self = self.per_run.get(run.run_id, (Counter(), Counter()))
        gold = self.gold[run.split][key] - gold_self[key]
        seen = self.seen[run.split][key] - seen_self[key]
        return (gold + 1) / (seen + 10)  # smoothed rate; unseen types get a small floor


def graph_features(db: FalkorDB, run_id: str) -> dict[int, dict]:
    feats = {r["idx"]: dict(r) for r in store.read(db, run_id, q.CANDIDATES)}
    for r in store.read(db, run_id, q.DOWNSTREAM):
        feats[r["idx"]].update(downstream=r["downstream"], reaches_final=bool(r["reaches_final"]))
    for r in store.read(db, run_id, q.BETWEENNESS):
        if r["idx"] in feats:
            feats[r["idx"]]["betweenness"] = float(r["score"] or 0.0)
    return feats


def score(f: dict, prior: float, mode: str = "graph") -> float:
    early = math.exp(-f["rel_pos"])
    if mode == "position":  # baseline: earliest plausible step wins
        return early
    if mode == "prior":  # baseline: step-type prior and earliness, no graph
        return prior * early
    return prior * early * math.log(2 + f.get("downstream", 0)) * math.log(2 + f["new_entities"])


def _reasons(f: dict) -> list[str]:
    out = []
    if f["new_entities"]:
        out.append(f"introduced {f['new_entities']} new entities")
    if f.get("downstream"):
        out.append(f"{f['downstream']} later steps build on it")
    if f.get("reaches_final"):
        out.append("the final step depends on it")
    if f.get("tool_error_next"):
        out.append("the next tool execution failed")
    return out


def rank_suspects(db: FalkorDB, run: Run, priors: Priors, mode: str = "graph") -> list[Suspect]:
    feats = graph_features(db, run.run_id)
    suspects = []
    for idx, f in feats.items():
        key = prior_key(run, idx)
        suspects.append(
            Suspect(
                idx=idx,
                agent=f["agent"],
                kind=f["kind"],
                score=score(f, priors.get(run, key), mode),
                downstream=f.get("downstream", 0),
                reaches_final=f.get("reaches_final", False),
                betweenness=f.get("betweenness", 0.0),
                new_entities=f["new_entities"],
                rel_pos=f["rel_pos"],
                first_turn=key.endswith("first=True"),
                tool_error_next=bool(f["tool_error_next"]),
                reasons=_reasons(f),
            )
        )
    return sorted(suspects, key=lambda s: (-s.score, s.idx))
