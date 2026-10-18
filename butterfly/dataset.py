"""Load Who&When failed runs into one normalized shape for both splits."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "whowhen"
SPLITS = ("hand_crafted", "algorithm_generated")
SPLIT_PREFIX = {"hand_crafted": "hc", "algorithm_generated": "ag"}

_INSTRUCTION = re.compile(r"^Orchestrator \(-> (?P<target>[^)]+)\)$")


@dataclass(frozen=True)
class Step:
    idx: int  # 0-based position in the log; this is what the benchmark labels
    speaker: str  # raw speaker label from the log
    agent: str  # normalized agent name (what we predict)
    kind: str  # task | instruction | thought | ledger | stub | termination | action | tool_output
    target: str | None  # for Orchestrator instructions: which agent is being instructed
    text: str


@dataclass(frozen=True)
class Run:
    run_id: str  # e.g. "hc-12", "ag-7"
    split: str
    question: str
    ground_truth: str
    steps: list[Step]
    mistake_agent: str
    mistake_step: int
    mistake_reason: str


def is_candidate(step: Step) -> bool:
    """Steps that can plausibly be the decisive mistake (filters prompts, stubs, raw tool output)."""
    return step.kind not in {"task", "stub", "termination", "tool_output"}


def _hand_crafted_step(idx: int, entry: dict) -> Step:
    speaker, text = entry["role"], entry["content"]
    if speaker == "human":
        return Step(idx, speaker, "human", "task", None, text)
    if match := _INSTRUCTION.match(speaker):
        return Step(idx, speaker, "Orchestrator", "instruction", match["target"], text)
    if speaker == "Orchestrator (termination condition)":
        return Step(idx, speaker, "Orchestrator", "termination", None, text)
    if speaker == "Orchestrator (thought)":
        stripped = text.strip()
        if stripped.startswith("Next speaker"):
            kind = "stub"
        elif "Updated Ledger" in stripped:
            kind = "ledger"
        else:
            kind = "thought"
        return Step(idx, speaker, "Orchestrator", kind, None, text)
    return Step(idx, speaker, speaker, "action", None, text)


def _algorithm_generated_step(idx: int, entry: dict) -> Step:
    name, text = entry.get("name") or entry["role"], entry["content"]
    kind = "tool_output" if name == "Computer_terminal" else "action"
    return Step(idx, name, name, kind, None, text)


def load_run(path: Path) -> Run:
    split = path.parent.name
    raw = json.loads(path.read_text())
    make_step = _hand_crafted_step if split == "hand_crafted" else _algorithm_generated_step
    steps = [make_step(i, entry) for i, entry in enumerate(raw["history"])]
    return Run(
        run_id=f"{SPLIT_PREFIX[split]}-{path.stem}",
        split=split,
        question=raw["question"],
        ground_truth=str(raw.get("groundtruth") or raw.get("ground_truth") or ""),
        steps=steps,
        mistake_agent=raw["mistake_agent"],
        mistake_step=int(raw["mistake_step"]),
        mistake_reason=raw.get("mistake_reason", ""),
    )


def load_runs(split: str | None = None, data_dir: Path = DATA_DIR) -> list[Run]:
    splits = [split] if split else list(SPLITS)
    runs = []
    for name in splits:
        paths = sorted((data_dir / name).glob("*.json"), key=lambda p: int(p.stem))
        runs.extend(load_run(p) for p in paths)
    return runs


def load_run_by_id(run_id: str, data_dir: Path = DATA_DIR) -> Run:
    prefix, num = run_id.split("-", 1)
    split = {v: k for k, v in SPLIT_PREFIX.items()}[prefix]
    return load_run(data_dir / split / f"{num}.json")
