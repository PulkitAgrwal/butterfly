"""Prompts: the official Who&When all-at-once baseline (verbatim) and Butterfly's graph-guided prompt."""

import json

from butterfly.dataset import Run
from butterfly.rank import Suspect

SYSTEM = "You are a helpful assistant skilled in analyzing conversations."

# Copied from ag2ai/Agents_Failure_Attribution Automated_FA/Lib/utils.py::all_at_once,
# with the "The Answer for the problem is" line removed (no ground truth given to any method).
_BASELINE_EXAMPLE = """
            {
                "agent a": "xx",
                "agent b": "xxxx",
                "agent c": "xxxxx",
                "agent a": "xxxxxxx"
            },
            """


def baseline_prompt(run: Run, numbered: bool = False) -> str:
    """numbered=True is a stronger variant: each step is prefixed with its index so the model never miscounts."""
    if numbered:
        chat_content = "\n".join(f"[step {s.idx}] {s.speaker}: {s.text}" for s in run.steps)
    else:
        chat_content = "\n".join(f"{s.speaker}: {s.text}" for s in run.steps)
    return (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
        f"The problem is:  {run.question}\n"
        "Identify which agent made an error, at which step, and explain the reason for the error. "
        "Here's the conversation:\n\n" + chat_content + "\n\nBased on this conversation, please predict the following:\n"
        "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
        "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
        + _BASELINE_EXAMPLE
        + "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
        "3. The reason for your prediction."
        "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: \n"
    )


SUSPECT_TEXT_CHARS = 1500  # long tool outputs (web page dumps) are clipped
STUB_CHARS = 110


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " …[clipped]"


def butterfly_prompt(run: Run, suspects: list[Suspect], paths: dict[int, list[int]]) -> str:
    focus = {s.idx for s in suspects}
    focus |= {i + d for i in list(focus) for d in (-1, 1) if 0 <= i + d < len(run.steps)}

    table = [
        {
            "step": s.idx,
            "agent": s.agent,
            "graph_rank": rank + 1,
            "evidence": s.reasons,
            "dependency_path_from_final_step": paths.get(s.idx, []),
        }
        for rank, s in enumerate(suspects)
    ]
    lines = []
    for step in run.steps:
        if step.idx in focus:
            lines.append(f"[{step.idx}] {step.speaker} (FULL): {_clip(step.text, SUSPECT_TEXT_CHARS)}")
        else:
            lines.append(f"[{step.idx}] {step.speaker}: {_clip(step.text, STUB_CHARS)}")

    return (
        "A multi-agent system failed to solve this task. Find the decisive mistake.\n\n"
        f"Task: {run.question}\n\n"
        "A dependency graph of the run (built in FalkorDB from which step reused which earlier step's "
        "URLs, files, numbers, names, instructions and tool results) ranked these suspect steps:\n"
        f"{json.dumps(table, indent=1)}\n\n"
        "The run, with official step numbers in brackets. Suspects and their neighbours are shown in full; "
        "other steps are shortened:\n" + "\n".join(lines) + "\n\n"
        "Decide which agent made the decisive mistake and at which step. The decisive mistake is the EARLIEST "
        "step whose correction would have let the run succeed. Steps that only repeat or carry forward an "
        "earlier error are not decisive. You may choose a step outside the suspect list if the evidence "
        "requires it. Use the bracketed step numbers.\n"
        'Answer with JSON only: {"agent": "<agent name exactly as shown before the colon, without parentheses>", '
        '"step": <int>, "reason": "<one sentence>"}'
    )
