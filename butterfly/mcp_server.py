"""MCP server: lets an AI agent ask Butterfly which step broke a failed multi-agent run.

Run over stdio: uv run python -m butterfly.mcp_server (FalkorDB must be up: docker compose up falkordb).
"""

from mcp.server.fastmcp import FastMCP

from butterfly import store
from butterfly.attribute import diagnose
from butterfly.dataset import load_runs
from butterfly.present import clean, role
from butterfly.rank import Priors

SYSTEMS = {"hand_crafted": "Magentic-One", "algorithm_generated": "CaptainAgent"}
MAX_ROWS = 200

mcp = FastMCP("butterfly")
_runs = {r.run_id: r for r in load_runs()}
_priors = Priors(list(_runs.values()))


def _run(run_id: str):
    if run_id not in _runs:
        raise ValueError(f"unknown run {run_id!r}; call list_failed_runs for valid ids")
    return _runs[run_id]


def _plain(value):
    """Graph nodes and edges become their properties so results stay JSON."""
    if hasattr(value, "properties"):
        return dict(value.properties)
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


@mcp.tool()
def list_failed_runs(system: str | None = None, limit: int = 50) -> list[dict]:
    """List failed multi-agent runs from the Who&When benchmark. system: "Magentic-One" or "CaptainAgent" (optional)."""
    runs = [r for r in _runs.values() if system is None or SYSTEMS[r.split] == system]
    return [
        {"run_id": r.run_id, "system": SYSTEMS[r.split], "steps": len(r.steps), "question": r.question[:200]}
        for r in runs[:limit]
    ]


@mcp.tool()
def diagnose_run(run_id: str, use_llm: bool = True) -> dict:
    """Find the decisive mistake in a failed run: the agent, the step, why, and the dependency chain from the
    failure back to that step when the graph has one. use_llm=False ranks with FalkorDB graph queries alone."""
    run = _run(run_id)
    d = diagnose(store.connect(), run, _priors, use_llm=use_llm)
    return {
        "run_id": run_id,
        "question": run.question,
        "agent": d.agent,
        "agent_role": role(d.agent)["name"],
        "step": d.step,
        "reason": d.reason,
        "chain_found": len(d.blame_path) > 1,
        "chain": d.blame_path,
        "chain_via": d.blame_via,
        "top_suspects": [
            {"step": s.idx, "agent": s.agent, "score": round(s.score, 4), "downstream_steps": s.downstream}
            for s in d.suspects[:5]
        ],
        "llm_prompt_tokens": d.prompt_tokens,
        "benchmark_label": {"agent": run.mistake_agent, "step": run.mistake_step},
    }


@mcp.tool()
def get_step(run_id: str, idx: int) -> dict:
    """Read one step of a run: which agent wrote it and its message, with page dumps and OCR noise removed."""
    run = _run(run_id)
    if not 0 <= idx < len(run.steps):
        raise ValueError(f"step {idx} out of range; run {run_id} has {len(run.steps)} steps")
    step = run.steps[idx]
    return {"idx": idx, "agent": step.agent, "agent_role": role(step.agent)["name"], "kind": step.kind, "text": clean(step)[:8000]}


@mcp.tool()
def query_run_graph(run_id: str, cypher: str) -> list[dict]:
    """Run a read-only Cypher query on one run's FalkorDB graph (butterfly:<run_id>), with a 3 s timeout.
    Labels: Step(idx, agent, kind, candidate, is_final, rel_pos), Agent(name), Entity(key, kind).
    Relationships: (later)-[:DEPENDS_ON {via}]->(earlier), (earlier)-[:FEEDS]->(later), NEXT, BY, INTRODUCES."""
    _run(run_id)
    rows = store.read(store.connect(), run_id, cypher)[:MAX_ROWS]
    return [{k: _plain(v) for k, v in row.items()} for row in rows]


if __name__ == "__main__":
    mcp.run()
