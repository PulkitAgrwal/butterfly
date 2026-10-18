"""FalkorDB access: one graph per run, idempotent writes, read-only analysis with timeouts."""

import os
from dataclasses import asdict

from falkordb import FalkorDB, Graph

from butterfly import queries as q
from butterfly.graph import RunGraph

QUERY_TIMEOUT_MS = 3000
GRAPH_PREFIX = "butterfly:"


def connect() -> FalkorDB:
    return FalkorDB(
        host=os.getenv("FALKORDB_HOST", "localhost"),
        port=int(os.getenv("FALKORDB_PORT", "6380")),
        password=os.getenv("FALKORDB_PASSWORD") or None,
    )


def graph_name(run_id: str) -> str:
    return GRAPH_PREFIX + run_id


def run_graph(db: FalkorDB, run_id: str) -> Graph:
    return db.select_graph(graph_name(run_id))


def write_run(db: FalkorDB, rg: RunGraph) -> None:
    g = run_graph(db, rg.run.run_id)
    for stmt in q.INDEXES:
        try:
            g.query(stmt)
        except Exception as exc:  # index already exists on re-import
            if "already" not in str(exc).lower():
                raise
    g.query(
        q.MERGE_RUN,
        {"id": rg.run.run_id, "split": rg.run.split, "question": rg.run.question, "n_steps": len(rg.nodes)},
    )
    g.query(q.MERGE_STEPS, {"steps": [asdict(n) for n in rg.nodes]})
    g.query(q.MERGE_NEXT, {"pairs": [[i, i + 1] for i in range(len(rg.nodes) - 1)]})
    entities = [{"key": k, "step": s, "kind": kind} for k, (s, kind) in rg.introduced.items()]
    if entities:
        g.query(q.MERGE_ENTITIES, {"entities": entities})
    if rg.edges:
        g.query(q.MERGE_DEPENDENCIES, {"edges": [asdict(e) for e in rg.edges]})


def read(db: FalkorDB, run_id: str, query: str, params: dict | None = None) -> list[dict]:
    """Read-only query (GRAPH.RO_QUERY) with a hard timeout; rows come back as dicts."""
    res = run_graph(db, run_id).ro_query(query, params or {}, timeout=QUERY_TIMEOUT_MS)
    cols = [c[1] if isinstance(c, (list, tuple)) else c for c in res.header]
    return [dict(zip(cols, row)) for row in res.result_set]
