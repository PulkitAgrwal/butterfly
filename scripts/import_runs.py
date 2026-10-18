"""Build every run's dependency graph and load it into FalkorDB (safe to re-run)."""

import time

from butterfly import store
from butterfly.dataset import load_runs
from butterfly.graph import build


def main() -> None:
    db = store.connect()
    runs = load_runs()
    start = time.time()
    steps = edges = 0
    for run in runs:
        rg = build(run)
        store.write_run(db, rg)
        steps += len(rg.nodes)
        edges += len(rg.edges)
    print(f"imported {len(runs)} runs ({steps} steps, {edges} dependency edges) in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
