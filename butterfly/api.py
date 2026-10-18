"""HTTP API + static UI: list runs, return a run's graph, and diagnose it."""

import json
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from butterfly import queries as q
from butterfly import store, voice
from butterfly.attribute import blame_path, diagnose
from butterfly.dataset import load_runs
from butterfly.present import clean, role, summarize
from butterfly.rank import Priors
from butterfly.stats import mcnemar

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SYSTEMS = {"hand_crafted": "Magentic-One", "algorithm_generated": "CaptainAgent"}
WHOLE_LOG_BASELINE = "baseline_numbered_gpt-5.4-mini"

app = FastAPI(title="Butterfly")
_runs = {r.run_id: r for r in load_runs()}
_priors = Priors(list(_runs.values()))


def _get_run(run_id: str):
    if run_id not in _runs:
        raise HTTPException(404, f"unknown run {run_id}")
    return _runs[run_id]


def _baseline_rows() -> dict[str, dict]:
    path = RESULTS_DIR / f"{WHOLE_LOG_BASELINE}.json"
    if not path.exists():
        return {}
    return {r["run_id"]: r for r in json.loads(path.read_text())["rows"]}


_whole_log = _baseline_rows()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/app")
def app_page() -> FileResponse:
    return FileResponse(WEB_DIR / "app.html")


@app.get("/api/runs")
def list_runs() -> list[dict]:
    return [
        {
            "run_id": r.run_id,
            "split": r.split,
            "system": SYSTEMS[r.split],
            "steps": len(r.steps),
            "question": r.question[:160],
        }
        for r in _runs.values()
    ]


@app.get("/api/results")
def results() -> list[dict]:
    """Benchmark summaries written by `python -m butterfly.evaluate` (pilot runs excluded)."""
    out = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.stem.endswith("_pilot"):
            continue
        data = json.loads(path.read_text())
        out.append({"name": path.stem, "method": data["method"], "report": data["report"]})
    return out


@app.get("/api/benchmark")
def benchmark() -> dict:
    """Headline comparison of Butterfly against the strongest whole-log baseline, with paired tests."""

    def load(name: str) -> dict | None:
        path = RESULTS_DIR / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    ours, base = load("butterfly_gpt-5.4-mini"), load(WHOLE_LOG_BASELINE)
    if not ours or not base:
        return {}
    return {
        "model": "gpt-5.4-mini",
        "runs": ours["report"]["all"]["n"],
        "step": mcnemar(ours["rows"], base["rows"], "step_hit"),
        "agent": mcnemar(ours["rows"], base["rows"], "agent_hit"),
    }


@app.get("/api/runs/{run_id}/graph")
def run_graph(run_id: str) -> dict:
    run = _get_run(run_id)
    db = store.connect()
    steps = store.read(db, run_id, q.GRAPH_STEPS)
    for row in steps:
        row.update(summarize(run.steps[row["idx"]]))
    team = list(dict.fromkeys(s.agent for s in run.steps))
    return {
        "run_id": run_id,
        "system": SYSTEMS[run.split],
        "question": run.question,
        "ground_truth": run.ground_truth,
        "team": [role(a) for a in team],
        "steps": steps,
        "edges": store.read(db, run_id, q.GRAPH_DEPENDENCIES),
        "gold": {"agent": run.mistake_agent, "step": run.mistake_step, "reason": run.mistake_reason},
    }


@app.get("/api/runs/{run_id}/diagnose")
def run_diagnose(run_id: str, llm: bool = False) -> dict:
    run = _get_run(run_id)
    db = store.connect()
    result = diagnose(db, run, _priors, use_llm=llm)
    whole = _whole_log.get(run_id, {})
    started = time.perf_counter()
    blame_path(db, run_id, result.step)
    proof_ms = (time.perf_counter() - started) * 1000
    return {
        **asdict(result),
        "proof_ms": round(proof_ms, 1),
        "whole_log_tokens": whole.get("prompt_tokens"),
        "cypher": {"rank": [q.CANDIDATES, q.DOWNSTREAM], "proof": q.BLAME_PATH},
        "briefing": voice.available(run, result),
    }


@app.get("/api/runs/{run_id}/briefing")
def run_briefing(run_id: str, llm: bool = True) -> dict:
    run = _get_run(run_id)
    result = diagnose(store.connect(), run, _priors, use_llm=llm)
    try:
        segments = voice.briefing(run, result)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"provider": voice.provider(), "segments": [asdict(s) for s in segments]}


@app.get("/audio/{name}")
def audio(name: str) -> FileResponse:
    path = (voice.AUDIO_DIR / name).resolve()
    if path.parent != voice.AUDIO_DIR.resolve() or not path.exists():
        raise HTTPException(404, "no such clip")
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/api/runs/{run_id}/step/{idx}")
def run_step(run_id: str, idx: int) -> dict:
    run = _get_run(run_id)
    if not 0 <= idx < len(run.steps):
        raise HTTPException(404, "no such step")
    step = run.steps[idx]
    return {
        "idx": idx,
        "speaker": step.speaker,
        "agent": step.agent,
        "kind": step.kind,
        "clean": clean(step)[:12000],
        "raw": step.text[:20000],
    }
