"""Golden run hc-1 (the martial-arts task): pins the graph build, the import, and the proof query."""

from butterfly import queries as q
from butterfly import store
from butterfly.dataset import load_run_by_id
from butterfly.entities import extract
from butterfly.evaluate import same_agent
from butterfly.graph import build


def test_graph_shape_is_stable():
    rg = build(load_run_by_id("hc-1"))
    assert len(rg.nodes) == 29
    assert len(rg.edges) == 52
    assert rg.nodes[12].agent == "WebSurfer"
    assert rg.nodes[-1].is_final


def test_import_is_idempotent_and_proof_path_reaches_the_label():
    db = store.connect()
    rg = build(load_run_by_id("hc-1"))
    store.write_run(db, rg)
    store.write_run(db, rg)
    assert store.read(db, "hc-1", "MATCH (s:Step) RETURN count(s) AS n")[0]["n"] == 29
    assert store.read(db, "hc-1", "MATCH ()-[d:DEPENDS_ON]->() RETURN count(d) AS n")[0]["n"] == 52
    path = store.read(db, "hc-1", q.BLAME_PATH, {"idx": 12})[0]["steps"]
    assert path[0] == 28 and path[-1] == 12


def test_entities_skip_years_and_keep_urls():
    found = extract("Visited https://dojos.info/NewYork/?x=1 in 2023 and saw 4,250 reviews")
    assert "https://dojos.info/newyork/" in found
    assert "2023" not in found
    assert "4,250" in found


def test_strict_agent_match_has_no_substring_bug():
    assert same_agent("websurfer", "WebSurfer")
    assert not same_agent("Orchestrator (thought)", "Orchestrator")
