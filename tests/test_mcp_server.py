import pytest

from butterfly import mcp_server


def test_diagnose_tool_returns_the_labelled_mistake_and_chain():
    d = mcp_server.diagnose_run("hc-1")
    assert (d["agent"], d["step"], d["chain"], d["chain_found"]) == ("WebSurfer", 12, [28, 12], True)


def test_graph_query_tool_is_read_only():
    with pytest.raises(Exception):
        mcp_server.query_run_graph("hc-1", "CREATE (:Hack {x: 1})")
    assert mcp_server.query_run_graph("hc-1", "MATCH (s:Step) RETURN count(DISTINCT s.idx) AS n") == [{"n": 29}]
