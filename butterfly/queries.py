"""Every Cypher query Butterfly runs, in one place (the README links here).

Graph model, one graph per run (`butterfly:<run_id>`):
  (:Run {id, split, question, n_steps})
  (:Step {idx, agent, kind, candidate, rel_pos, chars, preview, new_entities, tool_error_next, is_final})
  (:Agent {name})
  (:Entity {key, kind})
  (:Step)-[:BY]->(:Agent)
  (:Step)-[:NEXT]->(:Step)                       conversation order
  (:Step)-[:INTRODUCES]->(:Entity)               first mention of a URL / file / number / name
  (:Step)-[:DEPENDS_ON {via, weight, entities}]->(:Step)   later step relied on earlier step
  (:Step)-[:FEEDS]->(:Step)                      same edge reversed (earlier -> later), for forward BFS
"""

# --- import (idempotent: every write is a MERGE, so re-running the import is safe) ---

INDEXES = [
    "CREATE INDEX FOR (s:Step) ON (s.idx)",
    "CREATE INDEX FOR (a:Agent) ON (a.name)",
    "CREATE INDEX FOR (e:Entity) ON (e.key)",
]

MERGE_RUN = """
MERGE (r:Run {id: $id})
SET r.split = $split, r.question = $question, r.n_steps = $n_steps
"""

MERGE_STEPS = """
UNWIND $steps AS s
MERGE (n:Step {idx: s.idx})
SET n.agent = s.agent, n.speaker = s.speaker, n.kind = s.kind, n.candidate = s.candidate,
    n.rel_pos = s.rel_pos, n.chars = s.chars, n.preview = s.preview,
    n.new_entities = s.new_entities, n.tool_error_next = s.tool_error_next, n.is_final = s.is_final
MERGE (a:Agent {name: s.agent})
MERGE (n)-[:BY]->(a)
"""

MERGE_NEXT = """
UNWIND $pairs AS p
MATCH (a:Step {idx: p[0]}), (b:Step {idx: p[1]})
MERGE (a)-[:NEXT]->(b)
"""

MERGE_ENTITIES = """
UNWIND $entities AS e
MATCH (s:Step {idx: e.step})
MERGE (x:Entity {key: e.key})
SET x.kind = e.kind
MERGE (s)-[:INTRODUCES]->(x)
"""

MERGE_DEPENDENCIES = """
UNWIND $edges AS e
MATCH (later:Step {idx: e.src}), (earlier:Step {idx: e.dst})
MERGE (later)-[d:DEPENDS_ON {via: e.via}]->(earlier)
SET d.weight = e.weight, d.entities = e.entities
MERGE (earlier)-[:FEEDS {via: e.via}]->(later)
"""

# --- analysis (read-only, always executed with a timeout) ---

# How much of the rest of the run builds on each candidate step, and does the final step?
DOWNSTREAM = """
MATCH (s:Step {candidate: true})
CALL algo.BFS(s, 200, 'FEEDS') YIELD nodes
RETURN s.idx AS idx, size(nodes) AS downstream,
       any(n IN nodes WHERE n.is_final) AS reaches_final
ORDER BY idx
"""

# Steps that many dependency chains pass through (information bottlenecks).
BETWEENNESS = """
CALL algo.betweenness({nodeLabels: ['Step'], relationshipTypes: ['DEPENDS_ON'], samplingSize: 1000})
YIELD node, score
RETURN node.idx AS idx, score
"""

CANDIDATES = """
MATCH (s:Step {candidate: true})
RETURN s.idx AS idx, s.agent AS agent, s.kind AS kind, s.rel_pos AS rel_pos,
       s.new_entities AS new_entities, s.tool_error_next AS tool_error_next
ORDER BY idx
"""

# The proof: the chain of dependencies from the failure back to the suspect.
BLAME_PATH = """
MATCH (f:Step {is_final: true}), (s:Step {idx: $idx})
WITH shortestPath((f)-[:DEPENDS_ON*]->(s)) AS p
RETURN [n IN nodes(p) | n.idx] AS steps, [r IN relationships(p) | r.via] AS via
"""

GRAPH_STEPS = """
MATCH (s:Step)
RETURN s.idx AS idx, s.agent AS agent, s.kind AS kind, s.candidate AS candidate,
       s.preview AS preview, s.chars AS chars, s.is_final AS is_final
ORDER BY idx
"""

GRAPH_DEPENDENCIES = """
MATCH (a:Step)-[d:DEPENDS_ON]->(b:Step)
RETURN a.idx AS src, b.idx AS dst, d.via AS via, d.weight AS weight, d.entities AS entities
"""
