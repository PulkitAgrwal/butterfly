# Demo video script (about 2.5 minutes)

Record at 1512×900 or larger, browser zoom 100%, `docker compose up -d` already running, page open at `http://localhost:8000/#hc-1`.

| Time | Screen | What you say |
|---|---|---|
| 0:00–0:15 | Scroll the raw log of hc-1 in the step panel (click step 28, then 20, then 12 quickly) | "When a team of AI agents fails, this is what you get: a long log. Which agent broke it, and where? People scroll. LLMs struggle once logs get long." |
| 0:15–0:35 | The chart for hc-1: lanes, edges | "Butterfly turns every failed run into a graph in FalkorDB. Each line means a later step relied on an earlier one: it reused a URL, a file, a number, or followed an instruction. No LLM builds this graph." |
| 0:35–0:55 | Click **Find the butterfly** (LLM on). The storm track draws. | "FalkorDB ranks the suspects, using BFS to see how much of the run builds on each step. One LLM call reads only those suspects. Here: WebSurfer, step 12. It clicked an ad for laser microscopes instead of the dojo." |
| 0:55–1:10 | Point at the green **Matches the benchmark label** badge and the storm track list | "That matches the benchmark's human label. And the red path is the proof: the final step depends on step 12. It's a shortestPath query, not a guess." |
| 1:10–1:30 | Click **Play the briefing** (if the voice key is set) | Let the briefing play. |
| 1:30–1:45 | Open the two **Cypher** sections in the right panel | "Every query is read-only with a timeout, and you can see exactly what ran." |
| 1:45–2:15 | The scoreboard in the left panel | "On all 184 failures in Who&When, same model, no answers in any prompt: Butterfly finds the step as often as the strongest whole-log method, names the responsible agent more often, 62 versus 54 percent, and sends the model half the tokens. With no LLM at all, the graph alone gets 28 percent." |
| 2:15–2:30 | Switch the run picker to hc-2 (86 steps), click Find the butterfly | "Long runs are where the graph pays off most: 58 percent fewer tokens. Butterfly: find the flap that caused the storm." |

Tips:
- Keep the cursor still while the track animates.
- Say numbers exactly as the scoreboard shows them; judges will check.
- End on the full screen with the logo visible.
