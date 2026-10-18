"""Export a recorded, server-free copy of the site (for GitHub Pages): every API answer saved as JSON.

Usage: python scripts/export_static.py <out_dir>
Runs against the local FalkorDB and the committed LLM cache. TTS keys are blanked so only
briefings whose audio is already committed are offered.
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

for key in ("OPENAI_API_KEY", "ELEVENLABS_API_KEY", "CARTESIA_API_KEY"):
    os.environ[key] = ""  # blank before import so .env cannot fill them in

from fastapi.testclient import TestClient  # noqa: E402

from butterfly.api import WEB_DIR, app  # noqa: E402
from butterfly.voice import AUDIO_DIR  # noqa: E402

SHIM = """<script>
// Recorded copy: serve the API from saved JSON files instead of a server.
(() => {
  const realFetch = window.fetch.bind(window);
  const steps = {};
  const json = (o) => new Response(JSON.stringify(o), { headers: { "content-type": "application/json" } });
  const routes = [
    [/^\\/api\\/(runs|results|benchmark)$/, (m) => `data/${m[1]}.json`],
    [/^\\/api\\/runs\\/([\\w-]+)\\/graph$/, (m) => `data/runs/${m[1]}/graph.json`],
    [/^\\/api\\/runs\\/([\\w-]+)\\/(diagnose|briefing)\\?llm=(true|false)$/, (m) => `data/runs/${m[1]}/${m[2]}-${m[3]}.json`],
  ];
  window.fetch = async (input, init) => {
    const url = typeof input === "string" ? input : input.url;
    const step = url.match(/^\\/api\\/runs\\/([\\w-]+)\\/step\\/(\\d+)$/);
    if (step) {
      steps[step[1]] ??= realFetch(`data/runs/${step[1]}/steps.json`).then((r) => r.json());
      const all = await steps[step[1]];
      return all[+step[2]] ? json(all[+step[2]]) : new Response("{}", { status: 404 });
    }
    for (const [re, to] of routes) {
      const m = url.match(re);
      if (m) return realFetch(to(m), init);
    }
    return realFetch(input, init);
  };
})();
</script>"""

NOTE = """<div style="position:fixed;left:16px;bottom:16px;z-index:30;max-width:calc(100% - 32px);padding:7px 12px;border-radius:999px;background:#fff;border:1px solid #e6e8ed;box-shadow:0 6px 20px rgba(12,14,18,.08);font:12.5px/1.4 Geist,system-ui,sans-serif;color:#4a5260">Recorded from FalkorDB. For live queries, run <code style="font-family:'Geist Mono',monospace">docker compose up</code> from the <a href="https://github.com/PulkitAgrwal/butterfly" style="color:#0c0e12">repo</a>.</div>"""


def page(name: str) -> str:
    html = (WEB_DIR / name).read_text()
    html = html.replace("<head>", "<head>\n" + SHIM, 1)
    html = html.replace('href="/app', 'href="app.html').replace('href="/"', 'href="./"')
    html = html.replace("`/audio/${", "`audio/${")
    return html.replace("</body>", NOTE + "\n</body>", 1)


def main(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    (out / "data" / "runs").mkdir(parents=True)
    (out / "audio").mkdir()
    client = TestClient(app)

    def get(path: str):
        res = client.get(path)
        return res.json() if res.status_code == 200 else None

    def save(rel: str, obj) -> None:
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, separators=(",", ":")))

    runs = get("/api/runs")
    save("data/runs.json", runs)
    save("data/results.json", get("/api/results"))
    save("data/benchmark.json", get("/api/benchmark"))
    clips = set()
    for r in runs:
        rid = r["run_id"]
        graph = get(f"/api/runs/{rid}/graph")
        save(f"data/runs/{rid}/graph.json", graph)
        save(f"data/runs/{rid}/steps.json", [get(f"/api/runs/{rid}/step/{i}") for i in range(len(graph["steps"]))])
        for llm in ("true", "false"):
            diag = get(f"/api/runs/{rid}/diagnose?llm={llm}")
            save(f"data/runs/{rid}/diagnose-{llm}.json", diag)
            if diag.get("briefing") and (brief := get(f"/api/runs/{rid}/briefing?llm={llm}")):
                save(f"data/runs/{rid}/briefing-{llm}.json", brief)
                clips.update(s["audio"] for s in brief["segments"])
    for clip in clips:
        shutil.copy(AUDIO_DIR / clip, out / "audio" / clip)
    for name in ("index.html", "app.html"):
        (out / name).write_text(page(name))
    (out / ".nojekyll").write_text("")
    leftover = [p for p in ("index.html", "app.html") if re.search(r"""(href|src)=["']/(?!/)""", (out / p).read_text())]
    print(f"exported {len(runs)} runs, {len(clips)} audio clips to {out}" + (f"; absolute links left in {leftover}" if leftover else ""))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
