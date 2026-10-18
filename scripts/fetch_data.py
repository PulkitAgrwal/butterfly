"""Download the Who&When benchmark (MIT, ag2ai/Agents_Failure_Attribution) into data/whowhen/."""

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "Kevin355/Who_and_When"
TREE_URL = f"https://huggingface.co/api/datasets/{DATASET}/tree/main?recursive=true"
FILE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main/"
SPLITS = {"Hand-Crafted": "hand_crafted", "Algorithm-Generated": "algorithm_generated"}
OUT = Path(__file__).resolve().parent.parent / "data" / "whowhen"


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def main() -> None:
    entries = json.loads(get(TREE_URL))
    paths = [e["path"] for e in entries if e.get("type") == "file" and e["path"].endswith(".json")]
    count = 0
    for path in paths:
        parts = path.split("/")  # Who&When/<Split>/<n>.json
        if len(parts) != 3 or parts[1] not in SPLITS:
            continue
        dest = OUT / SPLITS[parts[1]] / parts[2]
        if dest.exists():
            count += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(get(FILE_URL + urllib.parse.quote(path)))
        count += 1
    print(f"{count} runs in {OUT}")
    if count != 184:
        sys.exit(f"expected 184 runs, got {count}")


if __name__ == "__main__":
    main()
