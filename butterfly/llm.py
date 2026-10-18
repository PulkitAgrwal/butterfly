"""Single OpenAI call wrapper with an on-disk cache, so re-running the eval never re-spends tokens."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

CACHE_DIR = Path(__file__).resolve().parent.parent / "results" / "cache"
MODEL = os.getenv("BUTTERFLY_MODEL", "gpt-5.4-mini")
MAX_OUTPUT_TOKENS = int(os.getenv("BUTTERFLY_MAX_OUTPUT_TOKENS", "6000"))


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str


_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def complete(system: str, user: str, *, tag: str, model: str = MODEL) -> Completion:
    key = hashlib.sha256(f"{model}\n{system}\n{user}".encode()).hexdigest()[:24]
    path = CACHE_DIR / model / f"{tag}-{key}.json"
    if path.exists():
        return Completion(**json.loads(path.read_text()))
    resp = client().chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_completion_tokens=MAX_OUTPUT_TOKENS,
    )
    usage = resp.usage
    out = Completion(
        text=resp.choices[0].message.content or "",
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        model=model,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out.__dict__))
    return out
