"""Butterfly Briefing: a short spoken account of a diagnosis, one audio clip per sentence."""

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from butterfly.attribute import Diagnosis
from butterfly.dataset import Run
from butterfly.present import role

AUDIO_DIR = Path(__file__).resolve().parent.parent / "results" / "audio"

_VIA_WORDS = {
    "entity": "a fact",
    "instruction": "an instruction",
    "plan": "the plan",
    "observes": "a report",
    "executes": "code",
    "tool_result": "a tool result",
}


@dataclass
class Segment:
    text: str
    steps: list[int] = field(default_factory=list)  # steps to light up while this sentence plays
    audio: str | None = None  # file name under AUDIO_DIR


def _short(text: str, words: int) -> str:
    parts = text.split()
    return " ".join(parts[:words]) + ("…" if len(parts) > words else "")


def script(run: Run, d: Diagnosis) -> list[Segment]:
    final = len(run.steps) - 1
    who = f"the {role(d.agent)['name'].lower()}"
    why = re.split(r"(?<=[.!?])\s", d.reason.strip(), maxsplit=1)[0] if d.prompt_tokens else ""
    segments = [
        Segment(f"This run failed. A user asked: {_short(run.question, 22)}"),
        Segment(f"The mistake was at step {d.step}, by {who}. {why}".strip(), [d.step]),
    ]
    if len(d.blame_path) >= 2:
        hops = list(reversed(d.blame_path))  # butterfly first, failure last
        via = _VIA_WORDS.get(d.blame_via[-1], "its output")
        if len(hops) > 2:
            text = f"From there it spread through step {hops[1]} to step {final}, where the run ended."
        else:
            text = f"Step {final}, where the run ended, relied directly on {via} from step {hops[0]}."
        segments.append(Segment(text, hops))
    how = (
        "then asking a language model about only the top suspects."
        if d.prompt_tokens
        else "using graph queries alone, with no language model."
    )
    segments.append(Segment(f"Butterfly found it by ranking every step in a FalkorDB graph, {how}"))
    return segments


def _elevenlabs(text: str) -> bytes:
    voice = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
    resp = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
        params={"output_format": "mp3_44100_128"},
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
        json={"text": text, "model_id": os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.content


def _cartesia(text: str) -> bytes:
    resp = httpx.post(
        "https://api.cartesia.ai/tts/bytes",
        headers={
            "Authorization": f"Bearer {os.environ['CARTESIA_API_KEY']}",
            "Cartesia-Version": os.getenv("CARTESIA_VERSION", "2026-08-14"),
        },
        json={
            "model_id": os.getenv("CARTESIA_MODEL", "sonic-3"),
            "transcript": text,
            "voice": {"id": os.getenv("CARTESIA_VOICE_ID", "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4")},
            "output_format": {"container": "mp3", "sample_rate": 44100, "bit_rate": 128000},
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.content


def provider() -> str | None:
    if os.getenv("ELEVENLABS_API_KEY"):
        return "elevenlabs"
    if os.getenv("CARTESIA_API_KEY"):
        return "cartesia"
    return None


def synthesize(text: str) -> str:
    """Return the cached clip's file name, generating it on first use."""
    name = provider()
    digest = hashlib.sha256(text.encode()).hexdigest()[:20]
    if cached := _cached(text, name):
        return cached
    if name is None:
        raise RuntimeError("Add ELEVENLABS_API_KEY or CARTESIA_API_KEY to .env to enable the briefing.")
    file = f"{name}-{digest}.mp3"
    path = AUDIO_DIR / file
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_elevenlabs(text) if name == "elevenlabs" else _cartesia(text))
    return file


def _cached(text: str, name: str | None) -> str | None:
    """A clip already on disk; committed clips play without a key."""
    digest = hashlib.sha256(text.encode()).hexdigest()[:20]
    for prov in [name] if name else ["elevenlabs", "cartesia"]:
        if (AUDIO_DIR / f"{prov}-{digest}.mp3").exists():
            return f"{prov}-{digest}.mp3"
    return None


def available(run: Run, d: Diagnosis) -> bool:
    """True when a TTS key is set or every clip of this briefing is already cached."""
    return provider() is not None or all(_cached(s.text, None) for s in script(run, d))


def briefing(run: Run, d: Diagnosis) -> list[Segment]:
    segments = script(run, d)
    for seg in segments:
        seg.audio = synthesize(seg.text)
    return segments
