"""Readable views of raw agent messages: a one-line summary per step and a cleaned full text."""

import json
import re
from urllib.parse import urlparse

from butterfly.dataset import Step

_SCREENSHOT = re.compile(r"Here is a screenshot of \[(?P<title>[^\]]*)\]\((?P<url>[^)\s]+)\)")
_META_BLOCK = re.compile(r"The following metadata was extracted from the webpage:.*?(?=\n\n[A-Z]|\Z)", re.DOTALL)
_OCR_BLOCK = re.compile(r"Automatic OCR of the page screenshot has detected the following text:.*", re.DOTALL)
_VIEWPORT = re.compile(r"The viewport shows [^\n]*")
_CODE = re.compile(r"```(\w*)\n.*?```", re.DOTALL)
_EXIT = re.compile(r"exitcode:\s*(\d+)|exit code:\s*(\d+)", re.IGNORECASE)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _ledger(text: str) -> dict | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        data = json.loads(text[start:])
    except json.JSONDecodeError:
        return None

    def answer(key: str):
        value = data.get(key, {})
        return value.get("answer") if isinstance(value, dict) else None

    return {
        "satisfied": answer("is_request_satisfied"),
        "in_loop": answer("is_in_loop"),
        "progress": answer("is_progress_being_made"),
        "next": answer("next_speaker"),
        "instruction": answer("instruction_or_question"),
    }


ROLES = {
    "human": ("User", "user", "Asked the question"),
    "Orchestrator": ("Planner", "planner", "Plans and gives instructions"),
    "WebSurfer": ("Web browser", "browser", "Browses websites"),
    "FileSurfer": ("File reader", "file", "Reads files"),
    "Assistant": ("Coder", "code", "Writes code"),
    "ComputerTerminal": ("Terminal", "terminal", "Runs code"),
    "Computer_terminal": ("Terminal", "terminal", "Runs code"),
}


def role(agent: str) -> dict:
    """Plain-language name, icon key and one-line job for an agent, so nobody needs framework jargon."""
    if agent in ROLES:
        name, icon, job = ROLES[agent]
    else:
        name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", agent).replace("_", " ").strip()
        name = name[:1].upper() + name[1:].lower()
        icon, job = "expert", "Specialist agent"
    return {"agent": agent, "name": name, "icon": icon, "job": job}


def _the(agent: str | None) -> str:
    name = role(agent or "")["name"]
    return f"the {name.lower()}" if agent in ROLES else name


_TYPED = re.compile(r"^I typed '(?P<q>[^']*)'")
_CLICKED = re.compile(r"^I clicked '(?P<label>[^']*)'")
_SCROLL = re.compile(r"^I scrolled (?P<dir>up|down)")


def _cut(text: str, limit: int) -> str:
    """Cut at a word boundary so short phrases never end mid-word."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    head = text[: limit - 1].rsplit(" ", 1)[0].rstrip(",.;:")
    return head + "…"


def short(step: Step, limit: int = 42) -> str:
    """A verb phrase of about 40 characters, for one-line rows."""
    text = step.text.strip()
    if step.kind == "task":
        return "Asked the question"
    if step.kind == "stub":
        return f"Handed the turn to {_the(text.split()[-1])}" if text else "Handed over"
    if step.kind == "termination":
        return "Stopped the run"
    if step.kind == "ledger":
        nxt = (_ledger(text) or {}).get("next")
        return f"Re-planned: {_the(nxt)} goes next" if nxt else "Re-planned"
    if step.kind == "thought":
        return "Wrote the plan" if text.startswith("Initial plan") else _cut(text, limit)
    if step.kind == "instruction":
        ask = re.sub(r"^(please\s+)", "", text, flags=re.IGNORECASE)
        ask = ask[:1].lower() + ask[1:]
        prefix = f"Asked {_the(step.target)}: "
        return prefix + _cut(ask, max(16, limit + 6 - len(prefix)))
    if step.kind == "tool_output" or step.agent == "ComputerTerminal":
        match = _EXIT.search(text)
        code = next((g for g in (match.groups() if match else ()) if g), None)
        return f"Code failed (exit {code})" if code and code != "0" else "Ran the code"
    action = _SCREENSHOT.split(text)[0].strip() if _SCREENSHOT.search(text) else text
    if m := _TYPED.match(action):
        return f"Searched “{_cut(m['q'], limit - 12)}”"
    if "browser back button" in action:
        return "Went back"
    if m := _CLICKED.match(action):
        label = m["label"]
        return "Opened a search result" if label.startswith("http") else f"Clicked “{_cut(label, limit - 10)}”"
    if action.startswith("I clicked"):
        return "Clicked a page control"
    if m := _SCROLL.match(action):
        return f"Scrolled {m['dir']}"
    if text.startswith("You are given:"):
        return "Read the task and the plan"
    if text.upper().rstrip(".") == "TERMINATE":
        return "Ended the conversation"
    prose = _CODE.sub(" ", text).strip()
    first = re.split(r"(?<=[.!?])\s", prose, maxsplit=1)[0] if prose else "Wrote code"
    return _cut(first, limit)


def summarize(step: Step) -> dict:
    """{summary, short, meta, flags, page} for timeline rows."""
    text = step.text.strip()
    flags: list[str] = []
    page = None

    if step.kind == "task":
        summary = f"Asked: {_clip(text, 150)}"
    elif step.kind == "stub":
        summary = _clip(text.replace("Next speaker", "Hands the turn to"), 80)
    elif step.kind == "termination":
        summary = f"Ends the run: {_clip(text, 120)}"
    elif step.kind == "instruction":
        summary = f"To {step.target}: {_clip(text, 150)}"
    elif step.kind == "ledger":
        ledger = _ledger(text) or {}
        flags = [
            label
            for label, bad in (
                ("not done", ledger.get("satisfied") is False),
                ("in a loop", ledger.get("in_loop") is True),
                ("no progress", ledger.get("progress") is False),
            )
            if bad
        ]
        nxt = ledger.get("next")
        summary = f"Updates the plan; next: {nxt}" if nxt else "Updates the plan"
        if ledger.get("instruction"):
            summary += f". {_clip(str(ledger['instruction']), 110)}"
    elif step.kind == "thought" and text.startswith("Initial plan"):
        summary = "Writes the initial plan and assembles the team"
    elif step.kind == "tool_output" or step.agent == "ComputerTerminal":
        match = _EXIT.search(text)
        code = next((g for g in (match.groups() if match else ()) if g), None)
        body = re.sub(r"^exitcode:[^\n]*\n?(Code output:)?", "", text).strip()
        if code and code != "0":
            flags.append(f"exit {code}")
            summary = f"Code failed: {_clip(body, 120)}"
        else:
            summary = f"Code ran: {_clip(body, 130)}" if body else "Code ran with no output"
    else:
        shot = _SCREENSHOT.search(text)
        if shot:
            title = _clip(shot["title"], 80) if _readable_title(shot["title"]) else ""
            page = {"domain": _domain(shot["url"]), "title": title, "url": shot["url"]}
            action = text[: shot.start()].strip() or "Looks at a page"
            summary = f"{_clip(action, 120)} → {page['domain']}"
        elif text.startswith("You are given:"):
            summary = "Reads the task and the manager's plan, then starts working"
        else:
            blocks = _CODE.findall(text)
            prose = _CODE.sub(" ", text)
            summary = _clip(prose, 160) or "Writes code"
            if blocks:
                flags.append(f"{len(blocks)} code block{'s' if len(blocks) > 1 else ''}")
    return {
        "summary": summary,
        "short": short(step),
        "meta": page["domain"] if page else "",
        "flags": flags,
        "page": page,
        "role": role(step.agent),
    }


def _readable_title(title: str) -> bool:
    latin = sum(ch.isascii() for ch in title)
    return bool(title) and latin / len(title) > 0.8


def _page_line(m: re.Match) -> str:
    title = m["title"]
    if _readable_title(title):
        return f"Opened page: {_clip(title, 90)} ({_domain(m['url'])})"
    return f"Opened page on {_domain(m['url'])} (title not in English; see the raw log)"


def clean(step: Step) -> str:
    """Full text minus page-metadata dumps and OCR noise; the raw log stays one click away."""
    text = step.text
    text = _META_BLOCK.sub("", text)
    text = _OCR_BLOCK.sub("", text)
    text = _VIEWPORT.sub("", text)
    text = _SCREENSHOT.sub(_page_line, text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
