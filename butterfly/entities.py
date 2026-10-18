"""Cheap, deterministic entity extraction: the 'things' a step introduces that later steps reuse."""

import re

# Long tool outputs (page metadata, file dumps) would add thousands of noisy entities.
MAX_SCAN_CHARS = 6000

_URL = re.compile(r"https?://[^\s)\]'\"<>]+")
_FILE = re.compile(
    r"\b[\w\-]+\.(?:csv|xlsx?|json|py|txt|pdf|png|jpe?g|mp3|wav|docx?|pptx?|xml|html?|md|zip|jsonld|pdb)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d{2,}(?:\.\d+)?(?![\w.])")
_QUOTED = re.compile(r"[\"“']([^\"”'\n]{4,60})[\"”']")
_PROPER = re.compile(r"\b(?:[A-Z][a-zA-Z0-9&\-]+(?:\s+(?:of|the|and|de|&)?\s*[A-Z][a-zA-Z0-9&\-]+){1,4})\b")

# Frequent capitalized phrases from agent scaffolding that carry no task information.
_STOPLIST = {
    "updated ledger", "next speaker", "computer terminal", "web surfer", "file surfer",
    "here is", "the following", "the viewport", "general task", "please", "exitcode",
}


def _is_year(token: str) -> bool:
    return token.isdigit() and len(token) == 4 and 1900 <= int(token) <= 2099


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().strip(".,;:!?)").lower())


def extract(text: str) -> dict[str, str]:
    """Return {normalized_entity: kind} for one step's text."""
    text = text[:MAX_SCAN_CHARS]
    found: dict[str, str] = {}
    for match in _URL.findall(text):
        found[_norm(match.split("?")[0].split("#")[0])] = "url"
    for match in _FILE.findall(text):
        found.setdefault(_norm(match), "file")
    for match in _NUMBER.findall(text):
        if not _is_year(match):
            found.setdefault(_norm(match), "number")
    for match in _QUOTED.findall(text):
        found.setdefault(_norm(match), "quote")
    for match in _PROPER.findall(text):
        key = _norm(match)
        if key not in _STOPLIST and len(key) > 4:
            found.setdefault(key, "name")
    return found


def given_entities(question: str) -> set[str]:
    """Entities already in the task statement are given, not introduced by any agent."""
    return set(extract(question))
