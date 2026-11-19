"""Speaker-label -> canonical character.

Scripts label one person several ways (JULIANA / LADY JULIANA, GERALD /
THE FORMER GROOM). Rules handle honorifics; descriptive aliases come from an
optional per-show alias file (hand-edited or produced once by the LLM).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .parse import Episode

HONORIFICS = ("LADY ", "LORD ", "SIR ", "THE ", "MISS ", "MRS ", "MR ", "MADAM ", "DUKE ", "CAPTAIN ")


def strip_hon(label: str) -> str:
    s = label.upper().strip()
    changed = True
    while changed:
        changed = False
        for h in HONORIFICS:
            if s.startswith(h) and len(s) > len(h):
                s, changed = s[len(h):], True
    return s


def build_cast(eps: list[Episode], alias_file: str | Path | None = None) -> dict[str, str]:
    counts = Counter(ev.speaker for e in eps for ev in e.dialogue())
    names = set(counts)
    mapping = {}
    for lab in names:
        core = strip_hon(lab)
        mapping[lab] = core if core in names or any(strip_hon(n) == core for n in names if n != lab) else lab
    if alias_file and Path(alias_file).exists():
        extra = json.loads(Path(alias_file).read_text())
        for k, v in extra.items():
            if not k.startswith("_"):
                mapping[k.upper()] = strip_hon(v)
    for e in eps:
        for ev in e.events:
            if ev.kind == "dialogue":
                ev.char = mapping.get(ev.speaker, ev.speaker)
    return mapping


def main_characters(eps: list[Episode], k: int = 6) -> list[str]:
    c = Counter(ev.char for e in eps for ev in e.dialogue())
    return [n for n, _ in c.most_common(k)]


def first_seen(eps: list[Episode]) -> dict[str, int]:
    seen = {}
    for e in eps:
        for ev in e.dialogue():
            seen.setdefault(ev.char, e.num)
    return seen
