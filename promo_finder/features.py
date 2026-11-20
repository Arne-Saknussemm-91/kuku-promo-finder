"""Per-event signals (pure rules, deterministic, ~ms per show).

hook   : would this beat stop a scrolling thumb if it were the first 3 seconds?
cliff  : does this beat leave something unresolved if we cut right after it?
debt   : does this beat lean on backstory a stranger lacks?
explain: does it tell a stranger who is who (husband, sister, duke...)?

Leakage guard: writers sometimes annotate sound cues with words like
"cliffhanger". We strip those words before scoring so the evaluation that
uses them as ground truth is honest (see eval.py).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .parse import Episode, Event

import os

ROOT = Path(__file__).resolve().parent.parent
HOOK_WINDOW = float(os.environ.get("PROMO_HOOK_WINDOW", 3.0))   # hook must land within this many seconds
CLIFF_TAIL = float(os.environ.get("PROMO_CLIFF_TAIL", 4.0))     # cliffhanger must land in the last this many
DEFAULT_LEXICON = ROOT / "config" / "lexicon.json"


@dataclass
class Signals:
    hook: float = 0.0
    cliff: float = 0.0
    debt: float = 0.0
    explain: float = 0.0
    arousal: float = 0.0
    tags: tuple = ()


DEFAULT_CLIFF = ROOT / "config" / "cliff_weights.json"
CLIFF_KEYS = ["sfx_intensity", "threat_or_tease", "question", "cutoff", "arousal_emotion", "unresolved_action",
              "cut_camera", "scene_break"]


class Lexicon:
    def __init__(self, path: str | Path = DEFAULT_LEXICON, cliff_weights: str | Path | dict | None = DEFAULT_CLIFF):
        raw = json.loads(Path(path).read_text())
        self.cw = {k: 1.0 for k in CLIFF_KEYS}
        if isinstance(cliff_weights, dict):
            self.cw.update(cliff_weights)
        elif cliff_weights and Path(cliff_weights).exists():
            self.cw.update({k: v for k, v in json.loads(Path(cliff_weights).read_text()).items() if k in CLIFF_KEYS})
        self.leak = [w.lower() for w in raw.get("leak_words", [])]
        self.groups = {}
        for k, v in raw.items():
            if isinstance(v, dict) and "words" in v:
                words = sorted(set(w.lower() for w in v["words"]), key=len, reverse=True)
                # match common inflections too: marry -> marrying, betray -> betrayed, hang -> hangs
                pats = [re.compile(r"(?<![a-z])" + re.escape(w) + (r"(?:s|es|ed|d|ing)?" if " " not in w and len(w) >= 4 else "")
                                   + r"(?![a-z])") for w in words]
                self.groups[k] = (float(v.get("weight", 1.0)), words, pats)

    def scrub(self, text: str) -> str:
        t = text.lower()
        for w in self.leak:
            t = re.sub(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", " ", t)
        return t

    def hits(self, group: str, text: str) -> list[str]:
        if group not in self.groups:
            return []
        _, words, pats = self.groups[group]
        return [w for w, p in zip(words, pats) if p.search(text)]

    def score(self, group: str, text: str, cap: int = 3) -> float:
        if group not in self.groups:
            return 0.0
        w = self.groups[group][0]
        return w * min(cap, len(self.hits(group, text)))


BLIND_EXTRA = ["cut to black", "fade to black", "fade out", "fades out", "fades to black", "smash cut to black"]


def event_signals(ev: Event, lex: Lexicon, nxt: Event | None, is_last: bool, blind: bool = False) -> Signals:
    """blind=True is for evaluation: no position info and no end-of-episode
    words, so the detector cannot 'cheat' by knowing where the episode ends."""
    t = lex.scrub(ev.text)
    if blind:
        for w in BLIND_EXTRA:
            t = t.replace(w, " ")
    emo = lex.scrub(ev.emotion)
    tags = []
    hook = cliff = debt = explain = arousal = 0.0

    if ev.kind in ("action", "camera", "sfx"):
        ps = lex.score("physical_shock", t)
        hook += ps
        if ps:
            tags.append("physical:" + ",".join(lex.hits("physical_shock", t)[:2]))
    if ev.kind == "sfx":
        s = lex.score("sfx_intensity", t, cap=2)
        hook += 0.6 * s
        cliff += 0.5 * s * lex.cw["sfx_intensity"]
    if ev.kind == "dialogue":
        sl = lex.score("shock_line", t)
        hook += sl
        if sl:
            tags.append("line:" + ",".join(lex.hits("shock_line", t)[:2]))
        arousal = lex.score("arousal_emotion", emo, cap=2)
        hook += 0.8 * arousal
        words = len(t.split())
        if words <= 9 and ("!" in ev.text or "?" in ev.text):
            hook += 0.8                       # short punchy line
        if words >= 25:
            hook -= 0.8                       # monologue openers rarely stop a scroll
        tt = lex.score("threat_or_tease", t)
        cliff += tt * lex.cw["threat_or_tease"]
        if tt:
            tags.append("tease:" + ",".join(lex.hits("threat_or_tease", t)[:2]))
        if ev.text.rstrip().endswith("?"):
            cliff += 1.0 * lex.cw["question"]
        if ev.text.rstrip().endswith(("...", "…", "—", "-")):
            cliff += 1.2 * lex.cw["cutoff"]   # cut-off line
        cliff += 0.5 * arousal * lex.cw["arousal_emotion"]
        explain += lex.score("self_explaining", t, cap=2)
    if ev.kind in ("action", "camera"):
        ua = lex.score("unresolved_action", t)
        cliff += ua * lex.cw["unresolved_action"]
        cc = lex.score("cut_camera", t, cap=2)
        cliff += cc * lex.cw["cut_camera"]
        hook += 0.3 * cc
        explain += 0.5 * lex.score("self_explaining", t, cap=1)
    debt = lex.score("context_debt", t)

    # structural: a beat followed by a scene change or the episode end is a natural cut
    if not blind and (is_last or (nxt is not None and nxt.scene != ev.scene)):
        cliff += 1.0 * lex.cw["scene_break"]
        tags.append("scene-break")
    return Signals(hook=max(0.0, hook), cliff=cliff, debt=debt, explain=explain,
                   arousal=arousal, tags=tuple(tags))


def annotate(ep: Episode, lex: Lexicon, blind: bool = False) -> list[Signals]:
    evs = ep.events
    out = []
    for i, ev in enumerate(evs):
        nxt = evs[i + 1] if i + 1 < len(evs) else None
        out.append(event_signals(ev, lex, nxt, is_last=(i == len(evs) - 1), blind=blind))
    return out


def window_profile(ep: Episode, sigs: list[Signals], t0: float, t1: float):
    """Aggregate signals inside [t0, t1]: hook = best beat in first 3s,
    cliff = best beat ending in last 4s, plus density / debt / explain."""
    hook = cliff = dens = debt = explain = 0.0
    hook_ev = cliff_ev = None
    n = 0
    for ev, s in zip(ep.events, sigs):
        if ev.end <= t0 or ev.start >= t1:
            continue
        n += 1
        if ev.start < t0 + HOOK_WINDOW and s.hook > hook:
            hook, hook_ev = s.hook, ev
        if ev.end > t1 - CLIFF_TAIL and ev.end <= t1 + 0.01 and s.cliff > cliff:
            cliff, cliff_ev = s.cliff, ev
        dens += s.hook
        debt += s.debt
        explain += s.explain
    dur = max(1e-6, t1 - t0)
    return {"hook": hook, "hook_ev": hook_ev, "cliff": cliff, "cliff_ev": cliff_ev,
            "density": dens / dur * 10.0, "debt": debt, "explain": explain, "n": n}
