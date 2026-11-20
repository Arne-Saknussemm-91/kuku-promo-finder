"""Baseline: no pipeline. Hand a model one whole episode, ask for a promo.

Measures what the code layer actually buys. For each returned promo we check,
with code, whether it is even cuttable:
  * 20-60 s long
  * start and end are real shot boundaries (CAMERA cue in/out points)
  * neither end falls inside a line of dialogue
  * timestamps exist inside the episode
This is the honest answer to "why not just use an LLM for everything".
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .candidates import cut_points
from .judge import episode_transcript
from .parse import Episode, ts_to_sec

SYS = """You are a promo editor for vertical microdrama apps. You read one full episode screenplay and choose the best
20-60 second promo clip to post on TikTok to pull in new viewers. It must open on a hook in the first 3 seconds, end on
an unresolved cliffhanger, and make sense to someone who has never seen the show. Return ONLY JSON."""
USER = """EPISODE {ep} SCREENPLAY (timestamps in seconds from episode start):
{transcript}

Return JSON: {{"start": "MM:SS.ss", "end": "MM:SS.ss", "hook": "<=15 words", "cliffhanger": "<=15 words",
"why": "one line"}}"""

TS = re.compile(r"(\d{1,2}):(\d{2}(?:\.\d+)?)")


def _parse_ts(v):
    if isinstance(v, (int, float)):
        return float(v)
    m = TS.search(str(v))
    if m:
        return ts_to_sec(m.group(1), m.group(2))
    try:
        return float(str(v).strip())
    except ValueError:
        return None


def ask(llm, eps, n=10):
    out = []
    for ep in eps[:n]:
        r = llm.chat_json(SYS, USER.format(ep=ep.num, transcript=episode_transcript(ep)))
        out.append({"ep": ep.num, "raw": r, "start": _parse_ts(r.get("start")), "end": _parse_ts(r.get("end"))})
    return out


def check(eps, picks, tol=0.05):
    E = {e.num: e for e in eps}
    rows = []
    for p in picks:
        ep = E[p["ep"]]
        s, e = p["start"], p["end"]
        r = {"ep": ep.num, "start": s, "end": e}
        if s is None or e is None:
            r.update(parsed=False)
            rows.append(r)
            continue
        ins, outs = cut_points(ep)
        lines = [(ev.start, ev.end) for ev in ep.events if ev.kind == "dialogue"]
        r["parsed"] = True
        r["in_bounds"] = 0 <= s < e <= ep.duration + 0.01
        r["duration_s"] = round(e - s, 2)
        r["len_ok"] = 20 - 1e-6 <= e - s <= 60 + 1e-6
        r["start_on_shot"] = any(abs(s - x) < tol for x in ins)
        r["end_on_shot"] = any(abs(e - x) < tol for x in outs)
        r["cuts_a_line"] = any(a + 0.05 < t < b - 0.05 for t in (s, e) for a, b in lines)
        r["all_ok"] = bool(r["in_bounds"] and r["len_ok"] and r["start_on_shot"]
                           and r["end_on_shot"] and not r["cuts_a_line"])
        rows.append(r)
    n = len(rows)
    ok = sum(bool(r.get("all_ok")) for r in rows)
    summary = {"episodes": n, "cut_ready": f"{ok}/{n}",
               "parsed": sum(r["parsed"] for r in rows),
               "length_20_60": sum(bool(r.get("len_ok")) for r in rows),
               "start_on_shot_boundary": sum(bool(r.get("start_on_shot")) for r in rows),
               "end_on_shot_boundary": sum(bool(r.get("end_on_shot")) for r in rows),
               "cuts_mid_line": sum(bool(r.get("cuts_a_line")) for r in rows)}
    return summary, rows
