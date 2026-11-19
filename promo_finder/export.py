"""Editor-facing outputs: JSON, CSV, Markdown, CMX3600 EDL, ffmpeg script."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .candidates import Candidate
from .parse import sec_to_ts


def promo_record(rank: int | None, c: Candidate) -> dict:
    L = c.llm
    return {
        "rank": rank, "id": c.cid, "type": c.kind, "duration_s": round(c.dur, 2),
        "segments": [s.as_dict() for s in c.segments],
        "hook": L.get("hook") or c.hook_text, "cliffhanger": L.get("cliffhanger") or c.cliff_text,
        "why": L.get("why") or _rules_why(c),
        "hook_line_in_script": c.hook_text, "cliff_line_in_script": c.cliff_text,
        "emotion": L.get("emotion", ""), "platform_risk": L.get("platform_risk", ""),
        "stranger_summary": L.get("stranger_summary", ""), "stranger_match_0_2": L.get("stranger_match"),
        "confusing_for_new_viewer": L.get("confusing", ""),
        "scores": {"final": c.final, **{k: v for k, v in c.parts.items()},
                   "llm_hook": L.get("hook_score"), "llm_cliff": L.get("cliff_score"),
                   "llm_standalone": L.get("standalone_score"), "llm_rank_points": L.get("rank_points")},
        "characters": c.chars, "thread": c.thread,
    }


def _rules_why(c: Candidate) -> str:
    p = c.parts
    bits = [f"hook strength {p.get('hook', 0):.1f}", f"cliff strength {p.get('cliff', 0):.1f}"]
    if c.kind == "supercut":
        bits.append("cross-episode thread: " + ", ".join(c.thread[:3]))
    if p.get("debt"):
        bits.append(f"needs some backstory ({p['debt']:.0f})")
    return "; ".join(bits)


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def write_csv(path: Path, recs: list[dict]):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "type", "duration_s", "segments", "hook", "cliffhanger", "why", "final_score",
                    "emotion", "platform_risk", "stranger_match_0_2"])
        for r in recs:
            segs = " + ".join(f"Ep{s['episode']} {s['start']}-{s['end']}" for s in r["segments"])
            w.writerow([r["rank"], r["id"], r["type"], r["duration_s"], segs, r["hook"], r["cliffhanger"], r["why"],
                        r["scores"]["final"], r["emotion"], r["platform_risk"], r["stranger_match_0_2"]])


def _tc(sec: float, fps: int) -> str:
    fr = int(round(sec * fps))
    h, fr = divmod(fr, 3600 * fps)
    m, fr = divmod(fr, 60 * fps)
    s, f = divmod(fr, fps)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def write_edl(path: Path, rec: dict, fps: int = 30, video_pattern: str = "episode-{ep:03d}.mp4"):
    """CMX3600 EDL: imports into Premiere / Resolve / Final Cut (via converter)."""
    lines = [f"TITLE: PROMO_{rec['rank'] or rec['id']}_{rec['id']}", "FCM: NON-DROP FRAME", ""]
    rec_t = 0.0
    for n, s in enumerate(rec["segments"], 1):
        d = s["end_s"] - s["start_s"]
        lines.append(f"{n:03d}  EP{s['episode']:03d}    V     C        {_tc(s['start_s'], fps)} {_tc(s['end_s'], fps)} "
                     f"{_tc(rec_t, fps)} {_tc(rec_t + d, fps)}")
        lines.append(f"* FROM CLIP NAME: {video_pattern.format(ep=s['episode'])}")
        lines.append(f"{n:03d}  EP{s['episode']:03d}    A     C        {_tc(s['start_s'], fps)} {_tc(s['end_s'], fps)} "
                     f"{_tc(rec_t, fps)} {_tc(rec_t + d, fps)}")
        lines.append("")
        rec_t += d
    path.write_text("\n".join(lines))


def write_ffmpeg(path: Path, recs: list[dict], video_pattern: str = "videos/episode-{ep:03d}.mp4"):
    out = ["#!/usr/bin/env bash", "# Cuts every ranked promo from episode videos (set VIDEO dir / pattern as needed).",
           "set -euo pipefail", "mkdir -p promos", ""]
    for r in recs:
        name = f"promo_{r['rank']:02d}_{r['id']}"
        parts = []
        for k, s in enumerate(r["segments"]):
            seg = f"promos/{name}_part{k + 1}.mp4"
            out.append(f"ffmpeg -y -loglevel error -ss {s['start_s']:.2f} -to {s['end_s']:.2f} "
                       f"-i \"{video_pattern.format(ep=s['episode'])}\" -c:v libx264 -preset veryfast -c:a aac \"{seg}\"")
            parts.append(seg)
        if len(parts) == 1:
            out.append(f"mv \"{parts[0]}\" \"promos/{name}.mp4\"")
        else:
            lst = f"promos/{name}.txt"
            out.append(f"printf \"file '%s'\\n\" {' '.join(p.split('/')[-1] for p in parts)} > \"{lst}\"")
            out.append(f"ffmpeg -y -loglevel error -f concat -safe 0 -i \"{lst}\" -c copy \"promos/{name}.mp4\"")
        out.append("")
    path.write_text("\n".join(out))
    path.chmod(0o755)


def write_markdown(path: Path, recs: list[dict], meta: dict, reasons: list[str]):
    L = [f"# Promo picks — {meta['show']}", "",
         f"Model: `{meta['model']}` · episodes: {meta['episodes']} · candidates scored: {meta['candidates']} · "
         f"runtime: {meta['runtime_s']}s", ""]
    if reasons:
        L += ["**Why #1 goes first (LLM ranker):** " + reasons[0], ""]
    for r in recs:
        segs = " + ".join(f"Ep{s['episode']} {s['start']}–{s['end']}" for s in r["segments"])
        L += [f"## #{r['rank']} · {r['id']} · {r['type']} · {r['duration_s']:.0f}s",
              f"- **Cut:** {segs}",
              f"- **Hook (0–3s):** {r['hook']}",
              f"- **Cliffhanger:** {r['cliffhanger']}",
              f"- **Why it works:** {r['why']}"]
        if r.get("stranger_summary"):
            L.append(f"- **Stranger test:** \"{r['stranger_summary']}\" → match {r['stranger_match_0_2']}/2")
        if r.get("confusing_for_new_viewer") and r["confusing_for_new_viewer"].lower() != "nothing":
            L.append(f"- **Might confuse:** {r['confusing_for_new_viewer']}")
        L += [f"- score {r['scores']['final']} · emotion {r['emotion'] or '-'} · risk {r['platform_risk'] or '-'}", ""]
    path.write_text("\n".join(L))
