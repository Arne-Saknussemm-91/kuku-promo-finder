"""Parse Kuku-style timestamped screenplays into a flat list of timed events.

Format (per episode file):
    EPISODE NUMBER: Episode 1
    EPISODE LOGLINE: ...
    CONTINUITY NOTES: ...
    --- SCRIPT START ---
    SCENE 01
    Scene heading: INT. CATHEDRAL – DAY
    Scene start: 00:00.00
    Scene end: 02:18.00
    **CAMERA:** / **ACTION:** / **SOUND FX:** / **SPEAKER (Emotion):**
    Start: (00:00.00) | End: (00:04.80)
    text...
    --- SCRIPT END ---
Pure code, no AI: parsing must be exact because timestamps are the deliverable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

TS = r"(\d{1,2}):(\d{2}(?:\.\d+)?)"
RE_TS_LINE = re.compile(rf"Start:\s*\({TS}\)\s*\|\s*End:\s*\({TS}\)")
RE_TAG = re.compile(r"^\*\*(.+?):\*\*\s*$")
RE_SPEAKER = re.compile(r"^(.*?)\s*(?:\((.*)\))?$")
RE_EP_NUM = re.compile(r"EPISODE NUMBER:\s*Episode\s*(\d+)", re.I)
RE_FILE_EP = re.compile(r"episode-(\d+)", re.I)

KINDS = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sfx", "SFX": "sfx",
         "MUSIC": "sfx", "TEXT": "text", "ON-SCREEN TEXT": "text", "TITLE": "text"}


def ts_to_sec(m: str, s: str) -> float:
    return int(m) * 60 + float(s)


def sec_to_ts(x: float) -> str:
    x = max(0.0, x)
    m = int(x // 60)
    return f"{m:02d}:{x - 60 * m:05.2f}"


@dataclass
class Event:
    ep: int
    idx: int
    kind: str            # camera | action | sfx | dialogue | text
    start: float
    end: float
    text: str
    speaker: str = ""    # raw speaker label, e.g. "LADY JULIANA"
    emotion: str = ""    # parenthetical, e.g. "Confused, Breathless"
    scene: int = 0
    heading: str = ""
    char: str = ""       # canonical character id (filled by cast.py)

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass
class Episode:
    num: int
    path: str
    logline: str = ""
    continuity: str = ""
    duration: float = 0.0
    events: list[Event] = field(default_factory=list)
    scenes: list[dict] = field(default_factory=list)

    def dialogue(self) -> list[Event]:
        return [e for e in self.events if e.kind == "dialogue"]


def parse_episode(path: Path) -> Episode:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    m = None
    for ln in lines[:5]:
        m = RE_EP_NUM.search(ln) or m
    num = int(m.group(1)) if m else int(RE_FILE_EP.search(path.name).group(1))
    ep = Episode(num=num, path=str(path))
    scene, heading = 0, ""
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("EPISODE LOGLINE:"):
            ep.logline = ln.split(":", 1)[1].strip()
        elif ln.startswith("CONTINUITY NOTES:"):
            ep.continuity = ln.split(":", 1)[1].strip()
        elif re.match(r"^SCENE\s+\d+", ln):
            scene = int(re.findall(r"\d+", ln)[0])
            ep.scenes.append({"scene": scene, "heading": "", "start": None, "end": None})
        elif ln.startswith("Scene heading:"):
            heading = ln.split(":", 1)[1].strip()
            if ep.scenes:
                ep.scenes[-1]["heading"] = heading
        elif ln.startswith("Scene start:") or ln.startswith("Scene end:"):
            mm = re.search(TS, ln)
            if mm and ep.scenes:
                key = "start" if ln.startswith("Scene start") else "end"
                ep.scenes[-1][key] = ts_to_sec(*mm.groups())
        else:
            tag = RE_TAG.match(ln)
            if tag:
                label = tag.group(1).strip()
                # timestamp line follows (allow blank lines)
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                tm = RE_TS_LINE.search(lines[j]) if j < len(lines) else None
                if tm:
                    s = ts_to_sec(tm.group(1), tm.group(2))
                    e = ts_to_sec(tm.group(3), tm.group(4))
                    k = j + 1
                    body = []
                    while k < len(lines) and lines[k].strip() and not RE_TAG.match(lines[k].strip()) \
                            and not lines[k].startswith("---"):
                        body.append(lines[k].strip())
                        k += 1
                    text = " ".join(body)
                    up = label.upper()
                    if up in KINDS:
                        kind, spk, emo = KINDS[up], "", ""
                    else:
                        sm = RE_SPEAKER.match(label)
                        kind, spk, emo = "dialogue", sm.group(1).strip().upper(), (sm.group(2) or "").strip()
                        text = text.strip().strip('"').strip("“”")
                    if e < s:
                        e = s
                    ep.events.append(Event(ep=num, idx=len(ep.events), kind=kind, start=s, end=e,
                                           text=text, speaker=spk, emotion=emo, scene=scene,
                                           heading=heading))
                    i = k
                    continue
        i += 1
    ep.events.sort(key=lambda x: (x.start, x.end))
    for n, ev in enumerate(ep.events):
        ev.idx = n
    ends = [s["end"] for s in ep.scenes if s.get("end")] + [ev.end for ev in ep.events]
    ep.duration = max(ends) if ends else 0.0
    return ep


def load_show(folder: str | Path) -> list[Episode]:
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*.txt"))
    eps = [parse_episode(p) for p in files]
    eps.sort(key=lambda e: e.num)
    return eps


def to_dict(ep: Episode) -> dict:
    d = asdict(ep)
    return d
