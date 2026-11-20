"""Turn signals into cut-ready promo candidates (pure code).

Editing rules enforced here (so an editor never gets an unusable cut):
  * cuts land on shot boundaries (CAMERA cue starts/ends)
  * no line of dialogue is cut in half at either end
  * 20-60 s total; sweet spot 28-50 s
Two shapes:
  * single  : one continuous stretch from one episode
  * supercut: 2-3 beats, in story order, across episodes, linked by shared
              characters + a shared story thread (the ReelShort-trailer shape:
              setup humiliation -> escalation -> unresolved reveal)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .features import Signals, window_profile
from .parse import Episode, sec_to_ts

import os

MIN_LEN = float(os.environ.get("PROMO_MIN_LEN", 20.0))
MAX_LEN = float(os.environ.get("PROMO_MAX_LEN", 60.0))
SWEET = (float(os.environ.get("PROMO_SWEET_LO", 28.0)), float(os.environ.get("PROMO_SWEET_HI", 50.0)))
STOP = set("""the a an and or but if of to in on at by for with from as is are was were be been being this that these those
it its he she they them his her their you your i me my we our us not no yes so do does did have has had will would can could
shall should may might must just than then there here what who whom which when where why how all any some more most very
into out up down over under again still only own same too s t don now o oh ah let lets get got go going come came one""".split())
# film-grammar words say nothing about the story thread
STOP |= set("""close closeup medium wide shot shots tracking camera frame framing angle push pull dolly zoom insert profile
focus face faces eyes expression looking looks look stares staring turns toward towards behind front background foreground
slowly slight slightly into across beside holds holding hold steps step stands standing sits sitting stand moment beat
light lighting shadow shadows room hall door doors open opens table chair""".split())
_SCORER = None
MODEL_CLIFF_W = float(os.environ.get("PROMO_MODEL_CLIFF_W", 1.0))   # 0 = pure rules


def set_scorer(sc):
    """Install the learned cliffhanger model (promo_finder.bow.Scorer)."""
    global _SCORER
    _SCORER = sc if (sc is not None and sc.ok) else None


def model_cliff(ep, t):
    """0-6ish, same scale as the rule cliff score, so the two can be added."""
    return 0.0 if _SCORER is None else 6.0 * MODEL_CLIFF_W * _SCORER.cliff(ep, t)


RESPONSE_OPENERS = ("yes", "no", "because", "but", "and", "then", "so", "well", "exactly", "indeed", "that's why", "as you")


@dataclass
class Segment:
    ep: int
    start: float
    end: float

    @property
    def dur(self):
        return self.end - self.start

    def as_dict(self):
        return {"episode": self.ep, "start": sec_to_ts(self.start), "end": sec_to_ts(self.end),
                "start_s": round(self.start, 2), "end_s": round(self.end, 2), "dur_s": round(self.dur, 2)}


@dataclass
class Candidate:
    cid: str
    kind: str                          # single | supercut
    segments: list[Segment]
    rule_score: float = 0.0
    parts: dict = field(default_factory=dict)
    hook_text: str = ""
    cliff_text: str = ""
    chars: list = field(default_factory=list)
    thread: list = field(default_factory=list)
    llm: dict = field(default_factory=dict)
    final: float = 0.0

    @property
    def dur(self):
        return sum(s.dur for s in self.segments)

    @property
    def first_ep(self):
        return min(s.ep for s in self.segments)

    @property
    def last_ep(self):
        return max(s.ep for s in self.segments)


def cut_points(ep: Episode):
    """Valid in/out points: shot boundaries not inside any dialogue line."""
    lines = [(ev.start, ev.end) for ev in ep.events if ev.kind == "dialogue"]

    def clean(t):
        return all(not (a + 0.05 < t < b - 0.05) for a, b in lines)
    shots = sorted({ev.start for ev in ep.events if ev.kind == "camera"} | {0.0})
    outs = sorted({ev.end for ev in ep.events if ev.kind == "camera"} | {ep.duration})
    return [t for t in shots if clean(t)], [t for t in outs if clean(t)]


def describe(ev) -> str:
    if ev is None:
        return ""
    if ev.kind == "dialogue":
        return f'{ev.char or ev.speaker}: "{ev.text}"'
    return f"[{ev.kind}] {ev.text}"


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 3 and w not in STOP}


def span_info(ep: Episode, t0: float, t1: float):
    chars, words, first_line = [], set(), None
    for ev in ep.events:
        if ev.end <= t0 or ev.start >= t1:
            continue
        if ev.kind == "dialogue":
            if first_line is None:
                first_line = ev
            if ev.char and ev.char not in chars:
                chars.append(ev.char)
        if ev.kind != "camera":
            words |= content_words(ev.text)
    names = {c.lower() for c in chars}
    return chars, {w for w in words if w not in names}, first_line


def len_penalty(d: float) -> float:
    lo, hi = SWEET
    return 0.0 if lo <= d <= hi else 0.06 * (lo - d if d < lo else d - hi)


def score_span(ep, sigs, t0, t1, w=None):
    w = w or {}
    p = window_profile(ep, sigs, t0, t1)
    p["cliff_model"] = model_cliff(ep, t1)
    p["cliff"] = p["cliff"] + p["cliff_model"]
    chars, words, first_line = span_info(ep, t0, t1)
    mid_convo = bool(first_line and first_line.text.lower().startswith(RESPONSE_OPENERS) and first_line.start < t0 + 4)
    crowd = max(0, len(chars) - 4)
    s = (w.get("hook", 1.0) * p["hook"] + w.get("cliff", 1.0) * p["cliff"]
         + w.get("density", 0.25) * min(p["density"], 12)
         + w.get("explain", 0.4) * min(p["explain"], 4)
         - w.get("debt", 0.6) * p["debt"]
         - 1.0 * mid_convo - 0.4 * crowd - len_penalty(t1 - t0))
    p.update(chars=chars, words=words, mid_convo=mid_convo, crowd=crowd)
    return s, p


def iou(a: Segment, b: Segment) -> float:
    if a.ep != b.ep:
        return 0.0
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    return inter / max(1e-6, (a.end - a.start) + (b.end - b.start) - inter)


def singles(eps, sigmap, per_ep=4, weights=None, min_hook=1.5, min_cliff=1.0):
    out = []
    for ep in eps:
        sigs = sigmap[ep.num]
        ins, outs = cut_points(ep)
        pool = []
        for t0 in ins:
            for t1 in outs:
                d = t1 - t0
                if d < MIN_LEN or d > MAX_LEN:
                    continue
                s, p = score_span(ep, sigs, t0, t1, weights)
                if p["hook"] < min_hook or p["cliff"] < min_cliff:
                    continue
                pool.append((s, t0, t1, p))
        pool.sort(key=lambda x: -x[0])
        kept = []
        for s, t0, t1, p in pool:
            seg = Segment(ep.num, t0, t1)
            if any(iou(seg, k.segments[0]) > 0.25 for k in kept):
                continue
            c = Candidate(cid=f"E{ep.num:02d}-S{len(kept) + 1}", kind="single", segments=[seg],
                          rule_score=round(s, 3),
                          parts={k: (round(v, 2) if isinstance(v, float) else v) for k, v in p.items()
                                 if k in ("hook", "cliff", "cliff_model", "density", "debt", "explain",
                                          "mid_convo", "crowd")},
                          hook_text=describe(p["hook_ev"]), cliff_text=describe(p["cliff_ev"]),
                          chars=p["chars"], thread=sorted(p["words"])[:0])
            c._words = p["words"]
            kept.append(c)
            if len(kept) >= per_ep:
                break
        out.extend(kept)
    return out


def beats(eps, sigmap, lo=5.0, hi=18.0):
    """Short shot-aligned beats used as supercut building blocks."""
    res = []
    for ep in eps:
        sigs = sigmap[ep.num]
        ins, outs = cut_points(ep)
        for t0 in ins:
            for t1 in outs:
                if lo <= t1 - t0 <= hi:
                    p = window_profile(ep, sigs, t0, t1)
                    p["cliff"] += model_cliff(ep, t1)
                    chars, words, first = span_info(ep, t0, t1)
                    res.append({"ep": ep.num, "t0": t0, "t1": t1, "hook": p["hook"], "cliff": p["cliff"],
                                "dens": p["density"], "debt": p["debt"], "explain": p["explain"],
                                "chars": chars, "words": words, "hook_ev": p["hook_ev"], "cliff_ev": p["cliff_ev"],
                                "mid": bool(first and first.text.lower().startswith(RESPONSE_OPENERS))})
    return res


def _nms_beats(bs, key, k):
    bs = sorted(bs, key=lambda b: -b[key])
    kept = []
    for b in bs:
        if any(b["ep"] == x["ep"] and min(b["t1"], x["t1"]) - max(b["t0"], x["t0"]) > 0 for x in kept):
            continue
        kept.append(b)
        if len(kept) >= k:
            break
    return kept


def _idf(eps):
    import math
    docs = [content_words(e.logline) | {w for ev in e.events if ev.kind != "camera" for w in content_words(ev.text)}
            for e in eps]
    df = {}
    for d in docs:
        for w in d:
            df[w] = df.get(w, 0) + 1
    n = len(docs)
    return {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}


def supercuts(eps, sigmap, main_chars, top=40, keep=40):
    bs = beats(eps, sigmap)
    idf = _idf(eps)
    logw = {e.num: content_words(e.logline) for e in eps}
    for b in bs:
        b["words"] = b["words"] | logw[b["ep"]]
    hooks = _nms_beats([b for b in bs if not b["mid"]], "hook", top)
    cliffs = _nms_beats(bs, "cliff", top)
    mids = _nms_beats(bs, "dens", top)
    mains = set(main_chars)
    combos = []
    for h in hooks:
        for c in cliffs:
            if c["ep"] <= h["ep"]:
                continue                       # supercuts are cross-episode, in story order
            shared = (set(h["chars"]) & set(c["chars"])) | ((set(h["chars"]) | set(c["chars"])) & mains)
            names = {x.lower() for x in mains | set(h["chars"]) | set(c["chars"])}
            thread = {w for w in (h["words"] & c["words"]) if w not in names and idf.get(w, 0) > 1.0}
            tw = sum(idf[w] for w in thread)
            if not (set(h["chars"]) & set(c["chars"])) or tw < 4.0:
                continue
            base_d = (h["t1"] - h["t0"]) + (c["t1"] - c["t0"])
            options = [None]
            if base_d < 30:
                options += [m for m in mids if h["ep"] <= m["ep"] <= c["ep"] and m is not h and m is not c
                            and not (m["ep"] == h["ep"] and m["t0"] < h["t1"])
                            and not (m["ep"] == c["ep"] and m["t1"] > c["t0"])
                            and (set(m["chars"]) & (set(h["chars"]) | set(c["chars"])))][:8]
            for m in options:
                parts = [h] + ([m] if m else []) + [c]
                d = sum(p["t1"] - p["t0"] for p in parts)
                if not (MIN_LEN <= d <= MAX_LEN):
                    continue
                coh = 0.35 * min(tw, 12) + 0.6 * len(set(h["chars"]) & set(c["chars"]))
                jump = c["ep"] - h["ep"]
                s = (h["hook"] + c["cliff"] + 0.2 * (m["dens"] if m else 0) + 0.3 * coh
                     + 0.3 * min(4, sum(p["explain"] for p in parts))
                     - 0.5 * sum(p["debt"] for p in parts) - 0.04 * jump - len_penalty(d))
                combos.append((s, parts, thread, coh))
    combos.sort(key=lambda x: -x[0])
    out, used = [], []
    for s, parts, thread, coh in combos:
        sig = {(p["ep"], round(p["t0"])) for p in parts}
        if any(len(sig & u) >= 1 for u in used):
            continue                           # every supercut uses fresh beats
        used.append(sig)
        segs = [Segment(p["ep"], p["t0"], p["t1"]) for p in parts]
        c = Candidate(cid=f"X{len(out) + 1:02d}", kind="supercut", segments=segs, rule_score=round(s, 3),
                      parts={"hook": round(parts[0]["hook"], 2), "cliff": round(parts[-1]["cliff"], 2),
                             "coherence": round(coh, 2)},
                      hook_text=describe(parts[0]["hook_ev"]), cliff_text=describe(parts[-1]["cliff_ev"]),
                      chars=list(dict.fromkeys(ch for p in parts for ch in p["chars"])),
                      thread=sorted(thread, key=lambda w: -idf[w])[:8])
        c._words = set().union(*[p["words"] for p in parts])
        out.append(c)
        if len(out) >= keep:
            break
    return out


def from_scouted(eps, sigmap, scouted, existing, top=15, weights=None):
    """Turn LLM-scouted opening moments into cut-safe candidates: start on the last
    clean shot boundary at/just before the moment, end on the best clean out point."""
    E = {e.num: e for e in eps}
    out = []
    for epn, t, sc, hook in sorted(scouted, key=lambda x: -x[2]):
        ep = E[epn]
        sigs = sigmap[epn]
        ins, outs = cut_points(ep)
        starts = [x for x in ins if t - 2.5 <= x <= t + 0.5]
        if not starts:
            continue
        t0 = max(starts)
        best = None
        for t1 in outs:
            if MIN_LEN <= t1 - t0 <= MAX_LEN:
                s, p = score_span(ep, sigs, t0, t1, weights)
                cl = p["cliff"] - len_penalty(t1 - t0)
                if best is None or cl > best[0]:
                    best = (cl, t1, s, p)
        if not best:
            continue
        seg = Segment(epn, t0, best[1])
        if any(iou(seg, c.segments[0]) > 0.5 for c in existing + out if c.kind == "single"):
            continue
        _, t1, s, p = best
        c = Candidate(cid=f"H{epn:02d}-{len(out) + 1}", kind="single", segments=[seg], rule_score=round(s, 3),
                      parts={"hook": round(p["hook"], 2), "cliff": round(p["cliff"], 2), "density": round(p["density"], 2),
                             "debt": p["debt"], "explain": p["explain"], "source": "llm-scout", "scout_score": sc},
                      hook_text=hook, cliff_text=describe(p["cliff_ev"]), chars=p["chars"])
        c._words = p["words"]
        out.append(c)
        if len(out) >= top:
            break
    return out
