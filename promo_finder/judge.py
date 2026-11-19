"""LLM steps. Each one is narrow, returns JSON, and is checked by code.

1. judge_candidate : blind 'stranger' read of ONE clip (no logline, no
                     backstory). Scores hook / cliffhanger / standalone,
                     names the hook + cliffhanger, writes the one-line why,
                     flags platform risk, and guesses what the story is.
2. stranger_check  : a separate call compares that guess against the
                     writers' logline -> did a stranger get it right?
3. compose_supercuts: LLM as trailer editor; may only use beat IDs that
                     code already validated. Code re-validates the result.
4. rank_list       : listwise ranking of the finalists, run several times
                     with shuffled order (cancels position bias), Borda count.
"""
from __future__ import annotations

import random

from .candidates import Candidate, Segment, MIN_LEN, MAX_LEN, describe
from .parse import Episode, sec_to_ts

JUDGE_SYS = """You are a senior performance-marketing editor for vertical microdrama apps (ReelShort, DramaBox).
You judge whether a clip, cut from a show, works as a 20-60s paid social promo for TikTok/Reels/Shorts.
You see ONLY the clip (time-coded screenplay lines, relative to the clip start). You have NOT seen the show.
Judge like a cold scrolling viewer. Be strict: most clips are 4-6. Reserve 9-10 for clips you would bet ad money on.
Return ONLY JSON."""

JUDGE_USER = """CLIP ({dur:.0f}s{shape}):
{transcript}

Return JSON with exactly these keys:
{{
 "hook_score": 1-10,          // does something in the first 3 seconds stop a scroll (slap, reveal, shocking line)?
 "hook": "<what happens in the first 3s, <=15 words>",
 "cliff_score": 1-10,         // does the clip END on an unresolved moment that makes you need the next scene?
 "cliffhanger": "<the unresolved question at the cut, <=15 words>",
 "standalone_score": 1-10,    // could a stranger follow who is who and what is at stake?
 "stranger_summary": "<one sentence: what you think this story is about, as a new viewer>",
 "confusing": "<what a stranger would NOT understand, or 'nothing'>",
 "emotion": "<the main emotion it sells: revenge|humiliation|romance|betrayal|danger|mystery|triumph>",
 "platform_risk": "none|low|high",   // graphic violence, gore, sexual content that ad platforms reject
 "why": "<one line: why this would (or would not) pull new viewers>"
}}"""

CHECK_SYS = "You grade whether a new viewer's guess about a story matches what really happens. Return ONLY JSON."
CHECK_USER = """For each item, compare the VIEWER GUESS to the TRUTH (writers' episode summaries).
Score match 0-2: 2 = gets the core conflict and who-is-who right, 1 = partly right, 0 = wrong or lost.
{items}
Return JSON: {{"results": [{{"id": "<id>", "match": 0|1|2, "note": "<=12 words"}}]}}"""

COMPOSE_SYS = """You are the trailer editor for a vertical microdrama. You build multi-clip promos (supercuts)
that join 2-3 short beats from DIFFERENT episodes into one 20-60s promo. You may ONLY use beat IDs from the list.
Shape that works on TikTok: beat 1 = instant hook (shock line / violence / reveal), middle = escalation that
explains the stakes, last beat = unresolved cliffhanger. Beats must be in story order (episode ascending).
A stranger must understand it. Return ONLY JSON."""

COMPOSE_USER = """SHOW: {title}
STORY SO FAR (episode loglines):
{loglines}

BEATS (id | episode | seconds | text):
{beats}

Make {n} different supercuts. Each must total 20-60 seconds (sum of beat seconds) and tell ONE clear thread.
Return JSON: {{"supercuts": [{{"beats": ["B..", "B..", "B.."], "thread": "<the story thread, <=10 words>",
 "hook": "<opening moment>", "cliffhanger": "<closing unresolved moment>", "why": "<one line>"}}]}}"""

RANK_SYS = """You are the head of user acquisition at a microdrama app. You have budget to post promos one at a time.
Rank promos by which you would post FIRST: strongest scroll-stop hook, strongest need-to-know cliffhanger,
understandable to strangers, and lands viewers early in the series (fewer episodes before they get the payoff).
Return ONLY JSON."""
RANK_USER = """PROMOS:
{items}

Return JSON: {{"order": ["<id>", ...all ids, best first], "top_reason": "<why your #1 goes first, one line>"}}"""


def transcript(eps: dict[int, Episode], segs: list[Segment], max_cam=110) -> str:
    out, offset = [], 0.0
    for k, sg in enumerate(segs):
        if len(segs) > 1:
            out.append(f"--- CUT {k + 1} ---")
        ep = eps[sg.ep]
        for ev in ep.events:
            if ev.end <= sg.start + 0.01 or ev.start >= sg.end - 0.01:
                continue
            t = offset + max(0.0, ev.start - sg.start)
            if ev.kind == "dialogue":
                emo = f" ({ev.emotion})" if ev.emotion else ""
                out.append(f"[{t:05.1f}] {ev.char or ev.speaker}{emo}: \"{ev.text}\"")
            elif ev.kind == "sfx":
                out.append(f"[{t:05.1f}] SOUND: {ev.text.replace('cliffhanger', '').replace('Cliffhanger', '')[:max_cam]}")
            else:
                txt = ev.text if len(ev.text) <= max_cam else ev.text[:max_cam] + "…"
                out.append(f"[{t:05.1f}] {ev.kind.upper()}: {txt}")
        offset += sg.dur
    return "\n".join(out)


def _clip(x, lo=1, hi=10):
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return None


def judge_candidate(llm, eps, c: Candidate) -> dict:
    shape = "" if c.kind == "single" else f", supercut of {len(c.segments)} cuts"
    r = llm.chat_json(JUDGE_SYS, JUDGE_USER.format(dur=c.dur, shape=shape, transcript=transcript(eps, c.segments)))
    for k in ("hook_score", "cliff_score", "standalone_score"):
        r[k] = _clip(r.get(k))
    return r


def stranger_check(llm, eps, cands: list[Candidate], batch=8):
    todo = [c for c in cands if c.llm.get("stranger_summary")]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        items = []
        for c in chunk:
            truth = " | ".join(f"Ep{e}: {eps[e].logline}" for e in sorted({s.ep for s in c.segments}))
            items.append(f"ID {c.cid}\nVIEWER GUESS: {c.llm['stranger_summary']}\nTRUTH: {truth}")
        r = llm.chat_json(CHECK_SYS, CHECK_USER.format(items="\n\n".join(items)), temperature=0.0)
        by = {str(x.get("id", "")).replace("ID ", "").strip(): x for x in r.get("results", []) if isinstance(x, dict)}
        for c in chunk:
            x = by.get(c.cid)
            if x:
                c.llm["stranger_match"] = _clip(x.get("match"), 0, 2)
                c.llm["stranger_note"] = x.get("note", "")


def compose_supercuts(llm, eps_list, eps, beat_pool, title, n=8):
    """beat_pool: list of beat dicts from candidates.beats (already shot-aligned, no mid-line cuts)."""
    ids = {}
    lines = []
    for k, b in enumerate(beat_pool):
        bid = f"B{k + 1:03d}"
        ids[bid] = b
        text = " / ".join(x for x in (describe(b["hook_ev"]), describe(b["cliff_ev"])) if x)[:170]
        lines.append(f"{bid} | ep{b['ep']} | {b['t1'] - b['t0']:.0f}s | {text}")
    loglines = "\n".join(f"Ep{e.num}: {e.logline}" for e in eps_list)
    r = llm.chat_json(COMPOSE_SYS, COMPOSE_USER.format(title=title, loglines=loglines, beats="\n".join(lines), n=n),
                      temperature=0.4)
    out, rejected = [], []
    for k, sc in enumerate(r.get("supercuts", []) if isinstance(r.get("supercuts"), list) else []):
        bl = [ids.get(str(x).strip()) for x in sc.get("beats", [])]
        reason = None
        if not bl or any(b is None for b in bl):
            reason = "unknown beat id (hallucinated)"
        elif len(bl) < 2 or len(bl) > 3:
            reason = f"{len(bl)} beats"
        else:
            order = [(b["ep"], b["t0"]) for b in bl]
            d = sum(b["t1"] - b["t0"] for b in bl)
            if order != sorted(order):
                reason = "not in story order"
            elif len({b['ep'] for b in bl}) < 2:
                reason = "single episode"
            elif not (MIN_LEN <= d <= MAX_LEN):
                reason = f"duration {d:.0f}s outside 20-60"
        if reason:
            rejected.append({"proposal": sc, "rejected_because": reason})
            continue
        segs = [Segment(b["ep"], b["t0"], b["t1"]) for b in bl]
        c = Candidate(cid=f"L{k + 1:02d}", kind="supercut", segments=segs,
                      rule_score=round(bl[0]["hook"] + bl[-1]["cliff"], 3),
                      parts={"hook": round(bl[0]["hook"], 2), "cliff": round(bl[-1]["cliff"], 2), "source": "llm-composed"},
                      hook_text=describe(bl[0]["hook_ev"]), cliff_text=describe(bl[-1]["cliff_ev"]),
                      chars=list(dict.fromkeys(ch for b in bl for ch in b["chars"])),
                      thread=[sc.get("thread", "")])
        c._words = set().union(*[b["words"] for b in bl])
        out.append(c)
    return out, rejected


def rank_list(llm, cands: list[Candidate], rounds=3, seed=7) -> dict[str, float]:
    rng = random.Random(seed)
    pts = {c.cid: 0.0 for c in cands}
    reasons = []
    for r in range(rounds):
        order = cands[:]
        rng.shuffle(order)
        items = []
        for c in order:
            segs = ", ".join(f"ep{s.ep} {sec_to_ts(s.start)}-{sec_to_ts(s.end)}" for s in c.segments)
            items.append(f"{c.cid} [{c.kind}, {c.dur:.0f}s, {segs}] HOOK: {c.llm.get('hook') or c.hook_text[:90]} | "
                         f"CLIFF: {c.llm.get('cliffhanger') or c.cliff_text[:90]} | WHY: {c.llm.get('why', '')}")
        res = llm.chat_json(RANK_SYS, RANK_USER.format(items="\n".join(items)), temperature=0.0)
        got = [str(x).strip() for x in res.get("order", []) if str(x).strip() in pts]
        seen = []
        for x in got:
            if x not in seen:
                seen.append(x)
        n = len(cands)
        for pos, cid in enumerate(seen):
            pts[cid] += (n - pos) / n
        if res.get("top_reason"):
            reasons.append(res["top_reason"])
    return {k: v / rounds for k, v in pts.items()}, reasons


# ------------------------------------------------------------- hook scout
SCOUT_SYS = """You are a promo editor for vertical microdrama ads. You read one episode's screenplay and find the
moments that would stop a stranger scrolling TikTok in the first 3 seconds. Hooks are often SITUATIONS, not words:
a bride sees her groom with another woman, a slap, a secret revealed, a public humiliation, a shocking entrance.
Return ONLY JSON."""
SCOUT_USER = """EPISODE {ep} (time-coded, seconds from episode start):
{transcript}

Pick the {k} strongest promo OPENING moments (where the first 3 seconds of a promo should start).
Return JSON {{"moments": [{{"t": <seconds, a timestamp from the list>, "score": 1-10, "hook": "<what the viewer sees, <=15 words>"}}]}}"""


def episode_transcript(ep, max_cam=80):
    out = []
    for ev in ep.events:
        if ev.kind == "dialogue":
            emo = f" ({ev.emotion})" if ev.emotion else ""
            out.append(f"[{ev.start:06.1f}] {ev.char or ev.speaker}{emo}: \"{ev.text}\"")
        elif ev.kind in ("action", "sfx"):
            out.append(f"[{ev.start:06.1f}] {ev.kind.upper()}: {ev.text.replace('cliffhanger', '')[:max_cam]}")
        elif ev.kind == "camera" and len(out) and not out[-1].startswith(f"[{ev.start:06.1f}]"):
            out.append(f"[{ev.start:06.1f}] CAMERA: {ev.text[:max_cam]}")
    return "\n".join(out)


def scout_hooks(llm, eps, k=2):
    """One call per episode. Returns [(ep_num, t, score, hook_text)], validated by code."""
    res = []
    for ep in eps:
        r = llm.chat_json(SCOUT_SYS, SCOUT_USER.format(ep=ep.num, transcript=episode_transcript(ep), k=k))
        for m in (r.get("moments") or [])[:k]:
            try:
                t, sc = float(m.get("t")), float(m.get("score", 0))
            except (TypeError, ValueError):
                continue
            if 0 <= t < ep.duration - 20:                 # code rejects impossible timestamps
                res.append((ep.num, t, sc, str(m.get("hook", ""))[:120]))
    return res
