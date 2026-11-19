"""End-to-end run: screenplay folder -> ranked, cut-ready promos."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from . import candidates as C
from .cast import build_cast, main_characters
from .export import promo_record, write_csv, write_edl, write_ffmpeg, write_json, write_markdown
from .features import Lexicon, annotate
from .judge import compose_supercuts, judge_candidate, rank_list, stranger_check
from .llm import LLM
from .parse import load_show
from .rank import diverse_order, score_all

ROOT = Path(__file__).resolve().parent.parent

ALIAS_SYS = "You map screenplay speaker labels to named characters. Only map when the script makes it clear. Return ONLY JSON."
ALIAS_USER = """Speaker labels in a show (label | lines | first line spoken | episode loglines mentioning them are below).
Some labels are descriptions of a named character before their name is revealed (e.g. "MASKED RIDER" may later be named DAMIEN).
LABELS:
{labels}
LOGLINES:
{loglines}
Return JSON {{"aliases": {{"<DESCRIPTIVE LABEL>": "<NAMED LABEL from the list>"}}}} — only confident mappings."""


def llm_aliases(llm, eps):
    cnt = Counter(ev.speaker for e in eps for ev in e.dialogue())
    first = {}
    for e in eps:
        for ev in e.dialogue():
            first.setdefault(ev.speaker, (e.num, ev.text[:70]))
    labels = "\n".join(f"{k} | {n} | ep{first[k][0]}: \"{first[k][1]}\"" for k, n in cnt.most_common(60))
    loglines = "\n".join(f"Ep{e.num}: {e.logline}" for e in eps[:20])
    r = llm.chat_json(ALIAS_SYS, ALIAS_USER.format(labels=labels, loglines=loglines), temperature=0.0)
    out = {}
    for k, v in (r.get("aliases") or {}).items():
        if k.upper() in cnt and str(v).upper() in cnt and k.upper() != str(v).upper():
            out[k.upper()] = str(v).upper()          # code rejects any invented name
    return out


class Pending(Exception):
    pass


def _gate(llm, stage, log):
    if llm.pending:
        log(f"[queue] {llm.pending} prompts waiting in .cache/queue/ for stage '{stage}'. "
            f"Answer them (<key>.answer.json) and rerun the same command.")
        raise Pending(stage)


def run(*a, **kw):
    try:
        return _run(*a, **kw)
    except Pending:
        return None, None


def _run(input_dir, out_dir, provider="none", model=None, top_k=20, shortlist=40, weights_file=None,
        lexicon=None, fps=30, video_pattern="episode-{ep:03d}.mp4", log=print):
    t0 = time.time()
    out = Path(out_dir)
    (out / "edl").mkdir(parents=True, exist_ok=True)
    llm = LLM(provider, model)
    eps = load_show(input_dir)
    if not eps:
        raise SystemExit(f"no episode .txt files found under {input_dir}")
    show = Path(input_dir).name if len({Path(e.path).parent for e in eps}) > 1 else Path(eps[0].path).parent.name
    E = {e.num: e for e in eps}
    log(f"[1/7] parsed {len(eps)} episodes, {sum(len(e.events) for e in eps)} timed events, "
        f"{sum(e.duration for e in eps) / 60:.1f} min  ({show})")

    aliases = {}
    alias_file = out / "aliases.json"
    if llm.enabled:
        aliases = llm_aliases(llm, eps)
        _gate(llm, "aliases", log)
        write_json(alias_file, {"_note": "speaker label -> character; edit freely, rerun to apply", **aliases})
    build_cast(eps, alias_file if alias_file.exists() else None)
    mains = main_characters(eps)
    log(f"[2/7] cast: main = {', '.join(mains[:5])}; {len(aliases)} LLM aliases")

    lex = Lexicon(lexicon) if lexicon else Lexicon()
    sig = {e.num: annotate(e, lex) for e in eps}
    weights = None
    wf = Path(weights_file) if weights_file else ROOT / "config" / "weights.json"
    if wf.exists():
        weights = {k: v for k, v in json.loads(wf.read_text()).items() if not k.startswith("_")}
    S = C.singles(eps, sig, weights=weights)
    X = C.supercuts(eps, sig, mains)
    log(f"[3/7] rules: {len(S)} single-clip + {len(X)} cross-episode candidates (all cut-safe)")

    pool = sorted(S, key=lambda c: -c.rule_score)
    short = pool[: int(shortlist * 0.7)] + sorted(X, key=lambda c: -c.rule_score)[: shortlist - int(shortlist * 0.7)]
    rejected, reasons = [], []
    if llm.enabled:
        bs = C.beats(eps, sig)
        bpool = C._nms_beats([b for b in bs if not b["mid"]], "hook", 30) + C._nms_beats(bs, "cliff", 30)
        seen, uniq = set(), []
        for b in sorted(bpool, key=lambda b: (b["ep"], b["t0"])):
            if (b["ep"], b["t0"], b["t1"]) not in seen:
                seen.add((b["ep"], b["t0"], b["t1"]))
                uniq.append(b)
        composed, rejected = compose_supercuts(llm, eps, E, uniq, show)
        _gate(llm, "compose", log)
        short += composed
        log(f"[4/7] LLM trailer-editor proposed {len(composed) + len(rejected)} supercuts; "
            f"code kept {len(composed)}, rejected {len(rejected)}")
        for i, c in enumerate(short):
            c.llm = judge_candidate(llm, E, c)
            if (i + 1) % 10 == 0:
                log(f"      judged {i + 1}/{len(short)}")
        _gate(llm, "judge", log)
        stranger_check(llm, E, short)
        _gate(llm, "stranger", log)
        log(f"[5/7] blind judge + stranger test on {len(short)} candidates")
    else:
        log("[4/7] rules-only mode (no LLM): skipping composer, judge, stranger test")
        log("[5/7] -")

    score_all(short, len(eps), llm.enabled)
    if llm.enabled:
        finalists = sorted(short, key=lambda c: -c.final)[:15]
        pts, reasons = rank_list(llm, finalists, rounds=3)
        _gate(llm, "rank", log)
        for c in finalists:
            c.llm["rank_points"] = round(pts[c.cid], 3)
            c.final = round(c.final + 2.0 * pts[c.cid], 3)
        log("[6/7] listwise ranking x3 (shuffled) on top 15")
    else:
        log("[6/7] -")

    ranked = diverse_order(short, k=top_k)
    recs = [promo_record(i + 1, c) for i, c in enumerate(ranked)]
    rest = [promo_record(None, c) for c in sorted(S + X, key=lambda c: -c.rule_score) if c not in ranked]

    write_json(out / "promos.json", recs)
    write_csv(out / "promos.csv", recs)
    write_json(out / "all_candidates.json", {"ranked": recs, "unranked_rule_candidates": rest})
    for r in recs:
        write_edl(out / "edl" / f"promo_{r['rank']:02d}_{r['id']}.edl", r, fps=fps, video_pattern=video_pattern)
    write_ffmpeg(out / "cut_promos.sh", recs, video_pattern="videos/" + video_pattern)
    if rejected:
        write_json(out / "llm_rejected_proposals.json", rejected)
    meta = {"show": show, "episodes": len(eps), "candidates": len(S) + len(X) + (len(short) - 28 if False else 0),
            "rule_singles": len(S), "rule_supercuts": len(X), "judged": len(short) if llm.enabled else 0,
            "model": llm.tag, "llm": llm.stats(), "runtime_s": round(time.time() - t0, 1),
            "top_reason": reasons[:3], "main_characters": mains}
    meta["candidates"] = len(S) + len(X)
    write_json(out / "run_meta.json", meta)
    write_markdown(out / "promos.md", recs, meta, reasons)
    try:
        from .board import write_board
        write_board(out / "board.html", eps, sig, recs, meta)
    except ImportError:
        pass
    log(f"[7/7] wrote {len(recs)} ranked promos -> {out}  ({meta['runtime_s']}s, {llm.stats()})")
    return recs, meta
