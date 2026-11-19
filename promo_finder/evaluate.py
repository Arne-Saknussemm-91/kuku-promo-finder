"""How do we know it's good? Checks that need no human and no LLM, plus
audits of a finished run.

A. Cliffhanger detector vs writer ground truth (blind).
   Microdrama episodes are written to END on a cliffhanger; 22 of these 50
   scripts even label it ("...swells to a cliffhanger peak"). We hide every
   label word, every end-of-episode word ("cut to black") and all position
   info, then ask: among every legal cut point in the episode, does the
   detector rank the real ending near the top? Compared to chance.
B. Ablations: which signal groups carry the result (and is sound a leak?).
C. Run audit: every promo obeys the editing rules (20-60s, shot-aligned,
   no half-cut lines, story order, inside episode bounds).
D. Rules vs LLM agreement + stranger-test results (if the run used an LLM).
E. Determinism: same input -> byte-identical rules output.
"""
from __future__ import annotations

import json
import random
import statistics as st
from pathlib import Path

from .candidates import cut_points
from .cast import build_cast
from .features import Lexicon, annotate
from .parse import load_show


def _cliff_at(ep, sigs, t, tail=4.0):
    best = 0.0
    for ev, s in zip(ep.events, sigs):
        if t - tail < ev.end <= t + 0.01:
            best = max(best, s.cliff)
    return best


def writer_confirmed(eps):
    """Episodes whose ending the writers explicitly call a cliffhanger: a label
    inside the script, or the NEXT episode's notes say 'Episode N's cliffhanger'."""
    import re
    s = set()
    for ep in eps:
        if any("cliffhanger" in ev.text.lower() for ev in ep.events):
            s.add(ep.num)
        for m in re.finditer(r"episode\s+(\d+)'?s?\s+cliffhanger", ep.continuity.lower()):
            s.add(int(m.group(1)))
    return s


def cliff_benchmark(eps, lex, labelled_only=False, min_t=20.0, only=None):
    rows = []
    confirmed = writer_confirmed(eps) if labelled_only else None
    for ep in eps:
        if labelled_only and ep.num not in confirmed:
            continue
        if only is not None and ep.num not in only:
            continue
        sigs = annotate(ep, lex, blind=True)
        _, outs = cut_points(ep)
        outs = [t for t in outs if t >= min_t]
        if not outs:
            continue
        gold = max(outs)                                   # the real episode ending
        scores = {t: _cliff_at(ep, sigs, t) for t in outs}
        g = scores[gold]
        better = sum(1 for t, v in scores.items() if v > g)
        ties = sum(1 for t, v in scores.items() if v == g and t != gold)
        rank = 1 + better + ties / 2.0                     # average rank on ties (no tie luck)
        rows.append({"ep": ep.num, "n_cut_points": len(outs), "rank_of_real_ending": rank,
                     "pct_rank": round(1 - (rank - 1) / max(1, len(outs) - 1), 3)})
    n = len(rows)
    if not n:
        return {}
    res = {"episodes": n,
           "hit@1": round(sum(r["rank_of_real_ending"] <= 1 for r in rows) / n, 3),
           "hit@3": round(sum(r["rank_of_real_ending"] <= 3 for r in rows) / n, 3),
           "hit@5": round(sum(r["rank_of_real_ending"] <= 5 for r in rows) / n, 3),
           "MRR": round(sum(1 / r["rank_of_real_ending"] for r in rows) / n, 3),
           "mean_percentile": round(st.mean(r["pct_rank"] for r in rows), 3),
           "random_hit@1": round(st.mean(1 / r["n_cut_points"] for r in rows), 3),
           "random_hit@3": round(st.mean(min(1, 3 / r["n_cut_points"]) for r in rows), 3),
           "random_hit@5": round(st.mean(min(1, 5 / r["n_cut_points"]) for r in rows), 3),
           "random_MRR": round(st.mean(sum(1 / k for k in range(1, r["n_cut_points"] + 1)) / r["n_cut_points"]
                                       for r in rows), 3),
           "random_mean_percentile": 0.5,
           "per_episode": rows}
    return res


def ablations(eps, base_path=None):
    base = json.loads(Path(base_path or Lexicon().__init__.__defaults__[0]).read_text()) if False else None
    from .features import DEFAULT_LEXICON
    raw = json.loads(Path(DEFAULT_LEXICON).read_text())
    out = {}
    groups = [k for k, v in raw.items() if isinstance(v, dict) and "words" in v]
    tmp = Path(DEFAULT_LEXICON).parent / ".ablation_tmp.json"
    try:
        for g in ["(none removed)"] + groups:
            r2 = json.loads(json.dumps(raw))
            if g in r2:
                r2[g]["words"] = []
            tmp.write_text(json.dumps(r2))
            b = cliff_benchmark(eps, Lexicon(tmp))
            out[g] = {k: b[k] for k in ("hit@1", "hit@3", "MRR", "mean_percentile")}
        # sound-cue leak check: drop every SFX event entirely
        r2 = json.loads(json.dumps(raw))
        tmp.write_text(json.dumps(r2))
        lex = Lexicon(tmp)
        saved = {e.num: e.events for e in eps}
        for e in eps:
            e.events = [ev for ev in e.events if ev.kind != "sfx"]
        b = cliff_benchmark(eps, lex)
        out["ALL SOUND CUES REMOVED"] = {k: b[k] for k in ("hit@1", "hit@3", "MRR", "mean_percentile")}
        for e in eps:
            e.events = saved[e.num]
    finally:
        tmp.unlink(missing_ok=True)
    return out


def audit_run(eps, run_dir):
    recs = json.loads((Path(run_dir) / "promos.json").read_text())
    E = {e.num: e for e in eps}
    checks = {"duration_20_60": 0, "shot_aligned": 0, "no_half_cut_line": 0, "story_order": 0, "in_bounds": 0}
    fails = []
    for r in recs:
        segs = r["segments"]
        ok = {}
        ok["duration_20_60"] = 20 - 1e-6 <= sum(s["end_s"] - s["start_s"] for s in segs) <= 60 + 1e-6
        al, line, bnd = True, True, True
        for s in segs:
            ep = E[s["episode"]]
            ins, outs = cut_points(ep)
            al &= any(abs(s["start_s"] - x) < 0.01 for x in ins) and any(abs(s["end_s"] - x) < 0.01 for x in outs)
            for ev in ep.events:
                if ev.kind == "dialogue" and (ev.start + 0.05 < s["start_s"] < ev.end - 0.05 or
                                              ev.start + 0.05 < s["end_s"] < ev.end - 0.05):
                    line = False
            bnd &= 0 <= s["start_s"] < s["end_s"] <= ep.duration + 0.01
        ok["shot_aligned"], ok["no_half_cut_line"], ok["in_bounds"] = al, line, bnd
        order = [(s["episode"], s["start_s"]) for s in segs]
        ok["story_order"] = order == sorted(order)
        for k, v in ok.items():
            checks[k] += bool(v)
            if not v:
                fails.append({"id": r["id"], "check": k})
    n = len(recs)
    return {"promos": n, "pass_rate": {k: f"{v}/{n}" for k, v in checks.items()}, "failures": fails}


def _spearman(a, b):
    def rk(x):
        o = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rk(a), rk(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return round(num / den, 3) if den else None


def llm_agreement(run_dir):
    recs = json.loads((Path(run_dir) / "all_candidates.json").read_text())["ranked"]
    rows = [r for r in recs if r["scores"].get("llm_hook") is not None]
    if len(rows) < 5:
        return {}
    out = {"n": len(rows),
           "spearman_hook(rules,llm)": _spearman([r["scores"]["hook"] for r in rows], [r["scores"]["llm_hook"] for r in rows]),
           "spearman_cliff(rules,llm)": _spearman([r["scores"]["cliff"] for r in rows], [r["scores"]["llm_cliff"] for r in rows])}
    m = [r["stranger_match_0_2"] for r in rows if r["stranger_match_0_2"] is not None]
    if m:
        out["stranger_test"] = {"got_it (2)": sum(x == 2 for x in m), "partly (1)": sum(x == 1 for x in m),
                                "lost (0)": sum(x == 0 for x in m)}
    return out


def determinism(input_dir):
    from .candidates import singles, supercuts
    from .cast import main_characters
    sigs = []
    for seed in (1, 2):
        eps = load_show(input_dir)
        random.Random(seed).shuffle(eps)            # input order must not matter
        eps.sort(key=lambda e: e.num)
        build_cast(eps)
        lex = Lexicon()
        sg = {e.num: annotate(e, lex) for e in eps}
        S = singles(eps, sg)
        X = supercuts(eps, sg, main_characters(eps))
        sigs.append(json.dumps([[(s.ep, s.start, s.end) for s in c.segments] + [c.rule_score] for c in S + X]))
    return sigs[0] == sigs[1]


def evaluate(input_dir, out_dir, run_dir=None, provider="none", model=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eps = load_show(input_dir)
    build_cast(eps)
    lex = Lexicon()
    rep = {"A_cliff_all_50_endings": cliff_benchmark(eps, lex),
           "A_cliff_writer_confirmed": cliff_benchmark(eps, lex, labelled_only=True),
           "A_cliff_heldout_26_50": cliff_benchmark(eps, lex, only=set(range(26, 51)))}
    tuned = Path(__file__).resolve().parent.parent / "config" / "cliff_weights.json"
    if tuned.exists():
        untuned = Lexicon(cliff_weights=None)
        rep["A_cliff_heldout_26_50_UNTUNED"] = cliff_benchmark(eps, untuned, only=set(range(26, 51)))
    rep["B_ablations_all_50"] = ablations(eps)
    rep["E_deterministic"] = determinism(input_dir)
    if provider != "none":
        from .llm import LLM
        rep["F_cliff_choice_heldout"] = llm_cliff_benchmark(eps, LLM(provider, model), lex)
    if run_dir:
        rep["C_audit"] = audit_run(eps, run_dir)
        rep["D_llm"] = llm_agreement(run_dir)
    (out / "eval_report.json").write_text(json.dumps(rep, indent=2))
    L = ["# Evaluation", ""]
    for key in [k for k in rep if k.startswith("A_")]:
        a = rep[key]
        L += [f"## {key}  (n={a['episodes']} episodes, ~{st.mean(r['n_cut_points'] for r in a['per_episode']):.0f} cut points each)",
              "| metric | tool (blind) | random |", "|---|---|---|"]
        for m in ("hit@1", "hit@3", "hit@5", "MRR", "mean_percentile"):
            L.append(f"| {m} | {a[m]} | {a['random_' + m]} |")
        L.append("")
    L += ["## B. Ablations (remove one signal group, all 50 endings)", "| removed | hit@1 | hit@3 | MRR | mean pct |", "|---|---|---|---|---|"]
    for g, v in rep["B_ablations_all_50"].items():
        L.append(f"| {g} | {v['hit@1']} | {v['hit@3']} | {v['MRR']} | {v['mean_percentile']} |")
    L += ["", f"## E. Deterministic rules output: {rep['E_deterministic']}"]
    if run_dir:
        L += ["", f"## C. Editing-rule audit of {run_dir}", json.dumps(rep["C_audit"]["pass_rate"]),
              f"failures: {rep['C_audit']['failures']}", "", "## D. Rules vs LLM", json.dumps(rep.get("D_llm", {}), indent=1)]
    if "F_cliff_choice_heldout" in rep:
        f = rep["F_cliff_choice_heldout"]
        L += ["", f"## F. Pick the real episode ending from 4 blind options (held-out eps 26-50, {f['model']})",
              f"chance {f['chance']} · rules {f['rules_acc']} · LLM {f['llm_acc']} (LLM top-2: {f['llm_top2']}) · both wrong: {f['both_wrong']}/{f['episodes']}"]
    (out / "eval_report.md").write_text("\n".join(L))
    print("\n".join(L))
    return rep


# ---------------------------------------------------------------- LLM benchmark
import re as _re

_BLIND_RE = _re.compile(r"(cliffhanger|cut(s|ting)? (abruptly )?to black|fade[sd]? (out|to black)|to black|"
                        r"until fade|holding until|final (shot|frame|beat)|end(s|ing)? (of|the) episode)", _re.I)

CHOICE_SYS = """You are a microdrama story editor. Each option is the last ~20 seconds of a clip from the same episode.
Exactly one option ends where the writers ended the episode on a deliberate cliffhanger. Which ending leaves the
strongest unresolved question that forces the viewer to watch the next scene? Return ONLY JSON."""
CHOICE_USER = """{opts}

Return JSON {{"ranking": ["A","B","C","D"] best cliffhanger first, "reason": "<one line>"}}"""


def _blind_transcript(ep, t0, t1):
    from .judge import transcript
    from .candidates import Segment
    txt = transcript({ep.num: ep}, [Segment(ep.num, t0, t1)], max_cam=140)
    return _BLIND_RE.sub("…", txt)


def llm_cliff_benchmark(eps, llm, lex, test=range(26, 51), n_decoys=3, win=20.0, seed=11):
    from .candidates import cut_points
    rng = random.Random(seed)
    rows = []
    for ep in eps:
        if ep.num not in set(test):
            continue
        _, outs = cut_points(ep)
        gold = max(outs)
        pool = [t for t in outs if win <= t <= gold - 12.0]
        if len(pool) < n_decoys:
            continue
        decoys = rng.sample(pool, n_decoys)
        opts = [gold] + decoys
        rng.shuffle(opts)
        letters = "ABCD"[:len(opts)]
        sigs = annotate(ep, lex, blind=True)
        rule_scores = {L: _cliff_at(ep, sigs, t) for L, t in zip(letters, opts)}
        rule_pick = max(letters, key=lambda L: (rule_scores[L], -opts[letters.index(L)]))  # tie -> earliest (no end bias)
        text = "\n\n".join(f"OPTION {L}:\n{_blind_transcript(ep, max(0, t - win), t)}" for L, t in zip(letters, opts))
        r = llm.chat_json(CHOICE_SYS, CHOICE_USER.format(opts=text), temperature=0.0)
        rank = [str(x).strip().upper()[:1] for x in r.get("ranking", []) if str(x).strip()]
        gL = letters[opts.index(gold)]
        rows.append({"ep": ep.num, "gold": gL, "llm_pick": rank[0] if rank else None, "rule_pick": rule_pick,
                     "llm_rank_of_gold": (rank.index(gL) + 1) if gL in rank else None, "reason": r.get("reason", "")})
    n = len(rows)
    return {"episodes": n, "chance": round(1 / (n_decoys + 1), 3),
            "rules_acc": round(sum(r["rule_pick"] == r["gold"] for r in rows) / n, 3),
            "llm_acc": round(sum(r["llm_pick"] == r["gold"] for r in rows) / n, 3),
            "llm_top2": round(sum((r["llm_rank_of_gold"] or 9) <= 2 for r in rows) / n, 3),
            "both_wrong": sum(r["llm_pick"] != r["gold"] and r["rule_pick"] != r["gold"] for r in rows),
            "model": llm.tag, "rows": rows}
