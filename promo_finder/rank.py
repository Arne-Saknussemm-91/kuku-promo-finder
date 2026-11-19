"""Combine rule + LLM evidence, then pick a diverse posting order.

final = 0.35 * rules (percentile, 0-10) + 0.65 * llm quality (0-10)
        + stranger-test bonus + funnel bonus - platform-risk penalty
funnel: a promo from episode 3 delivers its payoff after 3 episodes, one
        from episode 45 after 45 -> early promos convert better (assumption,
        stated in docs). Diversity: MMR over story words, characters and
        the emotion being sold, so the first week of posts tests different
        angles instead of five versions of the same scene.
"""
from __future__ import annotations

from .candidates import Candidate


def _pct(vals):
    s = sorted(vals)
    n = len(s)
    return {v: 10.0 * (sum(1 for x in s if x <= v) / n) for v in vals}


def score_all(cands: list[Candidate], n_eps: int, use_llm: bool):
    pr = {}
    for kind in {c.kind for c in cands}:          # singles and supercuts score on different scales
        pr.update({(kind, k): v for k, v in _pct([c.rule_score for c in cands if c.kind == kind]).items()})
    for c in cands:
        rules10 = pr[(c.kind, c.rule_score)]
        funnel = 1.0 - (c.first_ep - 1) / max(1, n_eps - 1)
        c.parts["rules10"] = round(rules10, 2)
        c.parts["funnel"] = round(funnel, 2)
        if use_llm and c.llm.get("hook_score") is not None:
            L = c.llm
            q = 0.4 * (L.get("hook_score") or 5) + 0.35 * (L.get("cliff_score") or 5) + 0.25 * (L.get("standalone_score") or 5)
            bonus = 0.5 * (L.get("stranger_match") or 0)
            risk = {"high": 2.0, "low": 0.3}.get(str(L.get("platform_risk", "none")).lower(), 0.0)
            c.parts["llm10"] = round(q, 2)
            c.final = 0.35 * rules10 + 0.65 * q + bonus + 1.0 * funnel - risk
        else:
            c.final = rules10 + 1.0 * funnel
        c.final = round(c.final, 3)


def _sim(a: Candidate, b: Candidate) -> float:
    wa, wb = getattr(a, "_words", set()), getattr(b, "_words", set())
    j = len(wa & wb) / max(1, len(wa | wb))
    seg = any(x.ep == y.ep and min(x.end, y.end) > max(x.start, y.start) for x in a.segments for y in b.segments)
    emo = 0.15 if a.llm.get("emotion") and a.llm.get("emotion") == b.llm.get("emotion") else 0.0
    return min(1.0, j * 2.0 + (0.8 if seg else 0.0) + emo)


def diverse_order(cands: list[Candidate], k: int = 25, lam: float = 2.5) -> list[Candidate]:
    pool = sorted(cands, key=lambda c: -c.final)
    picked = []
    while pool and len(picked) < k:
        best, bv = None, -1e9
        for c in pool:
            pen = max((_sim(c, p) for p in picked), default=0.0)
            v = c.final - lam * pen
            if v > bv:
                best, bv = c, v
        picked.append(best)
        pool.remove(best)
    return picked
