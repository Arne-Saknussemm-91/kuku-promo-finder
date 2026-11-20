"""One test every scorer can take: 4 blind endings, pick the real one.

Each episode gives one item: the window before the writers' real ending, plus
3 decoy windows from legal cut points earlier in the same episode. All of it
blind -- label words, end-of-episode words and position info are stripped.

Why this test and not the full ~27-way ranking: an LLM can answer it in one
short prompt, so the learned model, the keyword rules and any model all take
the identical test and land on one comparable table. Chance is 25%.

Option sets are seeded per episode, so answers cache and any subset of
episodes stays comparable with any other run.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from . import bow
from .candidates import Segment, cut_points
from .evaluate import _BLIND_RE, _cliff_at
from .features import Lexicon, annotate
from .judge import transcript

WIN = 20.0
LETTERS = "ABCD"

SYS = """You are a microdrama story editor. Each option is the last ~20 seconds of a clip from the same episode.
Exactly one ends where the writers ended the episode, on a deliberate cliffhanger. Which ending leaves the strongest
unresolved question, the one that forces the viewer into the next scene? Return ONLY JSON."""
USER = """{opts}

Return JSON {{"ranking": ["A","B","C","D"] best cliffhanger first, "reason": "<one line>"}}"""


def items(eps, n_decoys=3, win=WIN):
    """[(ep_num, [times...], gold_letter)] -- deterministic per episode."""
    out = []
    for ep in eps:
        _, outs = cut_points(ep)
        if not outs:
            continue
        gold = max(outs)
        pool = [t for t in outs if win <= t <= gold - 12.0]
        if len(pool) < n_decoys:
            continue
        rng = random.Random(1000 + ep.num)          # per-episode seed: stable, cache-friendly
        opts = [gold] + rng.sample(pool, n_decoys)
        rng.shuffle(opts)
        out.append((ep.num, opts, LETTERS[opts.index(gold)]))
    return out


def blind_text(ep, t):
    return _BLIND_RE.sub("…", transcript({ep.num: ep}, [Segment(ep.num, max(0.0, t - WIN), t)], max_cam=140))


def ask_llm(llm, eps, its):
    E = {e.num: e for e in eps}
    picks = {}
    for num, opts, gold in its:
        text = "\n\n".join(f"OPTION {L}:\n{blind_text(E[num], t)}" for L, t in zip(LETTERS, opts))
        r = llm.chat_json(SYS, USER.format(opts=text), temperature=0.0)
        rank = [str(x).strip().upper()[:1] for x in (r.get("ranking") or []) if str(x).strip()]
        picks[num] = {"pick": rank[0] if rank else None,
                      "rank_of_gold": (rank.index(gold) + 1) if gold in rank else None,
                      "reason": r.get("reason", "")}
    return picks


def keyword_picks(eps, its, lex=None):
    lex = lex or Lexicon()
    E = {e.num: e for e in eps}
    out = {}
    for num, opts, gold in its:
        ep = E[num]
        sig = annotate(ep, lex, blind=True)
        sc = {L: _cliff_at(ep, sig, t) for L, t in zip(LETTERS, opts)}
        out[num] = max(LETTERS, key=lambda L: (sc[L], -opts[LETTERS.index(L)]))
    return out


def model_picks(eps, its, folds=5, seed=7, l2=2.0):
    """Fold-held-out: the model never sees the episode it is tested on."""
    from .benchmark import folds as make_folds
    E = {e.num: e for e in eps}
    drop = bow._names(eps)
    out = {}
    for fold in make_folds(eps, folds, seed):
        rest = {e.num for e in eps} - fold
        m = bow.train(eps, "cliff", train_eps=rest, l2=l2, seed=seed)
        for num, opts, gold in its:
            if num not in fold:
                continue
            ep = E[num]
            sc = {L: m.p(bow.featurize(bow.tokens(ep, max(0.0, t - WIN), t, drop)))
                  for L, t in zip(LETTERS, opts)}
            out[num] = max(LETTERS, key=lambda L: sc[L])
    return out


def score(its, picks):
    gold = {n: g for n, _, g in its}
    have = [n for n in gold if picks.get(n) is not None]
    if not have:
        return None
    acc = sum(picks[n] == gold[n] for n in have) / len(have)
    se = (acc * (1 - acc) / len(have)) ** 0.5
    return {"n": len(have), "accuracy": round(acc, 3), "se": round(se, 3),
            "ci95": [round(max(0, acc - 1.96 * se), 3), round(min(1, acc + 1.96 * se), 3)]}
