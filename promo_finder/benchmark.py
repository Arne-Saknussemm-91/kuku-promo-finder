"""Honest comparison of cliffhanger scorers on proper splits.

Task: among every legal cut point in an episode (~27), rank the one the
writers actually ended on. Blind -- no position info, no end-of-episode words.

Scorers compared:
  keyword   : hand-written word lists, weights fixed at 1.0
  keyword+  : same lists, 8 weights fitted on the TRAIN split only
  bow       : learned bag-of-words logistic regression (promo_finder/bow.py)
  bow+kw    : average of the two ranks

Splits (episodes are the unit, shuffled with a fixed seed):
  80 / 5 / 15  train / valid / test   -- valid picks hyperparameters
  5-fold CV                            -- every episode tested once; the
                                          stable number when n is only 50
"""
from __future__ import annotations

import json
import random
import statistics as st
from pathlib import Path

from . import bow, embed
from .candidates import cut_points
from .cast import build_cast
from .evaluate import _cliff_at
from .features import Lexicon, annotate
from .parse import load_show

ROOT = Path(__file__).resolve().parent.parent


def splits(eps, seed=7, frac=(0.80, 0.05, 0.15)):
    nums = [e.num for e in eps]
    random.Random(seed).shuffle(nums)
    n = len(nums)
    a = round(frac[0] * n)
    b = a + max(1, round(frac[1] * n))
    return set(nums[:a]), set(nums[a:b]), set(nums[b:])


def folds(eps, k=5, seed=7):
    nums = [e.num for e in eps]
    random.Random(seed).shuffle(nums)
    return [set(nums[i::k]) for i in range(k)]


def rank_metrics(eps, which, score_fn, min_t=20.0):
    """score_fn(ep, t) -> higher means 'more likely the real ending'."""
    rows = []
    for ep in eps:
        if ep.num not in which:
            continue
        _, outs = cut_points(ep)
        outs = [t for t in outs if t >= min_t]
        if len(outs) < 2:
            continue
        gold = max(outs)
        sc = {t: score_fn(ep, t) for t in outs}
        g = sc[gold]
        rank = 1 + sum(1 for t, v in sc.items() if v > g) + sum(1 for t, v in sc.items() if v == g and t != gold) / 2
        rows.append({"ep": ep.num, "n": len(outs), "rank": rank})
    n = len(rows)
    return {"episodes": n,
            "hit@1": round(sum(r["rank"] <= 1 for r in rows) / n, 3),
            "hit@3": round(sum(r["rank"] <= 3 for r in rows) / n, 3),
            "hit@5": round(sum(r["rank"] <= 5 for r in rows) / n, 3),
            "MRR": round(sum(1 / r["rank"] for r in rows) / n, 3),
            "random_MRR": round(st.mean(sum(1 / k for k in range(1, r["n"] + 1)) / r["n"] for r in rows), 3),
            "rows": rows}


def kw_scorer(eps, lex):
    cache = {e.num: annotate(e, lex, blind=True) for e in eps}
    return lambda ep, t: _cliff_at(ep, cache[ep.num], t)


def bow_scorer(model, win=20.0):
    return lambda ep, t: bow.score_window(model, ep, max(0.0, t - win), t)


def emb_scorer(eps, train, win=20.0, l2=1.0, seed=0, sound=True):
    """Train a dense classifier on pretrained embeddings of the window before each cut point."""
    from .bow import _names
    drop = {n.lower() for n in _names(eps)}
    items, y, keys = [], [], []
    for ep in eps:
        _, outs = cut_points(ep)
        gold = max(outs)
        for t in [x for x in outs if x >= win]:
            k = (ep.num, round(t, 2))
            items.append((k, ep, t - win, t))
            keys.append(k)
            y.append(1 if abs(t - gold) < 0.01 else 0)
    V = embed.embed_windows(items, drop)
    X = [V[k] for k in keys]
    tr = [i for i, k in enumerate(keys) if k[0] in train]
    m = embed.DenseLogReg(l2=l2).fit([X[i] for i in tr], [y[i] for i in tr], seed=seed)
    return lambda ep, t: m.p(V[(ep.num, round(t, 2))])


def blend(a, b, eps, which):
    """Average of the two scorers' within-episode ranks (scale-free)."""
    def f(ep, t):
        _, outs = cut_points(ep)
        outs = [x for x in outs if x >= 20.0]
        def rk(s):
            v = sorted((s(ep, x) for x in outs), reverse=True)
            return v.index(s(ep, t))
        return -(rk(a) + rk(b))
    return f


def tune_kw(eps, train, grid=(0.0, 0.5, 1.0, 2.0, 3.0), passes=2):
    from .features import CLIFF_KEYS
    w = {k: 1.0 for k in CLIFF_KEYS}
    best = rank_metrics(eps, train, kw_scorer(eps, Lexicon(cliff_weights=w)))["MRR"]
    for _ in range(passes):
        for k in CLIFF_KEYS:
            for g in grid:
                t = dict(w, **{k: g})
                v = rank_metrics(eps, train, kw_scorer(eps, Lexicon(cliff_weights=t)))["MRR"]
                if v > best + 1e-9:
                    best, w = v, t
    return w, best


def run(input_dir="input", out="outputs/benchmark", seed=7):
    eps = load_show(input_dir)
    build_cast(eps, Path("outputs/haiku/aliases.json"))
    tr, va, te = splits(eps, seed)
    rep = {"seed": seed, "split_sizes": {"train": len(tr), "valid": len(va), "test": len(te)},
           "split_episodes": {"train": sorted(tr), "valid": sorted(va), "test": sorted(te)}}
    print(f"split: train {len(tr)} / valid {len(va)} / test {len(te)} episodes")

    # --- keyword, untuned
    lex0 = Lexicon(cliff_weights=None)
    kw0 = kw_scorer(eps, lex0)
    # --- keyword, weights fitted on train
    w, trmrr = tune_kw(eps, tr)
    kw1 = kw_scorer(eps, Lexicon(cliff_weights=w))
    rep["keyword_weights_fitted_on_train"] = w
    # --- bow: pick L2 on valid
    cand, best = [], None
    for l2 in (0.1, 0.5, 2.0, 8.0):
        m = bow.train(eps, "cliff", train_eps=tr, l2=l2, seed=seed)
        v = rank_metrics(eps, va, bow_scorer(m))["MRR"]
        cand.append({"l2": l2, "valid_MRR": v})
        if best is None or v > best[0]:
            best = (v, m, l2)
    rep["bow_l2_search_on_valid"] = cand
    rep["bow_l2_chosen"] = best[2]
    bw = bow_scorer(best[1])
    print(f"bow: chose l2={best[2]} on valid (valid MRR {best[0]})")

    scorers = [("keyword", kw0), ("keyword+fitted", kw1), ("bow", bw)]
    if embed.available():
        ecand = []
        for l2 in (0.3, 1.0, 4.0):
            f = emb_scorer(eps, tr, l2=l2, seed=seed)
            ecand.append((rank_metrics(eps, va, f)["MRR"], l2, f))
        ecand.sort(key=lambda x: -x[0])
        rep["emb_l2_chosen"] = ecand[0][1]
        em = ecand[0][2]
        scorers += [("embedding", em), ("embedding+bow", blend(em, bw, eps, te))]
        print(f"embedding: chose l2={ecand[0][1]} on valid")
    scorers.append(("bow+keyword", blend(bw, kw1, eps, te)))
    for name, fn in scorers:
        rep[f"TEST_{name}"] = {k: v for k, v in rank_metrics(eps, te, fn).items() if k != "rows"}
        print(f"  TEST {name:15s} {rep[f'TEST_{name}']}")

    # --- 5-fold CV (stable estimate; every episode tested once)
    names = ["keyword", "keyword+fitted", "bow", "bow+keyword"] + (
        ["embedding", "embedding+bow", "embedding+keyword"] if embed.available() else [])
    cv = {n: [] for n in names}
    for i, fold in enumerate(folds(eps, 5, seed)):
        rest = {e.num for e in eps} - fold
        wf, _ = tune_kw(eps, rest)
        k1 = kw_scorer(eps, Lexicon(cliff_weights=wf))
        mf = bow.train(eps, "cliff", train_eps=rest, l2=best[2], seed=seed)
        b1 = bow_scorer(mf)
        pairs = [("keyword", kw0), ("keyword+fitted", k1), ("bow", b1),
                 ("bow+keyword", blend(b1, k1, eps, fold))]
        if embed.available():
            e1 = emb_scorer(eps, rest, l2=rep.get("emb_l2_chosen", 1.0), seed=seed)
            pairs += [("embedding", e1), ("embedding+bow", blend(e1, b1, eps, fold)),
                      ("embedding+keyword", blend(e1, k1, eps, fold))]
        for n, fn in pairs:
            cv[n].append(rank_metrics(eps, fold, fn))
        print(f"  fold {i + 1}/5 done")
    rep["CV5"] = {n: {m: round(st.mean(r[m] for r in rs), 3) for m in ("hit@1", "hit@3", "hit@5", "MRR", "random_MRR")}
                  for n, rs in cv.items()}
    for n, v in rep["CV5"].items():
        print(f"  CV5  {n:15s} {v}")
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "benchmark.json").write_text(json.dumps(rep, indent=1))
    bow.save(best[1], Path(out) / "bow_cliff.json")
    hi, lo = best[1].top(12)
    rep["bow_learned_words"] = {"ending": [k for k, _ in hi], "not_ending": [k for k, _ in lo]}
    (Path(out) / "benchmark.json").write_text(json.dumps(rep, indent=1))
    return rep
