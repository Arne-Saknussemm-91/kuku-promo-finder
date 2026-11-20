"""Learned bag-of-words scorers, trained on the show's own structure.

Why: the hand-written word lists in config/lexicon.json are my guesses. This
learns which words actually mark an ending or an opening, from labels the
scripts give us for free:

  cliffhanger : the last window of every episode IS a cliffhanger (microdrama
                episodes are written that way; 22 scripts say so outright)
  hook        : the first window of every episode IS a professional hook -- it
                has to hold a viewer who just came off the previous cliffhanger

Model: logistic regression on log(1+count) word features, L2, written by hand
(no deps). Trained on episodes 1-25, tested on 26-50, same benchmark as the
keyword rules, so the two are directly comparable.

Leak control (otherwise the numbers are meaningless):
  * drop "cliffhanger", "cut to black", "fade out" and friends
  * drop every character name, or it learns "this show" not "this hook"
  * no position features of any kind
  * class imbalance is kept in the data and handled with weights in training
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

from .features import BLIND_EXTRA
from .parse import Episode

ROOT = Path(__file__).resolve().parent.parent
LEAK = [w.lower() for w in BLIND_EXTRA] + ["cliffhanger", "hook", "episode", "script start", "script end"]
STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being this that these
those it its he she they them his her their you your i me my we our us not no yes do does did have has had will would
can could shall should may might must just than then there here what who whom which when where why how all any some
more most very into out up down over under again still only own same too s t don now o oh ah let lets get got go
going come came one""".split())
TOKEN = re.compile(r"[a-z][a-z']+")


def _names(eps) -> set[str]:
    out = set()
    for e in eps:
        for ev in e.events:
            if ev.kind == "dialogue":
                for w in TOKEN.findall((ev.char or ev.speaker).lower()):
                    out.add(w)
    return out


def tokens(ep: Episode, t0: float, t1: float, drop: set[str]) -> list[str]:
    """Words + emotion tags + cue kinds inside a window, scrubbed of leaks/names."""
    out = []
    for ev in ep.events:
        if ev.end <= t0 or ev.start >= t1:
            continue
        txt = " ".join((ev.text, ev.emotion)).lower()
        for w in LEAK:
            txt = txt.replace(w, " ")
        for w in TOKEN.findall(txt):
            w = w.rstrip("'").removesuffix("'s")       # juliana's -> juliana, so name-stripping catches it
            if len(w) > 2 and w not in STOP and w not in drop:
                out.append(f"{ev.kind[:3]}:{w}")      # a word in a camera cue != the same word spoken
        out.append(f"kind:{ev.kind}")
    return out


def featurize(toks: list[str]) -> dict[str, float]:
    c = {}
    for t in toks:
        c[t] = c.get(t, 0.0) + 1.0
    return {k: math.log1p(v) for k, v in c.items()}


class LogReg:
    def __init__(self, l2=0.5):
        self.w, self.b, self.l2 = {}, 0.0, l2

    def p(self, x):
        z = self.b + sum(self.w.get(k, 0.0) * v for k, v in x.items())
        return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))

    def fit(self, X, y, epochs=300, lr=0.3, seed=0):
        pos = sum(y) or 1
        neg = len(y) - pos or 1
        cw = {1: len(y) / (2 * pos), 0: len(y) / (2 * neg)}     # balance inside training only
        idx = list(range(len(X)))
        rng = random.Random(seed)
        for ep in range(epochs):
            rng.shuffle(idx)
            for i in idx:
                x, t = X[i], y[i]
                g = (self.p(x) - t) * cw[t]
                for k, v in x.items():
                    self.w[k] = self.w.get(k, 0.0) * (1 - lr * self.l2 / len(X)) - lr * g * v
                self.b -= lr * g
        return self

    def top(self, k=12):
        s = sorted(self.w.items(), key=lambda kv: -kv[1])
        return s[:k], s[-k:]


def _windows(eps, kind, win, train_eps, rng):
    """Positives: the real ending (or opening) window. Negatives: other legal windows."""
    from .candidates import cut_points
    drop = _names(eps)
    X, y, meta = [], [], []
    for ep in eps:
        if ep.num not in train_eps:
            continue
        ins, outs = cut_points(ep)
        if kind == "cliff":
            gold = max(outs)
            cands = [t for t in outs if t >= win]
        else:
            gold = min(ins)
            cands = [t for t in ins if t <= ep.duration - win]
        for t in cands:
            a, b = (t - win, t) if kind == "cliff" else (t, t + win)
            X.append(featurize(tokens(ep, a, b, drop)))
            y.append(1 if abs(t - gold) < 0.01 else 0)
            meta.append((ep.num, t))
    return X, y, meta


def train(eps, kind="cliff", win=20.0, train_eps=range(1, 26), l2=0.5, seed=0):
    rng = random.Random(seed)
    X, y, _ = _windows(eps, kind, win, set(train_eps), rng)
    m = LogReg(l2).fit(X, y, seed=seed)
    m.kind, m.win, m.drop = kind, win, _names(eps)
    return m


def score_window(model, ep, t0, t1):
    return model.p(featurize(tokens(ep, t0, t1, model.drop)))


def save(model, path):
    Path(path).write_text(json.dumps({"kind": model.kind, "win": model.win, "b": model.b,
                                      "w": {k: round(v, 4) for k, v in model.w.items() if abs(v) > 0.01}}, indent=0))


def load(path):
    """Load a shipped model (config/model_cliff.json) -- no training at run time."""
    d = json.loads(Path(path).read_text())
    m = LogReg()
    m.w, m.b, m.kind, m.win = d["w"], d["b"], d["kind"], d["win"]
    m.drop = set()
    return m


class Scorer:
    """Wraps a shipped model so candidates.py can ask: how cliffhanger-ish is this cut point?"""

    def __init__(self, path=None, eps=None):
        if path is None:
            # show-trained model if present (never committed: it carries show vocabulary),
            # otherwise the demo-trained one that ships with the repo
            p = ROOT / "config" / "model_cliff.json"
            path = p if p.exists() else ROOT / "config" / "model_cliff_demo.json"
        path = Path(path)
        self.model = load(path) if path.exists() else None
        self.path = path
        self.drop = _names(eps) if eps else set()
        self._cache = {}

    @property
    def ok(self):
        return self.model is not None

    def cliff(self, ep, t):
        if not self.ok:
            return 0.0
        k = (ep.num, round(t, 2))
        if k not in self._cache:
            w = self.model.win
            self._cache[k] = self.model.p(featurize(tokens(ep, max(0.0, t - w), t, self.drop)))
        return self._cache[k]
