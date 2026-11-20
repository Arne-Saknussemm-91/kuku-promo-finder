"""Pretrained static sentence embeddings as a third scorer.

Why bother: the word lists and the bag-of-words model both only know words
that appear in THIS show. A new show (mafia, werewolf, CEO) uses different
words for the same beats. Pretrained embeddings carry meaning, so a model
trained on one show has a chance of transferring to another. We cannot prove
that with one show in hand -- stated plainly in the docs -- but we can test
whether it beats the other scorers on held-out episodes of this one.

Model: minishlab/potion-base-8M (static, ~30MB, CPU, no torch).
A window becomes the mean of its line vectors; a logistic regression on top.
Optional: if the model is not installed, everything here degrades to None and
the benchmark simply skips the embedding rows.
"""
from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "emb"
_MODEL = None
MODEL_NAME = "minishlab/potion-base-8M"


def available() -> bool:
    try:
        import model2vec, numpy  # noqa: F401
        return True
    except ImportError:
        return False


def model():
    global _MODEL
    if _MODEL is None:
        from model2vec import StaticModel
        _MODEL = StaticModel.from_pretrained(MODEL_NAME)
    return _MODEL


def window_lines(ep, t0, t1, drop: set[str]) -> list[str]:
    """Plain-English lines inside a window, with character names removed so the
    model cannot learn 'this show' instead of 'this kind of beat'."""
    out = []
    for ev in ep.events:
        if ev.end <= t0 or ev.start >= t1:
            continue
        txt = ev.text
        for n in drop:
            if len(n) > 2:
                txt = txt.replace(n.title(), "someone").replace(n.upper(), "someone")
        if ev.kind == "dialogue":
            out.append(f"someone says ({ev.emotion.lower()}): {txt}")
        else:
            out.append(f"{ev.kind}: {txt}")
    return out


def embed_windows(items, drop, batch=256):
    """items: [(key, ep, t0, t1)] -> {key: vector}. Cached on disk by content."""
    import numpy as np
    CACHE.mkdir(parents=True, exist_ok=True)
    texts, keys = [], []
    for k, ep, t0, t1 in items:
        texts.append(" \n".join(window_lines(ep, t0, t1, drop)) or "empty")
        keys.append(k)
    h = hashlib.sha256(("|".join(texts) + MODEL_NAME).encode()).hexdigest()[:20]
    f = CACHE / f"{h}.npy"
    if f.exists():
        V = np.load(f)
    else:
        V = model().encode(texts, batch_size=batch)
        np.save(f, V)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return dict(zip(keys, V))


class DenseLogReg:
    """Logistic regression on dense vectors, class-balanced inside training."""

    def __init__(self, l2=1.0, epochs=300, lr=0.5):
        self.l2, self.epochs, self.lr = l2, epochs, lr

    def fit(self, X, y, seed=0):
        import numpy as np
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        n, d = X.shape
        self.w = np.zeros(d, dtype="float32")
        self.b = 0.0
        pos, neg = max(1, y.sum()), max(1, n - y.sum())
        cw = np.where(y > 0, n / (2 * pos), n / (2 * neg)).astype("float32")
        rng = np.random.default_rng(seed)
        for _ in range(self.epochs):
            idx = rng.permutation(n)
            Xi, yi, ci = X[idx], y[idx], cw[idx]
            p = 1 / (1 + np.exp(-np.clip(Xi @ self.w + self.b, -30, 30)))
            g = (p - yi) * ci
            self.w -= self.lr * ((Xi.T @ g) / n + self.l2 * self.w / n)
            self.b -= self.lr * g.mean()
        return self

    def p(self, x):
        import numpy as np
        return float(1 / (1 + math.exp(-max(-30.0, min(30.0, float(np.dot(self.w, x)) + self.b)))))
