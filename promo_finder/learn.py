"""Human in the loop: the editor approves/rejects promos on board.html,
exports feedback.json, and this refits the rule weights (logistic
regression, pure python). Next run ranks the way this editor ranks.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

FEATS = ["hook", "cliff", "density", "explain", "debt"]
DEFAULT = {"hook": 1.0, "cliff": 1.0, "density": 0.25, "explain": 0.4, "debt": 0.6}


def learn(feedback_path, candidates_path, out="config/weights.json", epochs=800, lr=0.05, l2=0.01):
    fb = json.loads(Path(feedback_path).read_text())
    fb = fb.get("feedback", fb) if isinstance(fb, dict) else fb
    data = json.loads(Path(candidates_path).read_text())
    recs = {r["id"]: r for r in data["ranked"] + data["unranked_rule_candidates"]}
    X, y = [], []
    for f in fb:
        r = recs.get(f.get("id"))
        if not r or f.get("verdict") not in ("approve", "reject") or r["type"] != "single":
            continue
        s = r["scores"]
        X.append([float(s.get(k, 0) or 0) for k in FEATS])
        y.append(1.0 if f["verdict"] == "approve" else 0.0)
    if len(X) < 8 or len(set(y)) < 2:
        raise SystemExit(f"need >=8 single-clip verdicts with both approve and reject (got {len(X)})")
    mu = [sum(c) / len(c) for c in zip(*X)]
    sd = [max(1e-6, (sum((v - m) ** 2 for v in c) / len(c)) ** 0.5) for c, m in zip(zip(*X), mu)]
    Z = [[(v - m) / s for v, m, s in zip(row, mu, sd)] for row in X]
    w, b = [0.0] * len(FEATS), 0.0
    for _ in range(epochs):
        gw, gb = [0.0] * len(FEATS), 0.0
        for z, t in zip(Z, y):
            p = 1 / (1 + math.exp(-(sum(a * c for a, c in zip(w, z)) + b)))
            for i in range(len(w)):
                gw[i] += (p - t) * z[i]
            gb += p - t
        w = [wi - lr * (g / len(Z) + l2 * wi) for wi, g in zip(w, gw)]
        b -= lr * gb / len(Z)
    acc = sum((1 / (1 + math.exp(-(sum(a * c for a, c in zip(w, z)) + b))) > 0.5) == (t == 1) for z, t in zip(Z, y)) / len(y)
    # map standardized coefficients back onto the rule-score weights (debt enters with a minus sign)
    new = {}
    for k, wi, s in zip(FEATS, w, sd):
        raw = wi / s
        new[k] = round(max(0.0, -raw if k == "debt" else raw) * 0.5 + DEFAULT[k] * 0.5, 3)   # blend with prior
    Path(out).write_text(json.dumps({"_note": f"learned from {len(y)} editor verdicts, train acc {acc:.2f}", **new}, indent=1))
    print(f"learned from {len(y)} verdicts (train acc {acc:.2f}) -> {out}: {new}")
    return new
