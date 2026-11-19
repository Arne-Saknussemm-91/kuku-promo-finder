"""Learn cliffhanger weights from the writers' own episode endings.

Train on episodes 1-25 only (coordinate search, objective = MRR of the real
ending among all legal cut points, blind). Episodes 26-50 stay untouched for
the honest test number reported by `eval`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .cast import build_cast
from .evaluate import cliff_benchmark
from .features import CLIFF_KEYS, Lexicon
from .parse import load_show

GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


def tune(input_dir, out="config/cliff_weights.json", train=range(1, 26)):
    eps = load_show(input_dir)
    build_cast(eps)
    tr = set(train)
    w = {k: 1.0 for k in CLIFF_KEYS}

    def obj(wt):
        b = cliff_benchmark(eps, Lexicon(cliff_weights=wt), only=tr)
        return b["MRR"] + 0.25 * b["mean_percentile"]
    best = obj(w)
    print(f"start train objective {best:.3f}")
    for p in range(3):
        for k in CLIFF_KEYS:
            for g in GRID:
                t = dict(w, **{k: g})
                v = obj(t)
                if v > best + 1e-6:
                    best, w = v, t
        print(f"pass {p + 1}: {best:.3f} {w}")
    Path(out).write_text(json.dumps({"_note": f"learned on episodes {min(tr)}-{max(tr)} only; test on the rest", **w}, indent=1))
    test = set(range(1, 51)) - tr
    for name, lx in (("untuned", Lexicon(cliff_weights=None)), ("tuned", Lexicon(cliff_weights=w))):
        b = cliff_benchmark(eps, lx, only=test)
        print(f"HELD-OUT {name:8s} hit@1 {b['hit@1']} hit@3 {b['hit@3']} hit@5 {b['hit@5']} MRR {b['MRR']} "
              f"pct {b['mean_percentile']}  | random hit@3 {b['random_hit@3']} MRR {b['random_MRR']}")
    return w


if __name__ == "__main__":
    tune(sys.argv[1] if len(sys.argv) > 1 else "input")
