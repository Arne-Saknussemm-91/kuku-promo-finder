"""CLI.

  python -m promo_finder run   --input <folder> --out outputs/run1 [--provider groq --model ...]
  python -m promo_finder eval  --input <folder> --out outputs/eval
  python -m promo_finder learn --feedback feedback.json
"""
import argparse


def main():
    ap = argparse.ArgumentParser(prog="promo_finder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="find + rank promos")
    r.add_argument("--input", required=True)
    r.add_argument("--out", default="outputs/run")
    r.add_argument("--provider", default="none")
    r.add_argument("--model", default=None)
    r.add_argument("--top", type=int, default=20)
    r.add_argument("--shortlist", type=int, default=40)
    r.add_argument("--lexicon", default=None)
    r.add_argument("--weights", default=None)
    r.add_argument("--fps", type=int, default=30)
    r.add_argument("--video-pattern", default="episode-{ep:03d}.mp4")
    e = sub.add_parser("eval", help="measure quality against writer-labelled ground truth")
    e.add_argument("--input", required=True)
    e.add_argument("--out", default="outputs/eval")
    e.add_argument("--run", default=None, help="a run folder to audit (promos.json)")
    e.add_argument("--provider", default="none")
    e.add_argument("--model", default=None)
    l = sub.add_parser("learn", help="refit rule weights from editor approve/reject feedback")
    l.add_argument("--feedback", required=True)
    l.add_argument("--candidates", required=True, help="all_candidates.json of the run the feedback came from")
    l.add_argument("--out", default="config/weights.json")
    t = sub.add_parser("train", help="train the cliffhanger model on a show's own episode endings")
    t.add_argument("--input", required=True)
    t.add_argument("--out", default="config/model_cliff.json")
    t.add_argument("--l2", type=float, default=2.0)
    a = ap.parse_args()
    if a.cmd == "run":
        from .pipeline import run
        run(a.input, a.out, a.provider, a.model, a.top, a.shortlist, a.weights, a.lexicon, a.fps, a.video_pattern)
    elif a.cmd == "eval":
        from .evaluate import evaluate
        evaluate(a.input, a.out, a.run, a.provider, a.model)
    elif a.cmd == "train":
        from .bow import train, save, _names
        from .cast import build_cast
        from .parse import load_show
        eps = load_show(a.input)
        build_cast(eps)
        m = train(eps, "cliff", train_eps=[e.num for e in eps], l2=a.l2)
        save(m, a.out)
        hi, _ = m.top(10)
        print(f"trained on {len(eps)} episodes -> {a.out} ({len(m.w)} features)")
        print("strongest ending signals:", ", ".join(k for k, _ in hi))
        print("NOTE: a model trained on a confidential pack carries its vocabulary. Keep it out of public repos.")
    elif a.cmd == "learn":
        from .learn import learn
        learn(a.feedback, a.candidates, a.out)


if __name__ == "__main__":
    main()
