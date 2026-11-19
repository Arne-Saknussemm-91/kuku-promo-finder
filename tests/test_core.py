"""python -m unittest discover tests   (runs on the public demo show; no API key needed)"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "examples" / "demo_show"
sys.path.insert(0, str(ROOT))

from promo_finder.parse import load_show, sec_to_ts           # noqa: E402
from promo_finder.candidates import cut_points               # noqa: E402
from promo_finder.export import _tc                          # noqa: E402


class Core(unittest.TestCase):
    def test_parse_counts_every_timestamp(self):
        eps = load_show(DEMO)
        n_ts = sum(p.read_text().count("Start: (") for p in DEMO.glob("*.txt"))
        self.assertEqual(sum(len(e.events) for e in eps), n_ts)

    def test_timecodes(self):
        self.assertEqual(sec_to_ts(83.2), "01:23.20")
        self.assertEqual(_tc(61.5, 30), "00:01:01:15")

    def test_cut_points_never_inside_a_line(self):
        for ep in load_show(DEMO):
            ins, outs = cut_points(ep)
            for t in ins + outs:
                for ev in ep.dialogue():
                    self.assertFalse(ev.start + 0.05 < t < ev.end - 0.05)

    def test_end_to_end_rules_only(self):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, "-m", "promo_finder", "run", "--input", str(DEMO), "--out", d],
                           cwd=ROOT, check=True, capture_output=True)
            recs = json.loads((Path(d) / "promos.json").read_text())
            self.assertTrue(recs)
            for r in recs:
                self.assertTrue(20 <= r["duration_s"] <= 60)
                self.assertTrue((Path(d) / "edl").glob("*.edl"))


if __name__ == "__main__":
    unittest.main()
