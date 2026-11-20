"""Streamlit front end for promo_finder.

Deploy: push to GitHub -> share.streamlit.io -> pick the repo -> main file app.py.
Runs free with no key (learned cliffhanger model); a key unlocks the LLM steps.
The Kuku pack is NOT bundled: upload your own scripts or use the public demo show.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")


@st.cache_data(show_spinner=False)
def run_pipeline(folder: str, provider: str, model: str | None, key: str | None, top: int):
    if key:
        os.environ[{"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY",
                    "together": "TOGETHER_API_KEY"}.get(provider, "PROMO_LLM_API_KEY")] = key
    out = Path(tempfile.mkdtemp()) / "run"
    from promo_finder.pipeline import run
    logs: list[str] = []
    recs, meta = run(folder, str(out), provider=provider, model=model or None, top_k=top, log=logs.append)
    scripts = {}
    if recs:
        # the paper edit: every line the viewer actually sees, middle included
        from promo_finder.candidates import Segment
        from promo_finder.cast import build_cast
        from promo_finder.judge import transcript
        from promo_finder.parse import load_show
        eps = load_show(folder)
        build_cast(eps, Path(out) / "aliases.json")
        E = {e.num: e for e in eps}
        for r in recs:
            segs = [Segment(x["episode"], x["start_s"], x["end_s"]) for x in r["segments"]]
            scripts[r["id"]] = transcript(E, segs, max_cam=160)
    return recs, meta, str(out), logs, scripts


st.title("🎬 Promo Finder")
st.caption("Reads timestamped microdrama screenplays and returns ranked, cut-ready 20–60 s promos — "
           "each with exact in/out points, the hook, the cliffhanger and why it works.")

with st.sidebar:
    st.header("1 · Scripts")
    src = st.radio("Input", ["Demo show (included)", "Upload a .zip of scripts"], label_visibility="collapsed")
    up = st.file_uploader("Zip of .txt screenplays", type="zip") if src.startswith("Upload") else None

    st.header("2 · Model")
    st.caption("Leave as *none* to run free: the learned cliffhanger model needs no key.")
    provider = st.selectbox("Provider", ["none", "groq", "gemini", "openai", "anthropic", "openrouter",
                                         "together", "ollama"])
    model = st.text_input("Model id (optional)", placeholder="llama-3.3-70b-versatile") if provider != "none" else None
    key = st.text_input("API key", type="password",
                        help="Sent only to that provider, never stored.") if provider not in ("none", "ollama") else None
    top = st.slider("How many promos", 5, 30, 20)
    go = st.button("Find promos", type="primary", use_container_width=True)

    st.divider()
    st.caption("Code: every cut lands on a shot change and never splits a line. "
               "Model: hook scouting, blind judging, ranking. "
               "[Repo](https://github.com/Arne-Saknussemm-91/kuku-promo-finder)")

if go:
    if src.startswith("Upload") and not up:
        st.error("Upload a zip, or switch to the demo show.")
        st.stop()
    if src.startswith("Upload"):
        tmp = Path(tempfile.mkdtemp()) / "scripts"
        tmp.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(up.getvalue())) as z:
            for n in z.namelist():
                if n.lower().endswith(".txt") and not n.startswith("__MACOSX"):
                    (tmp / Path(n).name).write_bytes(z.read(n))
        folder = str(tmp)
        n_files = len(list(tmp.glob("*.txt")))
        if not n_files:
            st.error("No .txt screenplays found in that zip.")
            st.stop()
    else:
        folder = str(ROOT / "examples" / "demo_show")

    with st.spinner("Parsing, finding every legal cut, scoring…"):
        try:
            recs, meta, outdir, logs, scripts = run_pipeline(folder, provider, model, key, top)
        except SystemExit as e:
            st.error(str(e))
            st.stop()
    if recs is None:
        st.error("The run needs answers it could not get. With a provider selected, check the key.")
        st.stop()

    c = st.columns(5)
    c[0].metric("Episodes", meta["episodes"])
    c[1].metric("Cut-safe candidates", meta["candidates"])
    c[2].metric("Promos", len(recs))
    c[3].metric("Model", meta["model"].split(":")[-1][:18])
    c[4].metric("Runtime", f"{meta['runtime_s']}s")

    st.subheader("Ranked promos")
    for r in recs:
        segs = " + ".join(f"Ep{s['episode']} {s['start']}–{s['end']}" for s in r["segments"])
        with st.expander(f"**#{r['rank']}** · {segs} · {r['duration_s']:.0f}s · {r['type']}", expanded=r["rank"] <= 3):
            a, b = st.columns(2)
            a.markdown(f"**Hook (0–3 s)**\n\n{r['hook']}")
            b.markdown(f"**Cliffhanger**\n\n{r['cliffhanger']}")
            st.markdown(f"**Why it works** · {r['why']}")
            if r.get("stranger_summary"):
                st.caption(f"Stranger test ({r['stranger_match_0_2']}/2): “{r['stranger_summary']}”")
            if r.get("confusing_for_new_viewer") and r["confusing_for_new_viewer"].lower() != "nothing":
                st.caption(f"Might confuse a newcomer: {r['confusing_for_new_viewer']}")
            st.markdown("**The whole clip — every line the viewer sees**")
            st.code(scripts.get(r["id"], ""), language=None)

    st.subheader("Take it away")
    o = Path(outdir)
    d = st.columns(4)
    d[0].download_button("promos.csv", (o / "promos.csv").read_bytes(), "promos.csv", use_container_width=True)
    d[1].download_button("promos.json", (o / "promos.json").read_bytes(), "promos.json", use_container_width=True)
    d[2].download_button("board.html", (o / "board.html").read_bytes(), "board.html",
                         help="Interactive board: series heat-map, paper edits, approve/skip", use_container_width=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in (o / "edl").glob("*.edl"):
            z.write(f, f.name)
        z.write(o / "cut_promos.sh", "cut_promos.sh")
    d[3].download_button("EDLs + ffmpeg", buf.getvalue(), "edl.zip",
                         help="CMX3600 edit lists for Premiere / Resolve", use_container_width=True)

    with st.expander("Run log"):
        st.code("\n".join(logs))
else:
    st.info("Pick an input on the left and press **Find promos**. "
            "The demo show is two synthetic episodes written for this repo — the Kuku pack is confidential "
            "and is not bundled here.")
    st.markdown("**What you get:** every cut lands on a shot change and never splits a line of dialogue, "
                "20–60 s, ranked, with EDLs an editor can import. "
                "Quality numbers are in [RESULTS.md]"
                "(https://github.com/Arne-Saknussemm-91/kuku-promo-finder/blob/main/docs/RESULTS.md).")
