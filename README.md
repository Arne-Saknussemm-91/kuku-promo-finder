# Promo Finder — find ready-to-cut promos in microdrama screenplays

Reads a folder of timestamped episode scripts and returns ranked 20–60 s promos
(single clips and cross-episode supercuts), each with exact cut points, the hook,
the cliffhanger, a one-line "why", and files an editor can import directly.

```
python -m promo_finder run --input <scripts folder> --out outputs/run1                  # rules only, no key, ~1 s
python -m promo_finder run --input <scripts folder> --out outputs/run1 --provider groq   # + LLM judge
python -m promo_finder eval --input <scripts folder> --run outputs/run1                  # quality report
python -m promo_finder learn --feedback feedback.json --candidates outputs/run1/all_candidates.json
```

Try it now on the public demo show (synthetic, written for this repo):
`python -m promo_finder run --input examples/demo_show --out outputs/demo` → open `outputs/demo/board.html`.

## How it works

| step | who | what |
|---|---|---|
| 1 parse | code | every CAMERA / ACTION / SOUND / dialogue cue → timed event (exact; timestamps are the deliverable) |
| 2 cast | code (+LLM) | merge labels for one person (JULIANA / LADY JULIANA); LLM proposes descriptive aliases, code rejects invented names |
| 3 signals | code | per beat: hook (slap, shock line, scream), cliff (unresolved action, reaction push-in, music sting), backstory debt, self-explaining relations. Word lists in `config/lexicon.json` → retune per genre without code |
| 4 candidates | code | every legal cut: on shot boundaries, never mid-line, 20–60 s. Supercuts join 2–3 beats from different episodes on one story thread |
| 5 compose | LLM | "trailer editor" builds supercuts from validated beat IDs only; code re-checks order, length, IDs (hallucinated IDs rejected and logged) |
| 6 judge | LLM | **blind**: sees only the clip, like a stranger on TikTok. Scores hook / cliff / standalone, names hook + cliffhanger, writes why, flags platform risk |
| 7 stranger test | LLM | separate call compares the blind judge's guess of the story to the writers' logline → measured "makes sense on its own" |
| 8 rank | code + LLM | 0.35·rules + 0.65·LLM + stranger bonus + funnel (early-episode promos first) − risk; listwise LLM ranking ×3 shuffled (cancels position bias); MMR so the first posts test different angles |
| 9 export | code | `promos.json/csv/md`, CMX3600 `edl/` per promo (Premiere/Resolve), `cut_promos.sh` (ffmpeg), interactive `board.html` |
| 10 learn | code | editor clicks Post/Skip on the board → `feedback.json` → `learn` refits weights |

## Models (swap with one flag)

`--provider none | groq | gemini | openrouter | openai | anthropic | together | ollama | custom | queue`
and `--model <id>`. Keys go in `.env` (`GROQ_API_KEY=...`). Every response is cached on disk
(`.cache/llm/`), so reruns are free and repeatable. `queue` writes prompts to `.cache/queue/`
for any agent to answer (used for testing with a Claude Code Haiku subagent; see
`promo_finder/QUEUE_AGENT.md`).

## Quality checks (`eval`)

* **A. Cliffhanger benchmark with writer ground truth.** Every episode is written to end on a
  cliffhanger (22 scripts even say so in the notes). With every label word, every
  "cut to black", and all position info removed, does the tool rank the real ending among
  ~27 legal cut points? Reported against chance, on held-out episodes 26–50.
* **B. Ablations** per signal group, plus "all sound cues removed" (leak check).
* **C. Editing-rule audit** of a run (length, shot-aligned, no half-cut lines, order, bounds).
* **D. Rules vs LLM agreement** and stranger-test results.
* **E. Determinism.**
* **F. LLM vs rules head-to-head**: pick the real episode ending from 4 blind options.

`python -m promo_finder.tune input` relearns cliff weights on episodes 1–25 only.

## Repo hygiene
The Kuku input pack and any output that quotes it are confidential: `input/` and `outputs/*`
(except the demo) are git-ignored. Share real-show outputs only through a private link.

Requires Python 3.10+. Core uses the standard library only; `pip install -r requirements.txt`
adds the official `anthropic` SDK for `--provider anthropic`. Tests: `python -m unittest discover tests`.
