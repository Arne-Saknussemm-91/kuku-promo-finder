# Results — every measurement, one page

All numbers come from the files in `outputs/`. Nothing here is typed by hand.

## 1. Find the real episode ending among ~27 legal cut points

Blind: label words, end-of-episode words and all position info removed. 5-fold cross-validation, every episode tested once.

| method | first try | top 3 | top 5 | MRR |
|---|---|---|---|---|
| random | 4% | 13% | 21% | 0.157 |
| hand-written word lists | 6% | 32% | 42% | 0.242 |
| word lists, 8 weights fitted | 24% | 56% | 74% | 0.457 |
| **bag of words (learned)** | 60% | 64% | 64% | 0.647 |
| pretrained embeddings | 52% | 64% | 74% | 0.621 |
| embeddings + bag of words | 54% | 62% | 72% | 0.648 |
| bag of words + word lists | 40% | 60% | 78% | 0.563 |
| embeddings + word lists | 46% | 68% | 82% | 0.614 |

Also run on an 80/5/15 split (40/2/8 episodes): embeddings came first (MRR 0.629) and bag of words last (0.436) — the reverse of cross-validation. With 8 test episodes one episode moves the score 12 points, so cross-validation is the number to trust.

## 2. One test every scorer takes: 4 blind endings, pick the real one

All 50 episodes, one item each. The learned model is fold-held-out, so it never sees the episode it is tested on. Option sets are seeded per episode, so every method sees identical choices.

| | correct | 95% range |
|---|---|---|
| chance | 25% | — |
| keyword word lists | 64% | 51%–77% |
| **learned bag of words** | **76%** | 64%–88% |

The two model rows below sit on a **different, smaller test set** (the 25 episodes 26-50, collected before this test was unified), so they are not directly comparable with the rows above.

| | correct | real ending in top 2 |
|---|---|---|
| chance | 25% | 50% |
| Claude Haiku 5.5 | 60% | 80% |
| Claude Opus 5.5 | 88% | 96% |

## 3. No pipeline: give a model the whole episode, ask for a promo

| check | Haiku | Opus | our tool |
|---|---|---|---|
| 20–60 s long | 9/10 | 10/10 | 20/20 |
| starts on a shot change | 6/10 | 5/10 | 20/20 |
| ends on a shot change | 1/10 | 3/10 | 20/20 |
| doesn't cut mid-sentence | 2/10 | 5/10 | 20/20 |
| **cut-ready as delivered** | **0/10** | **1/10** | **20/20** |

## 4. Editing rules, our own output

| check | result |
|---|---|
| duration 20 60 | 20/20 |
| shot aligned | 20/20 |
| no half cut line | 20/20 |
| story order | 20/20 |
| in bounds | 20/20 |

## 5. Does a stranger understand it

The judge's guess of the story, compared against the writers' logline: **17/20 got it**, 3 partly, 0 lost.

## 6. Is the judge's scale trustworthy (control)

Professionally written episode openings, judged with the identical prompt:

| | hook | cliffhanger | standalone |
|---|---|---|---|
| professional episode openings | 4.8 | 5.2 | 5.1 |
| our top 20 promos | 6.3 | 6.3 | 5.7 |

The low absolute numbers are the prompt ("be strict, most clips are 4-6"), not the cuts.

## 7. Is the learned model just hearing the music

Retrained with every SOUND FX line deleted:

| | first try | MRR |
|---|---|---|
| word lists, fitted | 24% → 8% | 0.457 → 0.31 |
| bag of words | 60% → 46% | 0.647 → 0.535 |

A sound cue appears in 52% of episode-ending windows but also 25% of mid-episode ones, and 82% of all sound cues happen mid-episode: a tension signal, not an end-of-episode tag.

## 8. Other checks

- Rules and the AI judge disagree on which shortlisted clip is best: rank correlation hook -0.244, cliffhanger 0.157.
- Tightening the hook window from 3 s to 1.5 s changed nothing measurable (3/53 strong scouted hooks matched either way).
- Rules output is deterministic: True. Model answers are cached, so a rerun is identical.
- Cost per 50-episode show: ~170k input / 15k output tokens ≈ $0.04 on Haiku, ≈$1.6 on Opus, $0 on Groq's free tier.
