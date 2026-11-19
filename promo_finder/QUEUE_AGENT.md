# Instructions for an agent answering the prompt queue (`--provider queue`)

You are standing in for an LLM API call. For EVERY file `.cache/queue/<key>.json`
that has no matching `<key>.answer.json`:

1. Read the file. It has `system` and `user` fields: that is the entire prompt.
2. Answer it exactly as the model would, following `system` and `user` only.
   Use NO other knowledge: do not open any other file, do not look at the show's
   scripts, other queue items, or earlier answers. Each prompt is independent.
3. Write ONLY the JSON object the prompt asks for (no prose, no code fences) to
   `.cache/queue/<key>.answer.json`.

Do not modify any other file. When done, reply with the number of answers written.
