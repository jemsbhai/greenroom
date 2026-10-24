# Demo script (about 3 minutes inside the 10-minute slot)

One end-to-end workflow, as the brief asks: a video enters, candidate moments appear with labels, a reviewer examines a clip in context, decides, and a structured record is saved. The demo runs on the openly licensed public footage unless Dr. Umapathy has confirmed that the provided footage may be shown on screen; the numbers on the slides come from the provided footage either way.

Before the session: laptop plugged in, high-performance power plan, Ollama running, the review app started on 127.0.0.1:8501, a terminal open in the repo folder, the windows and records for the demo video already built (the pipeline run is shown from the terminal log, not re-run live).

1. **A video enters** (20 s). Terminal: `python -m greenroom windows --subclips` output scrolled to the demo video's line: duration, number of 5-second windows. One sentence: every window gets scored by five channels, two audio, two vision, one transcript, all on this laptop.
2. **The pipeline run** (30 s). Terminal: the tail of `python -m greenroom run --video <demo>`: channels, primary label counts, review status split, processing seconds for minutes of video. Point at the line "X video-min per wall-min".
3. **The timeline and the queues** (30 s). Review app: sidebar shows the status split; pick "Confident suggestions" and show two windows that are plainly right (a dance block, a performance). Then switch to "Review required": the first item's "Why this is in the queue" line reads, for example, "conflict 0.52 >= 0.4": sound and picture disagree, so a human decides.
4. **A reviewer examines a clip in context** (40 s). Play the clip; toggle 5 s of context; open "Evidence behind the suggestion": the fused table and the per-channel table. One sentence: belief, disbelief and uncertainty instead of a single score; a channel that saw nothing says so with uncertainty 1.0.
5. **Decide** (30 s). Accept one window. On the next, open Edit, change the primary label, add a staff observation, Save edit. Defer a third. Show the toast each time; the queue advances.
6. **The record and the ledger** (30 s). Terminal: `python -m greenroom verify` shows the ledger with N decisions and "chain intact". Open the exported JSON from the app (download button) or `records.jsonld` in an editor: the brief's fields (video id, start and end, primary and secondary labels, confidence, origin, reviewer decision, guideline version, review date) plus the opinions and provenance with the git commit.
7. **Session summary** (20 s). The summary expander: minutes per label for the session, decisions so far, median seconds per decision. One sentence: session-level only, nothing about individuals, by construction.

If anything fails live: the recorded run of the same script on the public footage is on the desktop; the ledger and report files are static and open in any editor.
