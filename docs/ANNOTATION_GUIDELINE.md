# Annotation guideline, version 1.0

This guideline governs the gold labels used to evaluate greenroom. It is written for annotators who have not seen the model's output. The version string "1.0" is recorded in every label; any change of meaning bumps the version.

## 1. What you are labeling

- The unit is a fixed 5-second window of one video. You label what is visible or audible inside that window only.
- The context toggle plays 5 seconds before and after the window. Use it to understand what is going on; do not label things that happen only in the context part.
- You are labeling activities, not people. Never record a name, a nickname, or a description that would let someone identify a participant. Never record a judgment about how a participant feels.
- There is no wrong answer when you are unsure: tick "Uncertain". A window marked uncertain is still saved with your best primary label.

## 2. Primary label

Exactly one primary label per window: the activity that occupies most of the window or, if two are equal, the one that is more visible or audible. If no participant activity is present, pick one of the three non-activity labels.

Activity labels:

- Singing / vocal performance: a person sings, hums a melody, or performs a vocal line, live or along with a track. Melodic, sustained pitch is the cue. Lip-syncing to a recording in a performance counts.
- Songwriting: writing, editing or working out a song: lyrics in a notebook or on a screen, humming and revising a line, discussing a verse, playing a chord progression while trying words. A finished song being performed is Singing, not Songwriting, unless the window shows the work-in-progress.
- Lyricism / spoken word / rap: rhythmic speech performance: rap verses over a beat, spoken-word poetry, recited lyrics. Rhythm and emphasis without sustained melodic pitch is the cue. A rap performed with a sung hook in the same window gets both labels (the dominant one primary).
- Dance: choreographed or improvised movement to music or a count, by one person or a group.
- Acting / theater: a scripted or improvised scene, character dialogue, stage blocking, monologue delivered in character.
- Film / media production: operating a camera, setting up a shot, conducting or giving an on-camera interview, editing video on a computer.
- Recording arts / audio production: work in a recording booth or at a mixing desk, headphones and a studio microphone with a pop filter, a DAW on screen, a beat being built. Usually co-occurs with Singing or Lyricism; label both.
- Podcast production: seated conversation at a table with microphones in podcast style.
- Visual arts: painting, drawing, sketching, crafting, design concept work on paper or a tablet.
- Fashion design: working with fabric, sewing, draping, sketching garments, fitting a garment.
- Modeling / runway: walking a runway or posing for a camera as a model.
- Group performance / showcase piece: three or more participants performing together as one piece. Always add the specific activity as well (a group dance is primary Dance with secondary Group performance, or the reverse if the ensemble aspect dominates).

Non-activity labels (primary only when no participant activity is in the window):

- Instruction / facilitation (no participant activity in view): an instructor or facilitator explains, demonstrates, or gives notes while participants listen or wait.
- Audience / applause / transition: applause, a host or MC speaking between acts, performers walking on or off, bows.
- Setup / idle / empty room: an empty room, equipment being set up, people waiting or chatting socially, camera pointed at nothing in particular.

## 3. Secondary labels

Add every other activity label that is clearly present in the window, even briefly. Do not add a secondary label for something you only infer from context. Non-activity labels are never secondary.

## 4. Facilitation flag

Tick "Facilitation present" when an instructor, facilitator or peer is directing, teaching or giving feedback within the window, whether or not a participant activity is also present. Examples: counting in a dance, "again from the top", "try it with more breath", adjusting a camera for a participant, giving notes after a take. A participant performing without any direction in the window is not facilitation. The flag is independent of the primary label.

## 5. Edited promotional clips

Several files are edited promos with fast cuts, music beds and on-screen text. Rules:

- Label what the shots show; ignore on-screen text and captions.
- Music under a montage is not Singing unless a person is seen or clearly heard singing in the window.
- When a window spans several shots, the primary label is the shot content with the most screen time in the window; the other shots become secondary labels if they show other activities.

## 6. Uncertain

Tick "Uncertain" when, even after using context, you cannot tell what activity is happening (for example, dark footage, a back turned to the camera, audio too noisy to know whether a person is rapping or talking). Still choose your best primary label; the uncertainty tick is what the evaluation uses.

## 7. Notes

The note is optional. Use it for things the guideline does not cover ("two activities in different corners of the room", "audio and video seem out of sync"). No names, no identifying descriptions, no statements about emotion or well-being.

## 8. Process

- Two annotators label every window of every provided video independently, in gold mode, without talking about individual windows while labeling.
- When both are done, the two annotators review the windows where they disagree and agree on a final label together. The pipeline author takes no part in labeling or adjudication.
- Agreement before adjudication is reported with the results (Cohen's kappa on the primary label, per-label kappa for secondary labels and the facilitation flag).
