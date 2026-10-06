# Harmonia — 和声分析器

Local tool: audio (mp3/m4a/flac) → time-aligned chords, local keys, roman numerals, and
annotated "notable harmonic events" (ii–V–I, secondary dominants, borrowed chords, tritone
subs, modulations), every judgement with a confidence and its evidence.
Target styles: J-pop / City Pop / pop. Jazz reharmonisation accuracy is a non-goal.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Symbolic analysis layer (chord symbols → keys / numerals / events) | **done, awaiting user review** |
| 2 | Evaluation framework (mir_eval chord metrics, event P/R, error attribution) | not started |
| 3 | Audio front end (source separation, beats, bass, chord recognition, Viterbi, local key) | not started — research options first, user chooses |
| 4 | UI (timeline, playback, manual chord edits → re-analysis) | not started |
| 5 | Optional LLM narration of events | not started |

Each phase ends with: run tests, report metrics, stop for user confirmation.

## Commands

```bash
python3 -m unittest discover -s tests -t .          # full test suite (stdlib only)
python3 -m harmonia analyze "| Fmaj7 | E7 | Am7 | Gm7 C7 |" --key "C major"
python3 -m harmonia analyze song.lab --json -o out.json
```
Phase 1 has no third-party dependencies; Python ≥ 3.11 (`tomllib`). Dev machine has 3.14.

## Architecture

```
audio ─▶ [recognition layer, Phase 3] ─▶ RecognitionResult JSON ─▶ [analysis layer] ─▶ AnalysisResult JSON ─▶ UI / eval
text / .lab ─▶ harmonia/io ────────────▶ RecognitionResult ─────┘
```
- **Strict decoupling**: the two layers communicate ONLY via `harmonia/schema.py`
  (documented in `docs/schema.md`). The analysis layer never imports audio code.
- `harmonia/theory/` — pure music theory: `pitch.py`, `chord.py` (parser + Chord model),
  `key.py`, `roman.py`. No analysis logic.
- `harmonia/analysis/`
  - `segments.py` — frames → merged segments with a soft `ChordDist` (top-k candidates).
  - `key_model.py` — chord-level HMM over 24 keys: hand-written emission table
    (MAJOR_FIT / MINOR_FIT) + V→I cadence bonus; Viterbi path; short regions merged
    (tonicization ≠ modulation); forward–backward posteriors = key confidence.
  - `context.py` — *runs* (consecutive segments sharing a root) + navigation that skips
    passing chords and short N.
  - `rules/` — one module per rule family, each a pure `detect(ctx) -> list[Event]`,
    registered in `rules/__init__.py`, enabled/weighted in config.
  - `resolve.py` — competing explanations of the same chromatic chord: resolved functional
    readings (rank 2) beat borrowed / unresolved readings (rank 1); losers kept in
    `alternatives`.
  - `pipeline.py` — `analyze()` and `build_context()` (the latter for testing rules alone).
- `harmonia/default_config.toml` — every threshold / weight / borrowed-chord table entry,
  with theory comments. Override with `--config`.

## Design decisions (and why)

1. **Root motion + resolution is the main evidence; quality is a bonus.** Generic quality
   factor `base + (1-base)·P(match) − penalty·P(contradict)`; uninformative quality (power
   chord, sus, unknown) → `base`. *Contradicting* quality is penalised, because root motion
   alone cannot separate ii–V–I from diatonic fifth chains (vi–ii–V has a minor "V") or
   II7–V7–I (a major "ii" is V/V).
2. **Secondary dominants require a tone outside the key** (major 3rd or ♭7). Otherwise every
   I→IV would be V/IV. The degree of "dominant-ness" (7th > triad > 7sus4) is soft.
3. **Borrowed chords**: when the root is chromatic (♭VI, ♭VII, ♭III, ♭II) the root is the
   main evidence; when the root is diatonic (iv, v, i, iiø) the borrowed tone *is* the
   quality, so quality is required.
4. **Confidence = product of named factors**, each recorded as `Evidence`. Key-dependent
   events (secondary, borrowed) are multiplied by `key_prob ** 0.5`.
5. **Soft input everywhere**: rules work on P(root), P(quality | root) from top-k candidates,
   so a misrecognised `F` for `Dm7` still yields a low-confidence ii–V–I.
6. **Nothing fails silently**: parse errors raise (text input) or become `X` segments with
   warnings; low-confidence segments/events are flagged with reasons; key ambiguity is a
   warning plus `global_key.ambiguous`.
7. **Global key** = key with the most posterior-weighted time (the home key of a
   modulating song); ambiguity = competing readings of the same passage.

## Notation conventions

- Roman numerals are relative to the **major scale of the tonic in both modes**, with
  accidentals (Berklee / 度数 style): A minor → i, iiø7, ♭III, iv, v, V7, ♭VI, ♭VII.
- Case = quality (lower = minor/dim). Suffixes: `m7` (vim7), `maj7`, `7`, `ø7`, `°7`, `mM7`,
  `+`, `6`, `sus4`. Chromatic diminished chords are raised degrees (♯i°7, ♯ivø7).
- Inversion: arabic bass scale degree after a slash (`I/3`, `V7/4`); applied chords use a
  roman target (`V7/vi`, `iim7/IV`, `vii°7/ii`, `subV7/V`).
- Note spelling follows the same degree logic: chromatic ♭2 ♭3 ♯4 ♭6 ♭7 (A♭ not G♯ in C).
- Machine chord labels are Harte (`chord_harte`) so mir_eval can consume them directly.
- Text input: `|` bars, `.` extends a chord one slot, `%` repeats a bar, `[Section]`
  markers, `//` comments (or `#` at line start — elsewhere `#` is a sharp).

## Working rules for this repo

- **Evaluation first**: from Phase 2 on, every change reports reproducible dev scores.
- `data/eval/dev` for development and calibration; `data/eval/test` only for final numbers.
  Never hard-code or tune for specific evaluation songs.
- **Never change a test's expected value** to make it pass unless the music-theory reason
  has been explained to and confirmed by the user.
- Config numbers are theory-motivated priors, not fitted. If tuned (dev only), record the
  change and before/after scores here.
- Mark theoretically uncertain choices with `[UNCERTAIN]` / `[ASK]` and ask the user;
  do not invent theory.
- Dependencies: ask before adding any. Before downloading large model weights, state size
  and source. Phase 3 library choice: present 2–3 options, user decides.
- Prefer MPS / Metal / Core ML on this Mac (no NVIDIA GPU).

## Open questions for the user (Phase 1)

1. **Key reference for 丸サ進行** — with no key given, `Fmaj7 E7 Am7 Gm7 C7` is estimated as
   F major (= D♭ major in the original song: I–VII7–iii–ii–V7). The user's numerals
   (IVmaj7–III7–vim7–v–I7) use the C (= original A♭) reference. Tests currently fix the key
   for this case. Which reference should automatic analysis prefer?
2. **Relative-key ambiguity of bare loops** (IV–V–iii–vi with no tonic chord): currently a
   mild major prior (`key.minor_bias = -0.1`) + an explicit ambiguity flag. OK?
3. **Contradicting quality is penalised** (design decision 1) — a deliberate softening of
   "quality only as bonus". OK?
4. Unresolved ii–V (no I) is not reported yet; deceptive cadences (V→vi) are not events.
   Wanted?
5. `[[borrowed.major]] ♭VI7` entry is marked `[UNCERTAIN]`.

## Eval material

Local paths of user-provided audio / scores are in `CLAUDE.local.md` (gitignored — this repo
is public). Copyrighted audio and annotations NEVER get committed: `data/eval/**` is
gitignored except README. External raw inputs are never modified; reference them from
`data/eval/<split>/<song>/meta.toml` with provenance. The user allows cross-validation with
openly downloadable scores / audio (state source and size before downloading).
