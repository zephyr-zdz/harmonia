# Harmonia — 和声分析器

Local tool: audio (mp3/m4a/flac) → time-aligned chords, local keys, roman numerals, and
annotated "notable harmonic events" (ii–V–I, secondary dominants, borrowed chords, tritone
subs, modulations), every judgement with a confidence and its evidence.
Target styles: J-pop / City Pop / pop. Jazz reharmonisation accuracy is a non-goal.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Symbolic analysis layer (chord symbols → keys / numerals / events) | done (accepted 2026-10-06) |
| 2 | Evaluation framework (mir_eval chord metrics, event P/R, error attribution) | framework done; IdolSongsJp (15 songs, 8 dev / 7 test) evaluated; MyGO refs pending (user transcribing official scores) |
| 3 | Audio front end | option A running end-to-end (docs/phase3_options.md); decoder calibrated on IdolSongsJp dev (`harmonia.frontend.calibrate`) |
| 4 | UI (timeline, playback, manual chord edits → re-analysis) | running (song library + switching); awaiting user feedback |
| 5 | Optional LLM narration of events | not started |

Each phase ends with: run tests, report metrics, stop for user confirmation.

## Commands

```bash
uv sync --extra eval                                 # .venv with mir_eval (Phase 2)
.venv/bin/python -m unittest discover -s tests -t .  # full suite (eval tests skip without mir_eval)
.venv/bin/python -m harmonia eval --split dev --estimates outputs/<run>   # real system output
.venv/bin/python -m harmonia.eval.idolsongsjp                          # (re)build data/eval from IdolSongsJp
.venv/bin/python -m harmonia.frontend.calibrate --out outputs/calibration/<name>   # decoder grid on dev
bash tools/songformer_gpu/make_bundle.sh                                     # SongFormer bundle for the GPU host
.venv/bin/python tools/compare_structure.py outputs/songformer/runs/gpu_v1     # structure A vs SongFormer
.venv/bin/python -m harmonia eval --split dev --simulate --seed 0         # simulated recogniser
.venv/bin/python -m harmonia eval --split dev --sweep                     # robustness curve
uv sync --extra eval --extra audio                                       # + torch, lv-chordia, beat-this, demucs
.venv/bin/python -m harmonia analyze song.flac --save-recognition rec.json   # audio end to end
.venv/bin/python -m harmonia transcribe song.mp3 -o rec.json                 # recognition layer only
.venv/bin/python -m harmonia batch data/external/user_music -o outputs/runs/<name>   # whole folder
.venv/bin/python -m harmonia ui                                              # song library UI (127.0.0.1:8765)
.venv/bin/python -m harmonia report outputs/runs/<run>/*.recognition.json -o report.md   # per-section summary
python3 -m harmonia analyze "| Fmaj7 | E7 | Am7 | Gm7 C7 |" --key "C major"
python3 -m harmonia analyze song.lab --json -o out.json
```
The analysis layer is stdlib-only; Python ≥ 3.11 (`tomllib`). `.venv` = Python 3.14 + optional `eval` extra
(mir_eval 0.8.2, numpy, scipy), locked in `uv.lock`. Reports go to `outputs/eval/` (gitignored).

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
  - `idioms.py` — named progressions (王道 / 丸サ / 小室 / カノン), matched in all
    transpositions before key estimation; optional reference-key prior.
  - `context.py` — *runs* (consecutive segments sharing a root) + navigation that skips
    passing chords and short N.
  - `rules/` — one module per rule family, each a pure `detect(ctx) -> list[Event]`,
    registered in `rules/__init__.py`, enabled/weighted in config.
  - `resolve.py` — competing explanations of the same chromatic chord: resolved functional
    readings (rank 2) beat borrowed / unresolved readings (rank 1); losers kept in
    `alternatives`.
  - `pipeline.py` — `analyze()` and `build_context()` (the latter for testing rules alone).
- `harmonia/eval/` (Phase 2) — `dataset.py` (data/eval layout, gold events by time or bars),
  `align.py` (edit-distance chart alignment), `chords.py` (mir_eval root/majmin/sevenths; chart
  symbol-error rates + approximate aligned scores), `events.py` (event P/R/F1, error
  attribution oracle-vs-system), `simulate.py` (seeded simulated recogniser), `runner.py`
  (per-song + aggregate report with provenance: git commit, config hash, seed, versions).
  Attribution logic: analysis run on reference chords = "oracle" (analysis-layer errors);
  on recognised chords = "system"; the difference is recognition-induced.
- `harmonia/frontend/` (Phase 3, option A) — `beats.py` (Beat This!, librosa fallback),
  `chords.py` (lv-chordia ensemble; we tap its decomposition heads + observation function,
  not its final labels), `decode.py` (beat-synchronous HMM with root-motion transition prior,
  forward–backward → per-beat top-k; chord changes cost extra off the bar line / half bar),
  `calibrate.py` (dev-only grid over [frontend.decode] with cached network features),
  `structure.py` (sections: bar-level chord-identity repetition, novelty + phrase-prior DP,
  clustering with one transposition per section pair, J-pop labelling rules; frames get
  `section`, RecognitionResult.source["structure"] lists the sections),
  `meter.py` (user suggestion: BPM first, then bar lines — mode-seeded beat & bar periods,
  meter m = bar/beat, BPM = 60·m/bar, regularised bar grid with filled / dropped / genuinely
  irregular bars), `pipeline.py` (`transcribe()` → RecognitionResult; m4a etc. decoded once
  via ffmpeg into outputs/cache/decoded/).
- `harmonia/report.py` — `harmonia report`: per section bar numerals, shortest loop, keys, named
  progressions, confident events, low-confidence share (post-processing of the two JSON layers).
- `harmonia/batch.py` — `harmonia batch <dir> -o outputs/runs/<name>` (per-song JSON + summary.md).
- `harmonia/ui/` — `harmonia ui [dirs/files…]` (default `[ui].library`): local timeline UI with a song
  library (select / ◀ ▶ / `[` `]`), background transcription of new audio, cached recognitions
  from `outputs/runs/*`, per-song edits in localStorage, export to `outputs/annotations/`.
  Chord models get the FULL MIX (they were trained on mixes); separation is optional, bass only.
- Model weights live in `models/` (gitignored): `models/beat_this/final0.ckpt` (81 MB, MIT,
  https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt, sha256 8c328b45…).
  lv-chordia ships its 28 MB ensemble inside the package.
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

## Decisions log

- **2026-10-06 (user)** — Phase 1 accepted with:
  1. 丸サ進行 is analysed in its naming reference (IVmaj7–III7–vim7–vm7–I7) even without a
     given key. Implemented generically: named progressions (`[[named_progressions.idiom]]`)
     may set `use_key_prior = true`, adding a reference-key prior to the key HMM. Only 丸サ
     has it.
  2. Bare relative-ambiguous loops (王道進行): keep the mild major prior + ambiguity flag.
  3. Contradicting chord quality is penalised (design decision 1) — accepted.
  4. Added: unresolved ii–V (`ii_V`: deceptive / unresolved / end) and deceptive cadence
     (`deceptive_cadence`: prepared V or V7 → vi / ♭VI). The iv–♭VII7–I backdoor is *not* a
     failed ii–V of ♭III; it stays a borrowed ♭VII7 with `backdoor`.
  5. `♭VI7` borrowed-table entry kept.
- Repo is public on GitHub (zephyr-zdz/harmonia). No license chosen yet (user's call).
- **2026-10-06 (user)**: download IdolSongsJp + ChoCo; user annotates MyGO songs (online
  resources allowed); start Phase 3 selection. Phase 3 proceeds with option A under the
  session goal; every component is a swappable backend (user may override).
- **IdolSongsJp license**: non-commercial research; using its tracks for model TRAINING is
  prohibited → evaluation only. Gated on HuggingFace: the user must accept terms and log in.
- **2026-10-06 (user)**: estimate BPM first, then bar lines → `frontend/meter.py`. Check case:
  春日影 tab = 6/8, ♪=194 → system 193.5 BPM, 6/8, bar 1.860 s (tab 1.856 s).
- **2026-10-06 (user)**: copy the provided music into the project and recognise all of it →
  `data/external/user_music/` (gitignored, PROVENANCE.md), results in `outputs/runs/user_music_v1/`.
  User's impression of accuracy: "还行" (decent). No reference annotations yet → no scores.
- **2026-10-07 (user)**: (1) boundary tonic bonus ON (3.0; chords inside / restarting a named
  progression exempt, so 丸サ / 王道 decisions still hold); (2) too many deceptive cadences →
  graded preparation (IV–V–vi triads 0.5, mostly low-confidence now); (3) too many secondary
  dominants → unresolved ii–V of a non-tonic target needs chromatic V + minor ii, unresolved V/x
  of a non-diatonic target dropped. Scores in docs/results.md. Postponed: 影色舞 modulation.
  MyGO references: user will transcribe official printed scores.
- **2026-10-07 (user)**: 春日影 (MyGO ver.) is in **B major** (the system's reading), not B
  Mixolydian / E major despite the tab's key signature.
- IdolSongsJp (2026-10-07): access granted; data/external/idolsongsjp (gitignored) holds only the
  15 plain mixes (master_48k32b_-9LUFS, 1.4 GB) + chords/ + keys/. Imported with
  `python -m harmonia.eval.idolsongsjp` → data/eval/{dev,test}/<id>/ (chords.lab, keys.lab =
  time-varying keys, meta.toml with provenance + license). Split by choco.split_of: 8 dev / 7 test.
  NOTE: `hf download --include a b c --exclude …` ignored --exclude and fetched every variant
  (7 GB); use hf_hub_download per file.
- **2026-10-07 (user)**: V7 leading tone as minor-key evidence → implemented
  (`key.minor_leading_tone_bonus`) but every dev set got worse (III7→vim of the relative major is
  the same chord pair) → kept at 0. Decoder calibrated on dev: obs_weight 1.5, change_penalty 1.0,
  change_extra_offbeat 0.5, new seventh_bias 1.0 (dev majmin .812→.826, sevenths .623→.664).
  Open: relative major/minor confusions on test (m01, m05, m06).
- **2026-10-07 (user)**: recognise カノン進行 → two entries (Pachelbel core I–V–vi–iii–IV with
  J-pop second halves; descending bass 1–7–6–5–(4|♯4)); idiom matcher gained alternatives, bass
  constraints, `min_length`, geometric-mean confidence. [UNCERTAIN] whether the ♯4-bass variant
  (II/♯4, ♯ivø7) is still called カノン進行.
- **2026-10-07 (user)**: section detection → option A (own, `frontend/structure.py`, no new deps);
  B (all-in-one) / C (SongFormer) feasibility in docs/structure_options.md, nothing installed.
  Chord pooling across repeats measured useless on dev → off. Labels unscored until the user's
  score transcriptions carry `[Verse]` / `[Chorus]` markers.
- **2026-10-07 (user)**: SongFormer (C) ran on bw2-01 via the `gsched` skill (29 songs, 2 min); its
  labels follow J-pop form much better than A's rules → combined: SongFormer boundaries + labels
  (`[structure].external_dirs`), A's repetition groups + transposition shift. A alone is the fallback.
- ChoCo v1.0.0 (Zenodo 7706751, md5 c26f2380…): annotations only; used for symbolic
  validation of the key model and for transition-prior statistics.

## Results

All measured numbers live in `docs/results.md` (split, data, config, git state).
`harmonia.eval.choco` = symbolic validation of the analysis layer on ChoCo expert labels.

## Eval material

Local paths of user-provided audio / scores are in `CLAUDE.local.md` (gitignored — this repo
is public). Copyrighted audio and annotations NEVER get committed: `data/eval/**` is
gitignored except README. External raw inputs are never modified; reference them from
`data/eval/<split>/<song>/meta.toml` with provenance. The user allows cross-validation with
openly downloadable scores / audio (state source and size before downloading).
