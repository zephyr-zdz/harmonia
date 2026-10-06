# Phase 3 — audio front end: options (researched 2026-10-06)

Target: audio → RecognitionResult (per-beat top-k chord candidates + probabilities, bass,
beat/bar grid), on macOS without NVIDIA (CPU / MPS), Python 3.14 (current .venv).

## Component survey

| component | candidate | status (2026-10) | Py3.14 | notes |
|---|---|---|---|---|
| chords | **lv-chordia** (ISMIR 2019, Jiang et al.) | maintained wrapper, pushed 2026-10-05, MIT | yes (torch ≥ 2.13) | 5-model ensemble, ~28 MB weights bundled in the repo; *chord-structure decomposition* heads (root / bass / triad / 7th / 9th / 11th / 13th) → frame posteriors we can tap for top-k + bass. Paper: Billboard ~81 %, RWC-Pop ~78 %, Beatles ~83 % (submission vocab). MPS refused by the wrapper → CPU. |
| chords | BTC (ISMIR 2019) | original repo stale (2020), MIT; HF repackaging exists | yes | 170-chord vocab; HF wrapper hides logits (need own forward pass). Trained on full mixes. |
| chords | ChordFormer (2025) | research code | ? | best reported large-vocab numbers; would need (re)training |
| chords | chroma + templates + HMM (librosa) | — | yes | no weights; weak but a useful *baseline* and fallback |
| beats | **Beat This!** (ISMIR 2024, CPJKU) | `beat-this` 1.1.0 on PyPI (2026-04), MIT | yes | beats + downbeats, no madmom; torch |
| beats | madmom | last PyPI release 2018, no wheels | **no** | excluded |
| beats | librosa beat_track | — | yes | no downbeats; fallback only |
| separation | **Demucs htdemucs** (`demucs` 4.1.0, adefossez fork) | original repo archived 2025-01-01; fork maintained (slow) | yes | ~84 MB weights; we only need the bass stem |
| separation | BS-/Mel-RoFormer via `audio-separator` 0.47 | active, MIT | yes (≠3.14.1) | best SDR; weights several hundred MB; CoreML accel |
| bass | **pYIN (librosa) on the bass stem** | — | yes | monophonic bass is the easy case for pYIN; beat-synchronous voting |
| bass | basic-pitch | last release 2024; CoreML/TF deps lack 3.14 | risky | excluded for now |

Important: the chord models were trained on **full mixes**, so they get the full mix. Source
separation is used only for bass transcription (and as an optional cross-check).

## Options

**A. Pretrained, light, CPU-friendly (recommended).**
lv-chordia (frame posteriors tapped → beat-synchronous top-k) + Beat This! (beats/downbeats)
+ htdemucs bass stem → pYIN bass + our Viterbi with a harmonic-transition prior.
Weights ≈ 28 MB + ~80 MB (Beat This!) + 84 MB (htdemucs). Runs on CPU; no training.
Why: best published pop accuracy among ready-to-run models, MIT everywhere, bass comes
for free from the decomposition heads and is cross-checked by pYIN.

**B. BTC + RoFormer.** BTC frame logits (own forward pass) + Beat This! + BS-RoFormer bass
stem + pYIN. Better separation, simpler chord vocabulary, larger downloads (~hundreds MB),
BTC repo unmaintained.

**C. Train / distill our own (6000 Pro Blackwell).** Conformer (ChordFormer-style) student
distilled from A/B teachers on the user's own library (pseudo-labels, à la arXiv 2602.19778)
+ public annotations whose audio the user owns. Only worth it if evaluation shows
recognition errors dominate after A. IdolSongsJp's license forbids training on it — eval only.

All options share the same interface (`harmonia/frontend/`): each component is a backend
behind a small protocol, so switching A↔B is a config change. A chroma-template backend is
kept as baseline.

Decision status: proceeding with **A** under the session goal; user may override.
