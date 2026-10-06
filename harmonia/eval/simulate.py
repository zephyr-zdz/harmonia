"""Simulated recogniser: corrupts reference chords in musically typical ways.

Purpose: exercise the evaluation framework and measure how robust the ANALYSIS layer is to
recognition errors before the audio front end exists. It is NOT a model of any real system;
results obtained with it are labelled "simulated" in every report.

Error types (each applied per reference chord with its own probability):
  root     — root replaced by a typical confusion: a fifth up/down, or the relative
             major/minor (C ↔ Am shares two tones). Quality is kept (relative swaps flip it).
  quality  — seventh dropped, or third flipped (maj ↔ min).
  boundary — chord boundaries shifted by N(0, jitter) seconds (order preserved).
The true chord is kept as the 2nd candidate when the top-1 is corrupted, so soft-evidence
rules can recover; top-1 probability is drawn lower for corrupted chords.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from ..schema import ChordCandidate, Frame, RecognitionResult
from ..theory.chord import Chord, ChordParseError, parse_chord

_DROP_SEVENTH = {"7": "maj", "maj7": "maj", "min7": "min", "hdim7": "dim", "dim7": "dim", "minmaj7": "min",
                 "7sus4": "sus4", "aug7": "aug", "maj6": "maj", "min6": "min"}
_FLIP_THIRD = {"maj": "min", "min": "maj", "7": "min7", "min7": "7", "maj7": "min7", "maj6": "min6",
               "min6": "maj6"}


@dataclass(frozen=True)
class SimConfig:
    root_error: float = 0.1
    quality_error: float = 0.15
    boundary_jitter: float = 0.1
    seed: int = 0


def _with(ch: Chord, root: int | None = None, quality: str | None = None) -> Chord:
    return Chord(root if root is not None else ch.root, quality or ch.quality, (), None)


def _corrupt_root(ch: Chord, rng: random.Random) -> Chord:
    kind = rng.choice(["fifth_up", "fifth_down", "relative"])
    if kind == "fifth_up":
        return _with(ch, (ch.root + 7) % 12)
    if kind == "fifth_down":
        return _with(ch, (ch.root + 5) % 12)
    if ch.qclass in ("min",):
        return _with(ch, (ch.root + 3) % 12, "maj")
    return _with(ch, (ch.root + 9) % 12, "min")


def _corrupt_quality(ch: Chord, rng: random.Random) -> Chord:
    options = []
    if ch.quality in _DROP_SEVENTH:
        options.append(_DROP_SEVENTH[ch.quality])
    if ch.quality in _FLIP_THIRD:
        options.append(_FLIP_THIRD[ch.quality])
    if not options:
        return ch
    return _with(ch, quality=rng.choice(options))


def simulate(intervals: list[tuple[float, float]], labels: list[str], cfg: SimConfig,
             time_unit: str = "second") -> RecognitionResult:
    rng = random.Random(cfg.seed)
    n = len(intervals)
    bounds = [intervals[0][0]] + [intervals[i][1] for i in range(n)]
    for i in range(1, n):  # jitter interior boundaries, keep order
        lo, hi = bounds[i - 1] + 1e-3, intervals[i][1] - 1e-3
        bounds[i] = min(max(bounds[i] + rng.gauss(0, cfg.boundary_jitter), lo), max(lo, hi))
    frames = []
    stats = {"root_errors": 0, "quality_errors": 0, "chords": 0}
    for i, lab in enumerate(labels):
        try:
            true = parse_chord(lab)
        except ChordParseError:
            true = None
        start, end = bounds[i], bounds[i + 1]
        if true is None or true.root is None:
            cands = [ChordCandidate(lab, 1.0)]
        else:
            stats["chords"] += 1
            est = _with(true)  # strip extensions / bass: recognisers rarely output them
            if rng.random() < cfg.root_error:
                est = _corrupt_root(est, rng)
                stats["root_errors"] += 1
            if rng.random() < cfg.quality_error:
                est = _corrupt_quality(est, rng)
                stats["quality_errors"] += 1
            if est.harte() == _with(true).harte():
                p1 = rng.uniform(0.6, 1.0)
                cands = [ChordCandidate(est.harte(), round(p1, 3))]
                if p1 < 0.95:
                    cands.append(ChordCandidate(_corrupt_quality(est, rng).harte(), round((1 - p1) * 0.7, 3)))
            else:
                p1 = rng.uniform(0.35, 0.8)
                cands = [ChordCandidate(est.harte(), round(p1, 3)),
                         ChordCandidate(_with(true).harte(), round((1 - p1) * 0.8, 3))]
        frames.append(Frame(time=start, duration=end - start, candidates=cands))
    return RecognitionResult(frames=frames, time_unit=time_unit, beats_per_bar=None,
                             source={"type": "simulated", "config": asdict(cfg), "stats": stats})
