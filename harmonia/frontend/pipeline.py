"""Audio -> RecognitionResult (the recognition layer's only output).

beats (Beat This!) -> frame chord evidence (lv-chordia) -> beat-synchronous HMM with a
harmonic transition prior -> per-beat top-k candidates + bass -> RecognitionResult JSON.
Local key is NOT estimated here: the analysis layer owns key estimation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import load_config
from ..schema import BassObservation, ChordCandidate, Frame, RecognitionResult
from ..theory.pitch import note_name
from .beats import beat_intervals, track_beats
from .chords import frame_evidence
from .decode import decode

AUDIO_SUFFIXES = {".mp3", ".m4a", ".flac", ".wav", ".aiff", ".aif", ".ogg", ".opus"}


def load_prior(cfg: dict) -> dict | None:
    p = cfg["frontend"]["decode"].get("transition_prior")
    if not p:
        return None
    path = Path(p)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return json.loads(path.read_text(encoding="utf-8"))


def transcribe(audio_path: str | Path, cfg: dict | None = None) -> RecognitionResult:
    cfg = cfg or load_config()
    fc = cfg["frontend"]
    dc = fc["decode"]
    warnings: list[str] = []

    ev = frame_evidence(audio_path, cfg)
    grid = track_beats(audio_path, cfg)
    warnings += grid.warnings
    edges, bars, pos = beat_intervals(grid, ev.duration)
    if len(edges) < 3:
        warnings.append("fewer than 2 beats detected; falling back to 0.5 s frames")
        edges = np.arange(0.0, ev.duration + 0.5, 0.5)
        edges[-1] = ev.duration
        bars, pos = [None] * (len(edges) - 1), [None] * (len(edges) - 1)

    dec = decode(ev, edges, dc, load_prior(cfg))
    period = float(np.median(np.diff(grid.beats))) if len(grid.beats) > 1 else 0.5
    frames: list[Frame] = []
    n_low = 0
    for b in range(len(edges) - 1):
        p = dec.posteriors[b]
        order = np.argsort(-p)[: fc["top_k"]]
        cands = [ChordCandidate(dec.labels[i], round(float(p[i]), 4)) for i in order if p[i] >= fc["min_candidate_prob"]]
        if not cands:
            cands = [ChordCandidate(dec.labels[order[0]], round(float(p[order[0]]), 4))]
        if cands[0].prob < cfg["general"]["low_confidence_threshold"]:
            n_low += 1
        bass = None
        if dec.bass is not None:
            bp = dec.bass[b]
            k = int(np.argmax(bp))
            bass = BassObservation(None if k == 0 else note_name(k - 1), round(float(bp[k]), 4))
        dur = float(edges[b + 1] - edges[b])
        frames.append(Frame(time=float(edges[b]), duration=dur, candidates=cands,
                            beats=float(min(max(dur / period, 0.25), 4.0)),
                            bass=bass, bar=bars[b], beat_in_bar=pos[b]))
    if n_low:
        warnings.append(f"{n_low}/{len(frames)} beats have top chord probability < "
                        f"{cfg['general']['low_confidence_threshold']}")
    # most common bar length = last beat_in_bar before a downbeat
    lengths = [pos[i] for i in range(len(pos) - 1) if pos[i] is not None and pos[i + 1] == 1]
    beats_per_bar = int(np.bincount(lengths).argmax()) if lengths else None
    return RecognitionResult(
        frames=frames, time_unit="second", beats_per_bar=beats_per_bar,
        source={"type": "audio", "path": str(audio_path), "chord_backend": ev.backend,
                "beat_backend": grid.backend, "tempo_bpm": round(60.0 / period, 1),
                "decode": {k: v for k, v in dc.items()}},
        warnings=warnings,
    )


def is_audio(path: str | Path) -> bool:
    return Path(path).suffix.lower() in AUDIO_SUFFIXES
