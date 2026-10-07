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
from .beats import track_beats
from .meter import build_grid
from .chords import frame_evidence
from .decode import decode
from .structure import from_recognition_inputs as analyse_structure, pool_posteriors

AUDIO_SUFFIXES = {".mp3", ".m4a", ".flac", ".wav", ".aiff", ".aif", ".ogg", ".opus"}


def load_prior(cfg: dict) -> dict | None:
    p = cfg["frontend"]["decode"].get("transition_prior")
    if not p:
        return None
    path = Path(p)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return json.loads(path.read_text(encoding="utf-8"))


def readable_audio(path: Path) -> Path:
    """libsndfile (used by Beat This!) cannot open some containers (m4a/aac): decode those once
    to a cached 44.1 kHz wav with ffmpeg. The original file is never modified."""
    import soundfile as sf
    try:
        sf.info(str(path))
        return path
    except Exception:
        pass
    import hashlib
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(f"cannot decode {path.name}: libsndfile cannot read it and ffmpeg is not installed")
    st = path.stat()
    h = hashlib.sha256(f"{path}|{st.st_size}|{st.st_mtime}".encode()).hexdigest()[:12]
    out = Path(__file__).resolve().parents[2] / "outputs" / "cache" / "decoded" / f"{path.stem}.{h}.wav"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(path), "-ac", "2", "-ar", "44100",
                        str(out)], check=True)
    return out


def transcribe(audio_path: str | Path, cfg: dict | None = None) -> RecognitionResult:
    cfg = cfg or load_config()
    return from_features(*extract_features(audio_path, cfg), cfg)


def extract_features(audio_path: str | Path, cfg: dict) -> tuple:
    """The slow, network part (beats + frame chord evidence). Its output does not depend on
    [frontend.decode], so calibration caches it and re-runs only ``from_features``."""
    source_path = Path(audio_path).resolve()  # lv-chordia resolves relative paths against its package dir
    audio_path = readable_audio(source_path)
    return frame_evidence(audio_path, cfg), track_beats(audio_path, cfg), source_path, audio_path


def from_features(ev, grid, source_path: Path, audio_path: Path, cfg: dict) -> RecognitionResult:
    """Meter grid + beat-synchronous decoding + top-k candidates (fast)."""
    fc = cfg["frontend"]
    dc = fc["decode"]
    warnings: list[str] = []
    warnings += grid.warnings
    meter = None
    if len(grid.beats) >= 8:
        meter = build_grid(grid.beats, grid.downbeats, ev.duration)
        warnings += meter.warnings
        bt = meter.beats[(meter.beats >= 0) & (meter.beats < ev.duration - 1e-3)]
        keep = (meter.beats >= 0) & (meter.beats < ev.duration - 1e-3)
        bars = [b for b, k in zip(meter.bar_index, keep) if k]
        pos = [p for p, k in zip(meter.beat_in_bar, keep) if k]
        edges = np.concatenate([[0.0], bt, [ev.duration]]) if bt[0] > 1e-3 else np.concatenate([bt, [ev.duration]])
        if bt[0] > 1e-3:  # audio before the first beat: one pre-roll frame without bar info
            bars, pos = [None] + bars, [None] + pos
        period = meter.beat_period
    else:
        warnings.append("fewer than 8 beats detected; falling back to 0.5 s frames, no bar lines")
        edges = np.arange(0.0, ev.duration + 0.5, 0.5)
        edges[-1] = ev.duration
        bars, pos = [None] * (len(edges) - 1), [None] * (len(edges) - 1)
        period = 0.5

    dec = decode(ev, edges, dc, load_prior(cfg), positions=pos, beats_per_bar=meter.beats_per_bar if meter else None)
    structure = None
    if cfg.get("structure", {}).get("enabled") and meter is not None:
        try:
            structure = analyse_structure(audio_path, edges, bars, meter.beats_per_bar, dec.posteriors, dec.labels, cfg,
                                          source_path=source_path)
            warnings += [f"structure: {w}" for w in structure.warnings]
            dec.posteriors = pool_posteriors(dec.posteriors, structure, cfg)
        except Exception as e:  # structure is optional: report, never fail the transcription
            warnings.append(f"structure analysis failed: {type(e).__name__}: {e}")
            structure = None
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
                            bass=bass, bar=bars[b], beat_in_bar=pos[b],
                            section=structure.label_at(float(edges[b])) if structure else None))
    if n_low:
        warnings.append(f"{n_low}/{len(frames)} beats have top chord probability < "
                        f"{cfg['general']['low_confidence_threshold']}")
    beats_per_bar = meter.beats_per_bar if meter else None
    return RecognitionResult(
        frames=frames, time_unit="second", beats_per_bar=beats_per_bar,
        source={"type": "audio", "path": str(source_path), "decoded_from": None if audio_path == source_path else str(audio_path),
                "chord_backend": ev.backend,
                "beat_backend": grid.backend, "tempo_bpm": round(60.0 / period, 1),
                "meter": None if meter is None else {
                    "bpm": round(meter.bpm, 1), "beats_per_bar": meter.beats_per_bar,
                    "time_signature": meter.time_signature, "bar_seconds": round(meter.bar_period, 4),
                    "tempo_cv": round(meter.tempo_cv, 4), "n_bars": int(len(meter.bar_starts)),
                    "irregular_bars": meter.irregular_bars},
                "decode": {k: v for k, v in dc.items()},
                "structure": structure.to_dict() if structure else None},
        warnings=warnings,
    )


def is_audio(path: str | Path) -> bool:
    return Path(path).suffix.lower() in AUDIO_SUFFIXES
