"""Beat / downbeat tracking backends.

Each backend returns (beats, downbeats) in seconds. The "beat_this" backend (ISMIR 2024,
CPJKU, MIT) is the default; "librosa" is a no-download fallback without downbeats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BeatGrid:
    beats: np.ndarray       # seconds, increasing
    downbeats: np.ndarray   # seconds, subset of beats (may be empty)
    backend: str
    warnings: list[str]


def track_beats(audio_path: str | Path, cfg: dict) -> BeatGrid:
    fc = cfg["frontend"]["beats"]
    backend = fc["backend"]
    if backend == "beat_this":
        return _beat_this(audio_path, fc)
    if backend == "librosa":
        return _librosa(audio_path)
    raise ValueError(f"unknown beat backend {backend!r}")


def _beat_this(audio_path: str | Path, fc: dict) -> BeatGrid:
    from beat_this.inference import File2Beats

    ckpt = Path(fc["checkpoint"])
    if not ckpt.is_absolute():
        ckpt = Path(__file__).resolve().parents[2] / ckpt
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Beat This! checkpoint not found at {ckpt}. Download final0.ckpt (81 MB, MIT) from "
            "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt (see CLAUDE.md).")
    f2b = File2Beats(checkpoint_path=str(ckpt), device=fc.get("device", "cpu"), dbn=False)
    beats, downbeats = f2b(str(audio_path))
    warnings = []
    if len(downbeats) == 0:
        warnings.append("beat tracker found no downbeats; bar numbers unavailable")
    return BeatGrid(np.asarray(beats, float), np.asarray(downbeats, float), "beat_this", warnings)


def _librosa(audio_path: str | Path) -> BeatGrid:
    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    _, frames = librosa.beat.beat_track(y=y, sr=sr)
    beats = librosa.frames_to_time(frames, sr=sr)
    return BeatGrid(np.asarray(beats, float), np.array([]), "librosa",
                    ["librosa beat tracker: no downbeats; bar numbers unavailable"])


def beat_intervals(grid: BeatGrid, duration: float) -> tuple[np.ndarray, list[int | None], list[int | None]]:
    """Beat-synchronous intervals covering [0, duration]: returns (edges, bar, beat_in_bar).

    Audio before the first beat / after the last becomes one extra interval each (marked
    bar=None). Bars are counted from the first downbeat (pickup beats get bar 0).
    """
    b = grid.beats[(grid.beats > 0) & (grid.beats < duration)]
    edges = np.concatenate([[0.0], b, [duration]])
    edges = edges[np.concatenate([[True], np.diff(edges) > 1e-3])]
    if edges[-1] < duration:
        edges = np.append(edges, duration)
    db = grid.downbeats
    bars: list[int | None] = []
    pos: list[int | None] = []
    bar, k = 0, 0
    for i in range(len(edges) - 1):
        t = edges[i]
        is_beat = any(abs(t - x) < 1e-3 for x in b)
        if not is_beat:
            bars.append(None)
            pos.append(None)
            continue
        if len(db) and np.min(np.abs(db - t)) < 0.03:
            bar += 1
            k = 1
        else:
            k += 1
        bars.append(bar if len(db) else None)
        pos.append(k if len(db) else None)
    return edges, bars, pos
