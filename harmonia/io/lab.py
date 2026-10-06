"""MIREX-style .lab chord annotations: ``start end label`` per line (seconds, Harte labels)."""

from __future__ import annotations

from pathlib import Path

from ..schema import ChordCandidate, Frame, RecognitionResult


def read_lab(path: str | Path) -> list[tuple[float, float, str]]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            raise ValueError(f"{path}:{n}: expected 'start end label', got {line!r}")
        rows.append((float(parts[0]), float(parts[1]), parts[2].strip()))
    return rows


def lab_to_recognition(path: str | Path) -> RecognitionResult:
    frames = [Frame(time=s, duration=e - s, candidates=[ChordCandidate(lbl, 1.0)])
              for s, e, lbl in read_lab(path)]
    return RecognitionResult(frames=frames, time_unit="second", beats_per_bar=None,
                             source={"type": "lab", "path": str(path)})
