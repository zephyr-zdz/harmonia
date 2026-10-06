"""Evaluation-set discovery and reference annotations.

Layout (see data/eval/README.md):
    <root>/<split>/<song_id>/
        chords.lab   time-aligned reference (seconds, Harte or pop labels)      — or —
        chords.txt   bar-wise chord chart (harmonia.io.progression syntax)
        events.json  optional gold harmonic events
        meta.toml    optional: title, artist, key, audio, source, license, beats_per_bar

Gold events (events.json, a list):
    {"type": "secondary_dominant", "start": 12.3, "end": 14.0, "label": "V7/vi"}   # lab refs
    {"type": "ii_V_I", "bars": [17, 18], "label": "ii–V–I"}                         # chart refs
``bars`` is an inclusive 1-based bar range; ``label`` is optional (label accuracy is then
not scored for that event).
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..io.lab import read_lab
from ..io.progression import parse_progression
from ..schema import RecognitionResult

SPLITS = ("dev", "test")


@dataclass
class GoldEvent:
    type: str
    start: float          # in the reference timeline (seconds for lab, beats for chart)
    end: float
    label: str | None = None


@dataclass
class Reference:
    song_id: str
    split: str
    kind: str                                   # "lab" | "chart"
    recognition: RecognitionResult              # reference chords as an (oracle) input
    events: list[GoldEvent] | None              # None = no gold events
    meta: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @property
    def time_unit(self) -> str:
        return self.recognition.time_unit

    def intervals_labels(self) -> tuple[list[tuple[float, float]], list[str]]:
        fr = self.recognition.frames
        return [(f.time, f.time + f.duration) for f in fr], [f.candidates[0].label for f in fr]


def _load_events(path: Path, kind: str, beats_per_bar: int) -> list[GoldEvent]:
    out = []
    for n, e in enumerate(json.loads(path.read_text(encoding="utf-8"))):
        if "bars" in e:
            b0, b1 = e["bars"]
            start, end = (b0 - 1) * beats_per_bar, b1 * beats_per_bar
            if kind == "lab":
                raise ValueError(f"{path}[{n}]: 'bars' needs a chart reference (no bar grid for .lab)")
        elif "start" in e and "end" in e:
            start, end = float(e["start"]), float(e["end"])
        else:
            raise ValueError(f"{path}[{n}]: event needs 'start'/'end' or 'bars'")
        out.append(GoldEvent(e["type"], start, end, e.get("label")))
    return out


def load_reference(song_dir: Path, split: str) -> Reference:
    meta: dict[str, Any] = {}
    if (song_dir / "meta.toml").is_file():
        meta = tomllib.loads((song_dir / "meta.toml").read_text(encoding="utf-8"))
    bpb = int(meta.get("beats_per_bar", 4))
    if (song_dir / "chords.lab").is_file():
        kind = "lab"
        rows = read_lab(song_dir / "chords.lab")
        from ..schema import ChordCandidate, Frame
        rec = RecognitionResult(frames=[Frame(s, e - s, [ChordCandidate(l, 1.0)]) for s, e, l in rows],
                                time_unit="second", beats_per_bar=None,
                                source={"type": "reference", "path": str(song_dir / "chords.lab")})
    elif (song_dir / "chords.txt").is_file():
        kind = "chart"
        rec = parse_progression((song_dir / "chords.txt").read_text(encoding="utf-8"), beats_per_bar=bpb)
        rec.source = {"type": "reference", "path": str(song_dir / "chords.txt")}
    else:
        raise FileNotFoundError(f"{song_dir}: needs chords.lab or chords.txt")
    events = None
    if (song_dir / "events.json").is_file():
        events = _load_events(song_dir / "events.json", kind, bpb)
    return Reference(song_dir.name, split, kind, rec, events, meta, song_dir)


def discover(root: Path, split: str) -> list[Reference]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    d = root / split
    if not d.is_dir():
        return []
    return [load_reference(p, split) for p in sorted(d.iterdir())
            if p.is_dir() and not p.name.startswith((".", "_"))]


def load_estimate(path: Path) -> RecognitionResult:
    """A system output: RecognitionResult JSON, or a .lab (e.g. from another recogniser)."""
    if path.suffix == ".lab":
        from ..io.lab import lab_to_recognition
        return lab_to_recognition(path)
    return RecognitionResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
