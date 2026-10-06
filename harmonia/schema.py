"""The JSON contract between the recognition layer and the analysis layer.

RecognitionResult  (recognition -> analysis): time-ordered frames, normally one per beat,
    each with top-k chord candidates + probabilities, an optional bass observation and an
    optional local-key guess. Symbolic input produces one frame per chord slot instead.
AnalysisResult     (analysis -> UI / evaluation): merged chord segments with local key,
    roman numeral and functional labels; key regions; harmonic events with confidence and
    evidence. Documented with an example in docs/schema.md.

Bump SCHEMA_VERSION on any incompatible change.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "0.1.0"
TIME_UNITS = ("second", "beat")


# ----------------------------------------------------------------- recognition -> analysis

@dataclass
class ChordCandidate:
    label: str   # pop ("Bbm7b5") or Harte ("Bb:hdim7") symbol
    prob: float


@dataclass
class BassObservation:
    note: str | None   # e.g. "E"; None = no bass detected
    prob: float = 1.0


@dataclass
class KeyGuess:
    label: str          # e.g. "C major", "A:min"
    prob: float = 1.0


@dataclass
class Frame:
    time: float                      # onset, in RecognitionResult.time_unit
    duration: float
    candidates: list[ChordCandidate]  # top-k, sorted or not; probs need not sum to 1
    beats: float | None = None       # how many beats this frame spans (1.0 for per-beat frames)
    bass: BassObservation | None = None
    key: KeyGuess | None = None
    bar: int | None = None           # 1-based bar number, if known
    beat_in_bar: float | None = None # 1-based position within the bar, if known
    section: str | None = None       # e.g. "Verse", "Chorus"


@dataclass
class RecognitionResult:
    frames: list[Frame]
    time_unit: str = "second"
    beats_per_bar: int | None = 4
    source: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RecognitionResult":
        frames = []
        for f in d["frames"]:
            frames.append(Frame(
                time=float(f["time"]),
                duration=float(f["duration"]),
                candidates=[ChordCandidate(c["label"], float(c["prob"])) for c in f["candidates"]],
                beats=f.get("beats"),
                bass=BassObservation(**f["bass"]) if f.get("bass") else None,
                key=KeyGuess(**f["key"]) if f.get("key") else None,
                bar=f.get("bar"),
                beat_in_bar=f.get("beat_in_bar"),
                section=f.get("section"),
            ))
        unit = d.get("time_unit", "second")
        if unit not in TIME_UNITS:
            raise ValueError(f"time_unit must be one of {TIME_UNITS}, got {unit!r}")
        return cls(frames=frames, time_unit=unit, beats_per_bar=d.get("beats_per_bar", 4),
                   source=d.get("source", {}), warnings=list(d.get("warnings", [])),
                   schema_version=d.get("schema_version", SCHEMA_VERSION))


# ----------------------------------------------------------------- analysis -> consumers

@dataclass
class KeyLabel:
    label: str      # "C major"
    tonic: str      # "C"
    mode: str       # "major" | "minor"
    prob: float     # posterior / confidence (uncalibrated in schema 0.1)


@dataclass
class RomanInfo:
    numeral: str          # "IVmaj7"
    display: str          # numeral + inversion, e.g. "I/3"
    degree: int           # root, semitones above tonic
    diatonic: bool
    confidence: float     # chord prob * key prob


@dataclass
class Segment:
    index: int
    start: float
    end: float
    beats: float
    chord: str                         # top candidate, as given in the input
    chord_harte: str
    candidates: list[ChordCandidate]
    root: str | None
    bass: str | None
    key: KeyLabel | None
    roman: RomanInfo | None
    functions: list[str] = field(default_factory=list)  # e.g. ["V7/vi"]
    event_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0            # top candidate probability
    low_confidence: bool = False
    warnings: list[str] = field(default_factory=list)
    section: str | None = None
    bar: int | None = None


@dataclass
class KeyRegion:
    start: float
    end: float
    segment_range: list[int]   # [first, last_exclusive]
    key: KeyLabel


@dataclass
class Evidence:
    kind: str      # e.g. "root_motion", "quality_ii", "resolution", "key"
    detail: str
    factor: float  # multiplicative contribution to the confidence


@dataclass
class Event:
    id: str
    type: str                       # ii_V_I | secondary_dominant | secondary_leading_tone |
                                    # borrowed_chord | tritone_sub | aeolian_cadence | modulation
    label: str                      # human-readable, e.g. "V7/vi", "ii–V–I → IV"
    start: float
    end: float
    segment_indices: list[int]
    key: str | None                 # local key the numerals refer to
    chords: list[str]
    numerals: list[str]
    functions: list[str]            # functional reading per segment_indices entry
    confidence: float
    low_confidence: bool
    rule: str
    evidence: list[Evidence] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    explains: list[int] = field(default_factory=list)       # chromatic segments this explains
    alternatives: list[dict[str, Any]] = field(default_factory=list)  # losing explanations
    part_of: str | None = None


@dataclass
class GlobalKey:
    key: KeyLabel
    alternatives: list[KeyLabel]
    ambiguous: bool
    source: str   # "estimated" | "given"


@dataclass
class AnalysisResult:
    time_unit: str
    global_key: GlobalKey
    key_regions: list[KeyRegion]
    segments: list[Segment]
    events: list[Event]
    warnings: list[str] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def events_of(self, type_: str) -> list[Event]:
        return [e for e in self.events if e.type == type_]
