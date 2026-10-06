"""Test helpers. Expectations in tests encode music theory; never change them just to make a
test pass (see CLAUDE.md)."""

from __future__ import annotations

from harmonia import analyze_progression
from harmonia.schema import AnalysisResult, Event

# Event types that claim something non-trivial about a chord; diatonic progressions must
# produce none of these.
NOTABLE = ("ii_V_I", "secondary_dominant", "secondary_leading_tone", "borrowed_chord",
           "tritone_sub", "aeolian_cadence", "modulation")


def run(text: str, key: str | None = None) -> AnalysisResult:
    return analyze_progression(text, key=key)


def numerals(res: AnalysisResult) -> list[str]:
    return [s.roman.display if s.roman else "-" for s in res.segments]


def events(res: AnalysisResult, type_: str | None = None) -> list[Event]:
    return [e for e in res.events if type_ is None or e.type == type_]


def labels(res: AnalysisResult, type_: str | None = None) -> list[str]:
    return [e.label for e in events(res, type_)]


def at(res: AnalysisResult, type_: str, seg: int) -> list[Event]:
    """Events of a type that involve segment ``seg``."""
    return [e for e in events(res, type_) if seg in e.segment_indices]
