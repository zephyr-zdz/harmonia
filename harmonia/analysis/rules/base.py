"""Shared helpers for rules: confidence accounting and event construction.

Confidence model: an event's confidence is a product of named factors in [0, 1], each
recorded as an Evidence entry, so every judgement carries its reasons ("判断依据").
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...schema import Event, Evidence
from ...theory.chord import Chord
from ..context import AnalysisContext


class Score:
    def __init__(self) -> None:
        self.value = 1.0
        self.evidence: list[Evidence] = []

    def apply(self, kind: str, detail: str, factor: float) -> "Score":
        factor = min(max(factor, 0.0), 1.0)
        self.value *= factor
        self.evidence.append(Evidence(kind, detail, round(factor, 4)))
        return self

    def note(self, kind: str, detail: str) -> "Score":
        """Evidence that does not change the confidence."""
        self.evidence.append(Evidence(kind, detail, 1.0))
        return self


def is_class(*qclasses: str) -> Callable[[Chord], float]:
    def w(ch: Chord) -> float:
        return 1.0 if ch.qclass in qclasses else 0.0
    return w


def quality_factor(ctx: AnalysisContext, match: float, contradict: float) -> float:
    """base + (1-base) * P(match) - penalty * P(contradict)  (see [quality] in config)."""
    q = ctx.cfg["quality"]
    return q["base"] + (1.0 - q["base"]) * match - q["contradict_penalty"] * contradict


def make_event(ctx: AnalysisContext, *, type_: str, rule: str, label: str, seg_indices: list[int],
               score: Score, key_label: str | None, functions: list[str] | None = None,
               attributes: dict[str, Any] | None = None, explains: list[int] | None = None) -> Event:
    start, end = ctx.span(seg_indices)
    return Event(
        id="",
        type=type_,
        label=label,
        start=start,
        end=end,
        segment_indices=list(seg_indices),
        key=key_label,
        chords=[ctx.chord_label(s) for s in seg_indices],
        numerals=[ctx.numeral(s) for s in seg_indices],
        functions=functions or [ctx.numeral(s) for s in seg_indices],
        confidence=round(min(max(score.value, 0.0), 1.0), 4),
        low_confidence=False,
        rule=rule,
        evidence=score.evidence,
        attributes=attributes or {},
        explains=explains or [],
    )
