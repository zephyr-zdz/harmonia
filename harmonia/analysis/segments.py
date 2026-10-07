"""Turn recognition frames into chord segments carrying a soft chord distribution.

Consecutive frames with the same top chord (and section) are merged; their candidate
probabilities are averaged, weighted by beats. Unparseable candidates are dropped WITH a
warning; a frame with no parseable candidate becomes an "X" segment with confidence 0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..schema import RecognitionResult
from ..theory.chord import UNKNOWN_CHORD, Chord, ChordParseError, parse_chord
from ..theory.key import Key
from ..theory.pitch import parse_note_prefix


@dataclass(frozen=True)
class Cand:
    chord: Chord
    prob: float
    label: str


class ChordDist:
    """Soft chord evidence for one segment (top-k candidates; mass may sum to < 1)."""

    def __init__(self, cands: list[Cand]):
        self.cands = sorted(cands, key=lambda c: -c.prob)

    @property
    def top(self) -> Cand:
        return self.cands[0]

    def root_prob(self, pc: int) -> float:
        return sum(c.prob for c in self.cands if c.chord.root == pc)

    def mass(self, pc: int, weight: Callable[[Chord], float]) -> float:
        """Sum of prob * weight(chord) over candidates rooted on ``pc``."""
        return sum(c.prob * weight(c.chord) for c in self.cands if c.chord.root == pc)

    def cond(self, pc: int, weight: Callable[[Chord], float]) -> float:
        """E[weight | root == pc]."""
        rp = self.root_prob(pc)
        return self.mass(pc, weight) / rp if rp > 0 else 0.0

    def roots(self, min_prob: float) -> list[int]:
        roots: dict[int, float] = {}
        for c in self.cands:
            if c.chord.root is not None:
                roots[c.chord.root] = roots.get(c.chord.root, 0.0) + c.prob
        return [r for r, p in sorted(roots.items(), key=lambda kv: -kv[1]) if p >= min_prob]


@dataclass
class SegData:
    index: int
    start: float
    end: float
    beats: float
    dist: ChordDist
    frame_range: tuple[int, int]
    bass: str | None = None
    bass_head: dict[int, float] = field(default_factory=dict)   # P(bass pc) from the recogniser's bass head
    section: str | None = None
    bar: int | None = None
    warnings: list[str] = field(default_factory=list)
    key: Key | None = None
    key_prob: float = 0.0

    @property
    def top(self) -> Cand:
        return self.dist.top

    @property
    def chord(self) -> Chord:
        return self.dist.top.chord


def _frame_cands(rec: RecognitionResult, fi: int, warnings: list[str]) -> list[Cand]:
    fr = rec.frames[fi]
    cands: list[Cand] = []
    for c in fr.candidates:
        try:
            cands.append(Cand(parse_chord(c.label), float(c.prob), c.label))
        except ChordParseError as e:
            warnings.append(f"frame {fi} @ {fr.time:g}: dropped unparseable candidate {c.label!r} ({e})")
    total = sum(c.prob for c in cands)
    if total > 1.0 + 1e-6:
        warnings.append(f"frame {fi} @ {fr.time:g}: candidate probs sum to {total:.3f} > 1; normalised")
        cands = [Cand(c.chord, c.prob / total, c.label) for c in cands]
    if not cands:
        warnings.append(f"frame {fi} @ {fr.time:g}: no parseable chord candidate; marked X (unknown)")
        cands = [Cand(UNKNOWN_CHORD, 0.0, "X")]
    return cands


def build_segments(rec: RecognitionResult, cfg: dict) -> tuple[list[SegData], list[str]]:
    warnings: list[str] = []
    spb = cfg["general"]["default_seconds_per_beat"]
    warned_beats = False

    # (frame cands, beats) per frame
    per_frame: list[tuple[list[Cand], float, list[str]]] = []
    for fi, fr in enumerate(rec.frames):
        fw: list[str] = []
        cands = _frame_cands(rec, fi, fw)
        if fr.beats is not None:
            beats = float(fr.beats)
        elif rec.time_unit == "beat":
            beats = fr.duration
        else:
            beats = fr.duration / spb
            if not warned_beats:
                warnings.append(f"no beat information; weighting chords assuming {60 / spb:g} BPM")
                warned_beats = True
        warnings.extend(fw)
        per_frame.append((cands, max(beats, 1e-6), fw))

    segs: list[SegData] = []
    groups: list[list[int]] = []
    for fi, (cands, _, _) in enumerate(per_frame):
        top_key = (cands[0].chord.harte(), rec.frames[fi].section)
        if groups:
            prev = groups[-1][-1]
            prev_key = (per_frame[prev][0][0].chord.harte(), rec.frames[prev].section)
            if prev_key == top_key and top_key[0] != "X":
                groups[-1].append(fi)
                continue
        groups.append([fi])

    for gi, group in enumerate(groups):
        total_beats = sum(per_frame[fi][1] for fi in group)
        acc: dict[str, list] = {}  # harte -> [Cand(first seen), weighted prob]
        bass_votes: dict[str, float] = {}
        seg_warn: list[str] = []
        for fi in group:
            cands, beats, fw = per_frame[fi]
            seg_warn.extend(fw)
            w = beats / total_beats
            for c in cands:
                h = c.chord.harte()
                if h not in acc:
                    acc[h] = [c, 0.0]
                acc[h][1] += c.prob * w
            fb = rec.frames[fi].bass
            if fb is not None and fb.note:
                bass_votes[fb.note] = bass_votes.get(fb.note, 0.0) + fb.prob * beats
        merged = [Cand(c.chord, p, c.label) for c, p in acc.values()]
        first, last = rec.frames[group[0]], rec.frames[group[-1]]
        segs.append(SegData(
            index=gi,
            start=first.time,
            end=last.time + last.duration,
            beats=total_beats,
            dist=ChordDist(merged),
            frame_range=(group[0], group[-1] + 1),
            bass=max(bass_votes, key=bass_votes.get) if bass_votes else None,
            bass_head=_bass_head(bass_votes, total_beats),
            section=first.section,
            bar=first.bar,
            warnings=seg_warn,
        ))
    return segs, warnings


def _bass_head(votes: dict[str, float], total_beats: float) -> dict[int, float]:
    """note -> prob·beats votes → {pitch class: share of the segment} (sums to ≤ 1)."""
    out: dict[int, float] = {}
    if total_beats <= 0:
        return out
    for note, v in votes.items():
        pc = parse_note_prefix(note)[0]
        out[pc] = out.get(pc, 0.0) + v / total_beats
    return out
