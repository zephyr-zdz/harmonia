"""Shared state for the rules: segments, key estimates, root "runs" and navigation.

A *run* is a maximal stretch of consecutive segments whose top chord has the same root
(e.g. ``G7sus4 G7`` or ``C C7``). Rules reason about root motion between runs, which makes
them insensitive to quality changes over a held root and to repeated chords.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..theory.chord import Chord
from ..theory.key import Key
from ..theory.roman import roman
from .key_model import KeyAnalysis
from .segments import SegData


@dataclass
class Run:
    index: int
    segs: list[int]
    root: int | None
    beats: float


def build_runs(segs: list[SegData]) -> list[Run]:
    runs: list[Run] = []
    for s in segs:
        r = s.chord.root
        if runs and runs[-1].root == r:
            runs[-1].segs.append(s.index)
            runs[-1].beats += s.beats
        else:
            runs.append(Run(len(runs), [s.index], r, s.beats))
    return runs


class AnalysisContext:
    def __init__(self, segs: list[SegData], keys: KeyAnalysis | None, cfg: dict):
        # keys is None only during the key-independent pre-pass (named progressions)
        self.segs = segs
        self.keys = keys
        self.cfg = cfg
        self.runs = build_runs(segs)
        self.idioms: list = []  # IdiomMatch list, filled by pipeline.build_context
        g = cfg["general"]
        self.min_root_prob: float = g["min_root_prob"]
        self.min_conf: float = g["min_event_confidence"]

    # ---- navigation -------------------------------------------------------------------
    def _transparent(self, run: Run) -> bool | None:
        """True = skip over, False = usable, None = blocks the search."""
        g = self.cfg["general"]
        if run.root is None:
            return True if run.beats < g["skip_no_chord_beats"] else None
        return run.beats < g["passing_beats"]

    def _walk(self, i: int, step: int) -> int | None:
        j = i + step
        while 0 <= j < len(self.runs):
            t = self._transparent(self.runs[j])
            if t is None:
                return None
            if not t:
                return j
            j += step
        return None

    def next_run(self, i: int) -> int | None:
        return self._walk(i, +1)

    def prev_run(self, i: int) -> int | None:
        return self._walk(i, -1)

    # ---- soft evidence over a run -------------------------------------------------------
    def root_prob(self, run: Run, pc: int) -> float:
        return max(self.segs[s].dist.root_prob(pc % 12) for s in run.segs)

    def roots(self, run: Run) -> list[int]:
        out: list[int] = []
        for s in run.segs:
            for r in self.segs[s].dist.roots(self.min_root_prob):
                if r not in out:
                    out.append(r)
        return out

    def best_mass(self, run: Run, pc: int, weight: Callable[[Chord], float],
                  prefer_last: bool = True) -> tuple[float, int]:
        """max over the run's segments of sum_{cand rooted on pc} p * weight(chord)."""
        order = run.segs if prefer_last else list(reversed(run.segs))
        best_v, best_s = -1.0, run.segs[-1] if prefer_last else run.segs[0]
        for s in order:
            v = self.segs[s].dist.mass(pc % 12, weight)
            if v >= best_v:
                best_v, best_s = v, s
        return max(best_v, 0.0), best_s

    def cond(self, seg: int, pc: int, weight: Callable[[Chord], float]) -> float:
        return self.segs[seg].dist.cond(pc % 12, weight)

    # ---- keys / numerals ------------------------------------------------------------------
    def key_of(self, seg: int) -> tuple[Key, float]:
        s = self.segs[seg]
        assert s.key is not None
        return s.key, s.key_prob

    def key_factor(self, prob: float) -> float:
        return prob ** self.cfg["general"]["key_confidence_exponent"]

    def numeral(self, seg: int) -> str:
        s = self.segs[seg]
        rn = roman(s.chord, s.key) if s.key is not None else None
        return rn.display if rn else s.chord.label()

    def chord_label(self, seg: int) -> str:
        return self.segs[seg].top.label

    def span(self, seg_indices: list[int]) -> tuple[float, float]:
        return self.segs[min(seg_indices)].start, self.segs[max(seg_indices)].end
