"""Chord-recognition metrics.

* Time-aligned references (.lab): mir_eval.chord.evaluate — duration-weighted root /
  majmin / sevenths (+ thirds, triads, tetrads, mirex, segmentation), the MIREX standard.
* Chart references (no timing): chord-symbol error rate from sequence edit distance at
  root / majmin / sevenths level (unbiased), plus mir_eval scores on a chart→estimate
  alignment, reported separately as ``aligned_*`` (APPROXIMATE and optimistic: the timing of
  the reference is borrowed from the estimate itself).
"""

from __future__ import annotations

import warnings
from typing import Any

import mir_eval
import numpy as np

from ..schema import AnalysisResult, RecognitionResult
from ..theory.chord import UNKNOWN_CHORD, Chord, ChordParseError, parse_chord
from .align import align, est_to_ref_map

LEVELS = ("root", "majmin", "sevenths")
_REPORTED = ("root", "majmin", "sevenths", "thirds", "triads", "tetrads", "mirex", "seg")
_COMPARATORS = {"root": mir_eval.chord.root, "majmin": mir_eval.chord.majmin,
                "sevenths": mir_eval.chord.sevenths}


def to_harte(label: str) -> str:
    try:
        return parse_chord(label).harte()
    except ChordParseError:
        return "X"


def top1_intervals(rec: RecognitionResult) -> tuple[list[tuple[float, float]], list[str]]:
    """Top candidate per frame, consecutive identical labels merged."""
    iv: list[list[float]] = []
    labels: list[str] = []
    for f in rec.frames:
        lab = to_harte(max(f.candidates, key=lambda c: c.prob).label) if f.candidates else "X"
        if labels and labels[-1] == lab and abs(iv[-1][1] - f.time) < 1e-6:
            iv[-1][1] = f.time + f.duration
        else:
            iv.append([f.time, f.time + f.duration])
            labels.append(lab)
    return [tuple(x) for x in iv], labels


def mir_eval_scores(ref_iv, ref_lab, est_iv, est_lab) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # mir_eval warns about trimming / padding to ref span
        s = mir_eval.chord.evaluate(np.array(ref_iv, dtype=float), [to_harte(l) for l in ref_lab],
                                    np.array(est_iv, dtype=float), [to_harte(l) for l in est_lab])
    return {k: float(s[k]) for k in _REPORTED if k in s}


def _equal(level: str, a: str, b: str) -> bool:
    c = _COMPARATORS[level]([a], [b])[0]
    if c < 0:  # out of the level's vocabulary (e.g. sus under majmin): fall back to root
        c = mir_eval.chord.root([a], [b])[0]
    return c >= 1.0


def symbol_error_rates(ref_labels: list[str], est_labels: list[str]) -> dict[str, float]:
    """Normalised edit distance (ins/del/sub = 1) between merged chord sequences."""
    out = {}
    for level in LEVELS:
        ref = [to_harte(x) for x in ref_labels]
        est = [to_harte(x) for x in est_labels]

        def sub(a: Chord, b: Chord, level=level) -> float:
            return 0.0 if _equal(level, a.harte(), b.harte()) else 1.0

        cost, _ = align([_parse(x) for x in ref], [_parse(x) for x in est], sub=sub)
        out[f"cer_{level}"] = cost / max(len(ref), 1)
    return out


def _parse(h: str) -> Chord:
    try:
        return parse_chord(h)
    except ChordParseError:
        return UNKNOWN_CHORD


def chart_aligned(ref_res: AnalysisResult, est_res: AnalysisResult) -> tuple[dict[str, Any], list[int | None]]:
    """Align the chart's segments to the estimate's segments; return approximate mir_eval
    scores on the estimate's timeline, and the est-segment -> chart-segment map."""
    ref_chords = [_parse(s.chord_harte) for s in ref_res.segments]
    est_chords = [_parse(s.chord_harte) for s in est_res.segments]
    cost, path = align(ref_chords, est_chords)
    emap = est_to_ref_map(path, len(est_chords))
    est_iv = [(s.start, s.end) for s in est_res.segments]
    est_lab = [s.chord_harte for s in est_res.segments]
    ref_lab_on_est = [ref_res.segments[i].chord_harte if i is not None else "N" for i in emap]
    scores = mir_eval_scores(est_iv, ref_lab_on_est, est_iv, est_lab)
    out = {f"aligned_{k}": v for k, v in scores.items() if k in LEVELS}
    out["alignment_cost"] = cost
    out["unmatched_ref_chords"] = sum(1 for i, j in path if j is None)
    return out, emap
