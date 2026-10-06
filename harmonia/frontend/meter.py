"""Tempo -> meter -> bar grid (user suggestion 2026-10-06: estimate BPM first, then bar lines).

Raw tracker output (Beat This!) has reliable downbeats but beats that can be missing or
doubled. We therefore:
  1. estimate the beat period P robustly: inter-beat intervals folded by small integers
     (an interval of 2P or 3P is a missed beat), median of the folded values;
  2. estimate the bar period D the same way from downbeat intervals;
  3. meter m = round(D / P)  (6 at ♪-level in 6/8, 4 in 4/4 ...) and BPM = 60·m / D, which is
     more accurate than 60 / median(IBI) because bar lines are detected more reliably;
  4. bar lines: keep detected downbeats whose spacing is a whole number of beats; fill gaps of
     k bars with k-1 bar lines; drop downbeats closer than 2 beats to the previous one;
     a genuinely short / long bar (e.g. one 2/4 bar inside 4/4, common in J-pop) is kept and
     reported as irregular;
  5. beats: each bar is divided into round(bar / P) equal beats, each snapped to a detected
     beat within ±snap·P (keeps expressive timing); pickup before the first bar line and the
     tail after the last are extrapolated with period P.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Beat counts per bar -> time-signature label. Beat tracking cannot tell simple from compound
# meter when it tracks the dotted-quarter level (2 beats could be 2/4 or 6/8), so those are
# labelled with both readings rather than guessed.
_TIME_SIG = {2: "2 beats/bar (2/4 or 6/8)", 3: "3/4", 4: "4 beats/bar (4/4 or 12/8)", 6: "6/8",
             8: "8 beats/bar (4/4 at 8th level)", 9: "9/8", 12: "12/8"}


@dataclass
class MeterGrid:
    beats: np.ndarray            # regularised beat times (s)
    bar_index: list[int]         # bar number per beat (0 = pickup)
    beat_in_bar: list[int]       # 1-based position per beat
    bar_starts: np.ndarray       # bar line times (s)
    beat_period: float
    bar_period: float
    beats_per_bar: int
    bpm: float                   # at the tracked beat level
    time_signature: str
    tempo_cv: float              # coefficient of variation of bar durations
    irregular_bars: list[tuple[int, int]] = field(default_factory=list)  # (bar number, beats)
    warnings: list[str] = field(default_factory=list)


def folded_period(intervals: np.ndarray, max_fold: int = 4, lo: float = 0.15, hi: float = 2.5) -> float:
    """Robust period from intervals that may be integer multiples of it (missed events) or
    spuriously short (double triggers).

    Seed = mode of a 10 ms histogram of intervals in [lo, hi] (smoothed); refine with the median
    of intervals >= 0.7·seed folded by k = round(x / seed) ≤ max_fold.
    """
    iv = intervals[intervals > 1e-3]
    if len(iv) == 0:
        raise ValueError("no intervals")
    inr = iv[(iv >= lo) & (iv <= hi)]
    if len(inr) == 0:
        inr = iv
    bins = np.arange(inr.min() - 0.005, inr.max() + 0.015, 0.01)
    hist, edges = np.histogram(inr, bins=bins)
    hist = np.convolve(hist, np.ones(3), mode="same")
    p0 = float((edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2)
    for _ in range(2):
        folded = []
        for x in iv[iv >= 0.7 * p0]:
            k = max(1, min(max_fold, int(round(x / p0))))
            if abs(x / k - p0) / p0 < 0.12:
                folded.append(x / k)
        if not folded:
            break
        p0 = float(np.median(folded))
    return p0


def build_grid(beats: np.ndarray, downbeats: np.ndarray, duration: float, snap: float = 0.25) -> MeterGrid:
    beats = np.sort(np.asarray(beats, float))
    downbeats = np.sort(np.asarray(downbeats, float))
    warnings: list[str] = []
    P = folded_period(np.diff(beats))
    while 60.0 / P > 240.0:  # tempo-octave guard: beat level above 240 BPM is a sub-beat
        P *= 2
        warnings.append("beat level above 240 BPM; using every other beat")
    if len(downbeats) >= 3:
        D = folded_period(np.diff(downbeats), max_fold=3, lo=0.6, hi=6.0)
        m = max(1, int(round(D / P)))
    else:
        warnings.append("too few downbeats: assuming 4 beats per bar, bar lines unreliable")
        m, D = 4, 4 * P
        downbeats = np.array([beats[0]]) if len(beats) else np.array([0.0])
    P = D / m  # bar lines are the more reliable tempo evidence

    # --- bar lines
    bars = [float(downbeats[0])]
    irregular: list[tuple[int, int]] = []
    db = list(downbeats[1:])

    def whole_bars(gap: float) -> int | None:
        k = int(round(gap / D))
        return k if k >= 1 and abs(gap - k * D) < 0.5 * P else None

    for i, d in enumerate(db):
        gap = d - bars[-1]
        nbeats = int(round(gap / P))
        if nbeats < 2:
            continue  # spurious: too close to the previous bar line
        k = whole_bars(gap)
        if k is None:
            # Off-grid downbeat: spurious if skipping it leaves the next one on the bar grid;
            # otherwise a genuinely irregular bar (e.g. a 2/4 bar inside 4/4).
            if i + 1 < len(db) and whole_bars(db[i + 1] - bars[-1]) is not None:
                continue
            irregular.append((len(bars), nbeats))
        elif k >= 2:  # missed bar line(s): fill evenly
            bars.extend(bars[-1] + gap * j / k for j in range(1, k))
        bars.append(float(d))
    while bars[-1] + D < duration - 0.5 * P:  # extrapolate bar lines after the last downbeat
        bars.append(bars[-1] + D)
    bar_starts = np.array(bars)

    # --- beats: pickup, bars, tail
    def snapped(t: float) -> float:
        if len(beats) == 0:
            return t
        j = int(np.argmin(np.abs(beats - t)))
        return float(beats[j]) if abs(beats[j] - t) <= snap * P else t

    out_t: list[float] = []
    out_bar: list[int] = []
    out_pos: list[int] = []
    pickup = []
    t = bar_starts[0] - P
    while t > 0.25 * P:
        pickup.append(t)
        t -= P
    for j, t in enumerate(reversed(pickup)):
        out_t.append(snapped(t))
        out_bar.append(0)
        out_pos.append(m - len(pickup) + j + 1 if len(pickup) <= m else j + 1)
    for i, b0 in enumerate(bar_starts):
        b1 = bar_starts[i + 1] if i + 1 < len(bar_starts) else min(duration, b0 + D)
        n = max(1, int(round((b1 - b0) / P)))
        for k in range(n):
            tt = b0 + (b1 - b0) * k / n
            out_t.append(float(b0) if k == 0 else snapped(tt))
            out_bar.append(i + 1)
            out_pos.append(k + 1)

    bar_d = np.diff(bar_starts)
    cv = float(np.std(bar_d) / np.mean(bar_d)) if len(bar_d) > 1 else 0.0
    if cv > 0.08:
        warnings.append(f"tempo varies (bar-duration CV {cv:.2f}); grid follows detected bar lines")
    if irregular:
        warnings.append(f"{len(irregular)} irregular bar(s) (beats ≠ {m}): " +
                        ", ".join(f"bar {b}: {n}" for b, n in irregular[:8]))
    return MeterGrid(np.array(out_t), out_bar, out_pos, bar_starts, P, D, m, 60.0 / P,
                     _TIME_SIG.get(m, f"{m} beats/bar"), cv, irregular, warnings)
