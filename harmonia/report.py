"""Section-by-section harmonic summary of an analysed song (`harmonia report`).

Joins the structure (RecognitionResult.source["structure"], or `[Section]` markers on the
frames) with the analysis: per section the bar-by-bar numerals, the shortest repeating chord
loop, local key(s), named progressions, confident events and how much of it rests on
low-confidence chords. Pure post-processing of the two JSON layers (no audio code).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .schema import AnalysisResult, RecognitionResult


@dataclass
class SectionSummary:
    label: str
    group: str | None
    shift: int
    start: float
    end: float
    bars: list[str]                       # numerals per bar ("vim7 V/7" = two chords in the bar)
    loop: list[str]                       # shortest repeating bar pattern ([] if none)
    loop_cover: float                     # share of bars explained by repeating the loop
    keys: list[tuple[str, float]]         # local keys by share of section time
    progressions: list[str]
    events: list[tuple[str, str, float]]  # (type, label, confidence), not low-confidence
    low_confidence_share: float
    chords: list[str] = field(default_factory=list)   # chord symbols per bar (same layout as bars)


def _sections(rec: RecognitionResult) -> list[dict[str, Any]]:
    st = (rec.source or {}).get("structure") or {}
    if st.get("sections"):
        return st["sections"]
    out: list[dict[str, Any]] = []
    for f in rec.frames:
        if not f.section:
            continue
        if out and out[-1]["label"] == f.section:
            out[-1]["end"] = f.time + f.duration
        else:
            out.append({"label": f.section, "start": f.time, "end": f.time + f.duration, "cluster": None, "shift": 0})
    return out


def _bar_starts(rec: RecognitionResult) -> list[float]:
    return [f.time for f in rec.frames if f.beat_in_bar == 1]


def _loop(bars: list[str]) -> tuple[list[str], float]:
    """Shortest period p (1, 2, 4, 8 bars) such that most bars equal the bar p earlier."""
    n = len(bars)
    for p in (1, 2, 4, 8):
        if n < 2 * p:
            break
        same = sum(bars[i] == bars[i - p] for i in range(p, n))
        if same / (n - p) >= 0.6:
            return bars[:p], (same + p) / n
    return [], 0.0


def summarise(rec: RecognitionResult, res: AnalysisResult, min_overlap: float = 0.25) -> list[SectionSummary]:
    starts = _bar_starts(rec)
    end_time = res.segments[-1].end if res.segments else 0.0
    bars_iv = list(zip(starts, starts[1:] + [end_time]))
    out = []
    for s in _sections(rec):
        a, b = float(s["start"]), float(s["end"])
        segs = [g for g in res.segments if g.end > a and g.start < b]
        bars, chords = [], []
        for bs, be in bars_iv:
            if be <= a or bs >= b:
                continue
            here = [g for g in segs if min(g.end, be) - max(g.start, bs) >= min_overlap * (be - bs)]
            nums, chs = [], []
            for g in here:
                n = g.roman.display if g.roman else ("N" if g.chord_harte == "N" else "?")
                if not nums or nums[-1] != n:
                    nums.append(n)
                    chs.append(g.chord)
            bars.append(" ".join(nums) or "—")
            chords.append(" ".join(chs) or "—")
        dur = sum(min(g.end, b) - max(g.start, a) for g in segs) or 1.0
        kd: Counter[str] = Counter()
        low = 0.0
        for g in segs:
            w = min(g.end, b) - max(g.start, a)
            if g.key:
                kd[g.key.label] += w
            low += w * g.low_confidence
        keys = [(k, v / dur) for k, v in kd.most_common()]
        loop, cover = _loop(bars)
        progs = sorted({p.name for p in res.progressions if p.start < b and p.end > a and
                        min(p.end, b) - max(p.start, a) >= 0.5 * (p.end - p.start)})
        evs = [(e.type, e.label, round(e.confidence, 2)) for e in res.events
               if not e.low_confidence and a <= (e.start + e.end) / 2 < b]
        out.append(SectionSummary(s["label"], s.get("cluster"), int(s.get("shift") or 0), a, b, bars, loop,
                                  round(cover, 2), keys, progs, evs, round(low / dur, 2), chords))
    return out


def to_markdown(name: str, rec: RecognitionResult, res: AnalysisResult, secs: list[SectionSummary]) -> str:
    m = (rec.source or {}).get("meter") or {}
    st = (rec.source or {}).get("structure") or {}
    L = [f"## {name}", "",
         f"- global key **{res.global_key.key.label}** (p={res.global_key.key.prob:.2f})"
         + (" · ambiguous" if res.global_key.ambiguous else "")
         + (f" · {m.get('bpm')} BPM, {m.get('time_signature')}" if m else ""),
         f"- key regions: " + " → ".join(f"{r.key.label} @{r.start:.0f}s" for r in res.key_regions),
         f"- section labels from: {st.get('labels_from', 'markers')}", "",
         "| # | time | section | group | key | loop / bars | progressions | notable events | low-conf |",
         "|---|---|---|---|---|---|---|---|---|"]
    for i, s in enumerate(secs):
        key = ", ".join(f"{k}" + (f" {p:.0%}" if p < 0.95 else "") for k, p in s.keys[:2])
        body = (f"**loop** ‖ {' │ '.join(s.loop)} ‖ ({s.loop_cover:.0%})" if s.loop else " │ ".join(s.bars))
        ev = Counter(f"{lab}" for _, lab, _ in s.events)
        evs = ", ".join(f"{k}×{v}" if v > 1 else k for k, v in ev.most_common(4))
        L.append(f"| {i + 1} | {s.start:.0f}–{s.end:.0f}s | {s.label} | {s.group or ''}"
                 + (f" +{s.shift}" if s.shift else "") + f" | {key} | {body} | {', '.join(s.progressions)} | {evs} "
                 f"| {s.low_confidence_share:.0%} |")
    return "\n".join(L) + "\n"
