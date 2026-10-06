"""Analysis pipeline: RecognitionResult -> AnalysisResult.

segments -> local keys (HMM) -> roman numerals -> rules -> conflict resolution -> output.
"""

from __future__ import annotations

from typing import Any

from ..config import load_config
from ..schema import (
    AnalysisResult, ChordCandidate, GlobalKey, KeyLabel, KeyRegion, NamedProgression, RecognitionResult,
    RomanInfo, Segment,
)
from ..theory.key import Key, parse_key
from ..theory.roman import roman
from .context import AnalysisContext
from .idioms import key_prior_bonus, match_idioms
from .key_model import estimate_keys
from .resolve import resolve
from .rules import RULES
from .segments import SegData, build_segments


# Events whose per-segment `functions` are copied onto the segments (e.g. "V7/vi").
_FUNCTIONAL_EVENTS = {"ii_V_I", "ii_V", "secondary_dominant", "secondary_leading_tone", "tritone_sub"}


def _key_label(key: Key, prob: float) -> KeyLabel:
    return KeyLabel(key.label, key.tonic_name, key.mode, round(prob, 4))


def build_context(rec: RecognitionResult, cfg: dict[str, Any] | None = None,
                  key: str | Key | None = None) -> tuple[AnalysisContext, list[str]]:
    cfg = cfg or load_config()
    segs, warnings = build_segments(rec, cfg)
    if not segs:
        raise ValueError("input has no frames")
    given = parse_key(key) if isinstance(key, str) else key
    ctx = AnalysisContext(segs, None, cfg)
    idioms = match_idioms(ctx) if cfg["named_progressions"]["enabled"] else []
    ka = estimate_keys(segs, cfg, given, prior=key_prior_bonus(idioms, len(segs), cfg))
    for s, k, p in zip(segs, ka.seg_keys, ka.seg_probs):
        s.key, s.key_prob = k, p
    ctx.keys = ka
    ctx.idioms = idioms
    return ctx, warnings


def _segment_out(s: SegData, cfg: dict) -> Segment:
    g = cfg["general"]
    thr = g["low_confidence_threshold"]
    ch = s.chord
    rn = roman(ch, s.key) if s.key is not None else None
    warnings = list(s.warnings)
    conf = s.top.prob
    low = False
    if ch.is_unknown:
        low = True
        warnings.append("chord unknown / unparseable")
    elif conf < thr:
        low = True
        warnings.append(f"chord confidence {conf:.2f} < {thr}")
    if not ch.is_no_chord and s.key_prob < thr:
        low = True
        warnings.append(f"key confidence {s.key_prob:.2f} < {thr}")
    spell = s.key.spell if s.key is not None else (lambda pc: str(pc))
    return Segment(
        index=s.index, start=s.start, end=s.end, beats=round(s.beats, 4),
        chord=s.top.label, chord_harte=ch.harte(),
        candidates=[ChordCandidate(c.label, round(c.prob, 4)) for c in s.dist.cands],
        root=(ch.root_name or spell(ch.root)) if ch.root is not None else None,
        bass=s.bass or (spell(ch.bass_pc) if ch.bass_pc is not None else None),
        key=_key_label(s.key, s.key_prob) if s.key is not None else None,
        roman=RomanInfo(rn.numeral, rn.display, rn.degree, rn.diatonic, round(conf * s.key_prob, 4)) if rn else None,
        confidence=round(conf, 4), low_confidence=low, warnings=warnings, section=s.section, bar=s.bar,
    )


def analyze(rec: RecognitionResult, config: dict[str, Any] | None = None,
            key: str | Key | None = None) -> AnalysisResult:
    cfg = config or load_config()
    ctx, warnings = build_context(rec, cfg, key)
    warnings = list(rec.warnings) + warnings

    raw = []
    for name, detect in RULES.items():
        if cfg[name]["enabled"]:
            raw.extend(detect(ctx))
    events = resolve(raw, cfg)

    segments = [_segment_out(s, cfg) for s in ctx.segs]
    for e in events:
        for si, func in zip(e.segment_indices, e.functions):
            seg = segments[si]
            seg.event_ids.append(e.id)
            if e.type in _FUNCTIONAL_EVENTS and func not in seg.functions \
                    and (seg.roman is None or func != seg.roman.display):
                seg.functions.append(func)

    ka = ctx.keys
    gk = GlobalKey(
        key=_key_label(ka.global_key, ka.global_dist[0][1]),
        alternatives=[_key_label(k, p) for k, p in ka.global_dist[1:4]],
        ambiguous=ka.ambiguous,
        source=ka.source,
    )
    if ka.ambiguous:
        a, b = ka.global_dist[0], ka.global_dist[1]
        warnings.append(f"global key ambiguous: {a[0].label} ({a[1]:.2f}) vs {b[0].label} ({b[1]:.2f})")
    regions = []
    for a, b, k in ka.regions:
        mean_p = sum(ctx.segs[i].key_prob for i in range(a, b)) / (b - a)
        regions.append(KeyRegion(ctx.segs[a].start, ctx.segs[b - 1].end, [a, b], _key_label(k, mean_p)))

    progressions = []
    for m in ctx.idioms:
        progressions.append(NamedProgression(
            name=m.name, alias=m.alias, start=ctx.segs[m.seg_indices[0]].start,
            end=ctx.segs[m.seg_indices[-1]].end, segment_indices=m.seg_indices,
            reference_key=m.ref_key.label,
            numerals=[roman(ctx.segs[s].chord, m.ref_key).display for s in m.seg_indices],
            confidence=m.confidence, key_prior_applied=m.use_key_prior and ka.source != "given"))

    return AnalysisResult(
        time_unit=rec.time_unit, global_key=gk, key_regions=regions, segments=segments, events=events,
        progressions=progressions,
        warnings=warnings, source=rec.source,
        config={"key_given": ka.source == "given"},
    )
