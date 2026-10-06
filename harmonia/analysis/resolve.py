"""Resolve competing explanations of the same chromatic chord.

Several rules may "explain" one chromatic chord (Event.explains), e.g. Gm7 in C major can be
a borrowed v7 or the ii of a ii–V into IV. Policy:
  * rank 2 — contextual, resolved explanations (ii–V–I into a non-tonic target, resolved
    secondary dominant, secondary leading-tone chord, tritone substitution). The chord's
    chromatic tone is accounted for by where it goes, which is stronger evidence than mere
    membership in a parallel mode.
  * rank 1 — borrowed chord, and deceptive / unresolved secondary dominants.
  Winner = highest (rank, confidence). Losers are not discarded silently: they are recorded
  in the winner's ``alternatives``.
A secondary dominant that is the V of a kept secondary ii–V with the same target is
consistent with it, so both are kept and the dominant gets ``part_of``.
"""

from __future__ import annotations

from ..schema import Event


def rank(e: Event) -> int:
    if e.type == "borrowed_chord":
        return 1
    if e.type == "secondary_dominant" and e.attributes.get("resolution") != "resolved":
        return 1
    return 2


def _compatible(a: Event, b: Event) -> bool:
    pair = {a.type, b.type}
    if pair == {"ii_V_I", "secondary_dominant"}:
        ii = a if a.type == "ii_V_I" else b
        sd = b if ii is a else a
        return (sd.explains[0] == ii.segment_indices[1]
                and sd.attributes.get("target_root") is not None
                and sd.attributes.get("resolution") == "resolved")
    return False


def _summary(e: Event) -> dict:
    return {"type": e.type, "label": e.label, "confidence": e.confidence, "rule": e.rule}


def resolve(events: list[Event], cfg: dict) -> list[Event]:
    g = cfg["general"]
    events = [e for e in events if e.confidence >= g["min_event_confidence"]]

    # exact duplicates (same type over the same segments): keep the most confident
    uniq: dict[tuple, Event] = {}
    for e in events:
        k = (e.type, tuple(e.segment_indices))
        if k not in uniq or e.confidence > uniq[k].confidence:
            uniq[k] = e
    events = list(uniq.values())

    claimed: dict[int, list[Event]] = {}
    kept: list[Event] = []
    for e in sorted(events, key=lambda e: (rank(e), e.confidence), reverse=True):
        rivals = [w for s in e.explains for w in claimed.get(s, []) if not _compatible(w, e)]
        if rivals:
            rivals[0].alternatives.append(_summary(e))
            continue
        kept.append(e)
        for s in e.explains:
            claimed.setdefault(s, []).append(e)

    kept.sort(key=lambda e: (e.start, e.type != "modulation", -e.confidence))
    for n, e in enumerate(kept):
        e.id = f"ev{n}"
        e.low_confidence = e.confidence < g["low_confidence_threshold"]
    for sd in (e for e in kept if e.type == "secondary_dominant"):
        for ii in (e for e in kept if e.type == "ii_V_I" and e.attributes.get("tonicization")):
            if _compatible(ii, sd):
                sd.part_of = ii.id
    return kept
