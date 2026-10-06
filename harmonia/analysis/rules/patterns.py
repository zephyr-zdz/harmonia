"""Multi-chord patterns: ♭VI–♭VII–I (Aeolian cadence) and key modulations.

♭VI–♭VII–I: two rising whole-step major chords landing on the tonic. In a major key both
♭VI and ♭VII are borrowed from the parallel minor, so the cadence itself is notable. In a
minor key it is diatonic (♭VI–♭VII–i) and only reported if include_minor_keys = true.

Modulation: a boundary between two key regions of the HMM path (regions shorter than
key.min_region_beats were already merged away as tonicizations). Confidence = mean key
posterior on each side of the boundary.
"""

from __future__ import annotations

from ...schema import Event
from ...theory.key import KEY_INDEX, Key
from ..context import AnalysisContext
from .base import Score, is_class, make_event, quality_factor

RULE_AEOLIAN = "aeolian_cadence"
RULE_MOD = "modulation"

_INTERVAL_NAMES = {1: "half step up", 2: "whole step up", 3: "minor 3rd up", 4: "major 3rd up",
                   5: "perfect 4th up", 6: "tritone", 7: "perfect 5th up", 8: "major 3rd down",
                   9: "minor 3rd down", 10: "whole step down", 11: "half step down"}


def detect_aeolian(ctx: AnalysisContext) -> list[Event]:
    cfg = ctx.cfg[RULE_AEOLIAN]
    out: list[Event] = []
    maj = is_class("maj", "dom")
    for j, irun in enumerate(ctx.runs):
        if irun.root is None:
            continue
        i7 = ctx.prev_run(j)
        i6 = ctx.prev_run(i7) if i7 is not None else None
        if i6 is None:
            continue
        key, _ = ctx.key_of(irun.segs[0])
        if not key.is_major and not cfg["include_minor_keys"]:
            continue
        t = key.tonic
        r6, r7 = ctx.runs[i6], ctx.runs[i7]
        p = ctx.root_prob(r6, t + 8) * ctx.root_prob(r7, t + 10) * ctx.root_prob(irun, t)
        if p < ctx.min_root_prob:
            continue
        s6 = ctx.best_mass(r6, t + 8, maj)[1]
        s7 = ctx.best_mass(r7, t + 10, maj)[1]
        s1 = irun.segs[0]
        sc = Score()
        sc.apply("root_motion", "roots ♭6 → ♭7 → 1 (rising whole steps to the tonic)", p)
        for s, pc, name in ((s6, t + 8, "♭VI"), (s7, t + 10, "♭VII")):
            sc.apply(f"quality_{name}", f"{ctx.chord_label(s)} as major {name}",
                     quality_factor(ctx, ctx.cond(s, pc, maj), ctx.cond(s, pc, is_class("min", "dim", "hdim"))))
        tq = "maj" if key.is_major else "min"
        sc.apply("quality_I", f"{ctx.chord_label(s1)} as tonic",
                 quality_factor(ctx, ctx.cond(s1, t, is_class(tq)), ctx.cond(s1, t, is_class("dim", "hdim"))))
        if sc.value < ctx.min_conf:
            continue
        label = "♭VI–♭VII–I" if key.is_major else "♭VI–♭VII–i"
        out.append(make_event(ctx, type_="aeolian_cadence", rule=RULE_AEOLIAN, label=label,
                              seg_indices=[s6, s7, s1], score=sc, key_label=key.label,
                              attributes={"borrowed": key.is_major}))
    return out


def describe_modulation(old: Key, new: Key) -> str:
    if old.tonic == new.tonic:
        return "parallel key"
    if old.relative == new:
        return "relative key"
    return _INTERVAL_NAMES[(new.tonic - old.tonic) % 12]


def detect_modulations(ctx: AnalysisContext) -> list[Event]:
    cfg = ctx.cfg[RULE_MOD]
    n_ctx = cfg["context_segments"]
    out: list[Event] = []
    regions = ctx.keys.regions
    post = ctx.keys.posteriors
    for (a0, b0, k_old), (a1, b1, k_new) in zip(regions, regions[1:]):
        before = range(max(a0, b0 - n_ctx), b0)
        after = range(a1, min(b1, a1 + n_ctx))
        p_old = sum(post[t][KEY_INDEX[k_old]] for t in before) / len(before)
        p_new = sum(post[t][KEY_INDEX[k_new]] for t in after) / len(after)
        sc = Score()
        sc.apply("key_before", f"{k_old.label} before the boundary (mean p={p_old:.2f})", p_old)
        sc.apply("key_after", f"{k_new.label} after the boundary (mean p={p_new:.2f})", p_new)
        desc = describe_modulation(k_old, k_new)
        segs = [b0 - 1, a1]
        ev = make_event(ctx, type_="modulation", rule=RULE_MOD, label=f"{k_old.label} → {k_new.label} ({desc})",
                        seg_indices=segs, score=sc, key_label=k_new.label,
                        functions=[ctx.numeral(b0 - 1), ctx.numeral(a1)],
                        attributes={"from": k_old.label, "to": k_new.label, "interval": desc,
                                    "semitones": (k_new.tonic - k_old.tonic) % 12})
        ev.start = ev.end = ctx.segs[a1].start
        out.append(ev)
    return out


RULE_DECEPTIVE = "deceptive_cadence"


def detect_deceptive(ctx: AnalysisContext) -> list[Event]:
    """Deceptive cadence: the primary dominant moves to vi (major) or ♭VI (major or minor)
    instead of the tonic.

    Theory: a deceptive (interrupted) cadence needs a *cadential* dominant. In pop loops a
    bare V→vi is often just passing motion (カノン進行 I–V–vi), so the V must be prepared by a
    predominant (ii or IV family) or carry its 7th; unprepared V7 gets a lower factor.
    """
    cfg = ctx.cfg[RULE_DECEPTIVE]
    dom = is_class("maj", "dom")
    out: list[Event] = []
    for j, vrun in enumerate(ctx.runs):
        if vrun.root is None:
            continue
        k = ctx.next_run(j)
        if k is None:
            continue
        key, kp = ctx.key_of(vrun.segs[-1])
        t = key.tonic
        p_v, v_seg = ctx.best_mass(vrun, t + 7, dom)
        if p_v < ctx.min_root_prob:
            continue
        nxt = ctx.runs[k]
        targets = [(t + 8, "♭VI", is_class("maj"))]
        if key.is_major:
            targets.insert(0, (t + 9, "vi", is_class("min")))
        best: Event | None = None
        for root, name, tq in targets:
            p_t = ctx.root_prob(nxt, root)
            if p_t < ctx.min_root_prob:
                continue
            res_seg = nxt.segs[0]
            sc = Score()
            sc.apply("dominant", f"{ctx.chord_label(v_seg)} = V of {key.label}", p_v)
            sc.apply("root_motion", f"V moves up a step to {name} instead of I", p_t)
            sc.apply("quality_target", f"{ctx.chord_label(res_seg)} as {name}",
                     quality_factor(ctx, ctx.cond(res_seg, root, tq),
                                    ctx.cond(res_seg, root, is_class("dim", "hdim"))))
            prev = ctx.prev_run(j)
            predominant = 0.0
            if prev is not None:
                pr = ctx.runs[prev]
                predominant = max(ctx.best_mass(pr, t + 2, is_class("min", "hdim", "dim"))[0],
                                  ctx.best_mass(pr, t + 5, is_class("maj", "min"))[0])
            seventh = ctx.cond(v_seg, t + 7, lambda ch: 1.0 if ch.has_seventh else 0.0)
            if predominant >= 0.5:
                sc.apply("preparation", "V is prepared by a predominant (ii / IV)", cfg["prepared"])
            elif seventh >= 0.5:
                sc.apply("preparation", "unprepared, but V carries its 7th", cfg["seventh_only"])
            else:
                continue  # passing V→vi in a loop, not a cadence
            sc.apply("key", f"local key {key.label} (p={kp:.2f})", ctx.key_factor(kp))
            if sc.value < ctx.min_conf:
                continue
            ev = make_event(ctx, type_="deceptive_cadence", rule=RULE_DECEPTIVE,
                            label=f"V–{name} (deceptive cadence)", seg_indices=[v_seg, res_seg], score=sc,
                            key_label=key.label, attributes={"target": name, "borrowed_target": name == "♭VI"
                                                             and key.is_major})
            if best is None or ev.confidence > best.confidence:
                best = ev
        if best is not None:
            out.append(best)
    return out
