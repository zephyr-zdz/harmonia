"""ii–V–I (major) and ii–V–i / iiø7–V7–i (minor), secondary ii–Vs, and unresolved ii–Vs.

Theory: the ii–V–I is two successive root motions by ascending 4th (= descending 5th),
ending on a chord that acts as a (temporary) tonic. The prototype qualities are
iim7–V7–Imaj7 in major and iiø7–V7–im7 in minor; mixtures (iiø7–V7–I, iim7–V7–i) are common.

Evidence:
  * main  — root motion: P(root_ii) * P(root_V) * P(root_I) for the three successive runs.
  * bonus — quality of each chord (match raises, uninformative leaves base factor).
  * penalty — a quality that contradicts the function: a major/dominant "ii" (that is V/V,
    not ii) or a minor "V" (e.g. Am–Dm–G is vi–ii–V, not ii–V–I in G).
  * target — resolving to the local tonic is the textbook case; resolving to another
    chord is a tonicization (secondary ii–V), slightly less certain.

Unresolved ii–V (type "ii_V", user request 2026-10-06): the ii→V pair is a function on its
own (ツーファイブ) even when the expected I does not follow:
  deceptive  — next root a whole step above the target (ii–V–vi), or a half step above for a
               minor target (ii–V–♭VI). A deceptive ii–V to the LOCAL TONIC is left to the
               deceptive_cadence rule, which reports the same moment.
  unresolved — anything else;  end — the input ends after V.
"""

from __future__ import annotations

from ...schema import Event
from ...theory.key import Key
from ...theory.roman import applied_label, diatonic_triad_class, target_numeral
from ..context import AnalysisContext, Run
from .base import Score, is_class, make_event, quality_factor

RULE = "ii_V_I"

II_MATCH = is_class("min", "hdim", "dim")
II_CONTRA = is_class("maj", "dom", "aug")
II_MINOR = is_class("hdim", "dim")
V_MATCH = is_class("dom", "maj", "aug")
V_SUS = is_class("sus")
V_CONTRA = is_class("min", "dim", "hdim")
I_MATCH = is_class("maj", "min")
I_CONTRA = is_class("dim", "hdim")


def _score_ii_v(ctx: AnalysisContext, iirun: Run, vrun: Run, t: int) -> tuple[Score, int, int] | None:
    """Root + quality evidence for ii (root t+2) → V (root t+7). None if roots too unlikely."""
    cfg = ctx.cfg[RULE]
    p_ii = ctx.root_prob(iirun, t + 2)
    p_v = ctx.root_prob(vrun, t + 7)
    if min(p_ii, p_v) < ctx.min_root_prob:
        return None
    ii_seg = ctx.best_mass(iirun, t + 2, II_MATCH)[1]
    v_seg = ctx.best_mass(vrun, t + 7, V_MATCH)[1]
    sc = Score()
    sc.apply("root_motion", "root rises a 4th (ii → V)", p_ii * p_v)
    m, c = ctx.cond(ii_seg, t + 2, II_MATCH), ctx.cond(ii_seg, t + 2, II_CONTRA)
    sc.apply("quality_ii", f"ii chord {ctx.chord_label(ii_seg)}: P(minor/ø)={m:.2f}, P(major)={c:.2f}",
             quality_factor(ctx, m, c))
    m = ctx.cond(v_seg, t + 7, V_MATCH) + cfg["sus_v_match"] * ctx.cond(v_seg, t + 7, V_SUS)
    c = ctx.cond(v_seg, t + 7, V_CONTRA)
    sc.apply("quality_V", f"V chord {ctx.chord_label(v_seg)}: P(major 3rd)={m:.2f}, P(minor)={c:.2f}",
             quality_factor(ctx, m, c))
    return sc, ii_seg, v_seg


def _target_factor(ctx: AnalysisContext, sc: Score, key: Key, t: int) -> bool:
    cfg = ctx.cfg[RULE]
    if t == key.tonic:
        sc.apply("target", f"targets the local tonic of {key.label}", cfg["target_tonic"])
        return True
    if t in key.diatonic_pcs:
        sc.apply("target", f"tonicises a diatonic chord of {key.label}", cfg["target_diatonic"])
    else:
        sc.apply("target", f"tonicises a non-diatonic chord in {key.label}", cfg["target_other"])
    return False


def _resolved(ctx: AnalysisContext, iirun: Run, vrun: Run, irun: Run) -> Event | None:
    best: Event | None = None
    for t in ctx.roots(irun):
        p_i = ctx.root_prob(irun, t)
        if p_i < ctx.min_root_prob:
            continue
        r = _score_ii_v(ctx, iirun, vrun, t)
        if r is None:
            continue
        sc, ii_seg, v_seg = r
        i_seg = irun.segs[0]
        key, _ = ctx.key_of(i_seg)
        sc.apply("resolution", f"V resolves to {key.spell(t)}", p_i)
        m, c = ctx.cond(i_seg, t, I_MATCH), ctx.cond(i_seg, t, I_CONTRA)
        sc.apply("quality_I", f"target {ctx.chord_label(i_seg)}: P(stable triad)={m:.2f}", quality_factor(ctx, m, c))
        target_minor = ctx.cond(i_seg, t, is_class("min")) > ctx.cond(i_seg, t, is_class("maj", "dom"))
        tonic = _target_factor(ctx, sc, key, t)
        if sc.value < ctx.min_conf:
            continue
        mode = "minor" if target_minor else "major"
        segs = [ii_seg, v_seg, i_seg]
        label = "ii–V–I" if mode == "major" else "ii–V–i"
        tnum = target_numeral(key, t, "min" if target_minor else "maj")
        if tonic:
            functions, explains = [ctx.numeral(s) for s in segs], []
        else:
            label += f" → {tnum}"
            functions = [applied_label(ctx.segs[ii_seg].chord, t, mode, tnum),
                         applied_label(ctx.segs[v_seg].chord, t, mode, tnum), ctx.numeral(i_seg)]
            explains = [ii_seg, v_seg]
        ev = make_event(ctx, type_="ii_V_I", rule=RULE, label=label, seg_indices=segs, score=sc,
                        key_label=key.label, functions=functions, explains=explains,
                        attributes={"mode": mode, "target": tnum, "target_root": Key(t, mode).tonic_name,
                                    "tonicization": not tonic})
        if best is None or ev.confidence > best.confidence:
            best = ev
    return best


def _unresolved(ctx: AnalysisContext, iirun: Run, vrun: Run, irun: Run | None) -> Event | None:
    cfg = ctx.cfg[RULE]
    best: Event | None = None
    for r in ctx.roots(vrun):
        t = (r + 5) % 12
        res = _score_ii_v(ctx, iirun, vrun, t)
        if res is None:
            continue
        sc, ii_seg, v_seg = res
        key, _ = ctx.key_of(v_seg)
        # target quality: ø/° ii implies a minor target; otherwise the diatonic triad on that degree
        minor = ctx.cond(ii_seg, t + 2, II_MINOR) > 0.5 or diatonic_triad_class(key, key.degree(t)) == "min"
        mode = "minor" if minor else "major"
        if irun is None:
            status, factor, nxt_seg = "end", cfg["unresolved_other"], None
        else:
            if ctx.root_prob(irun, t) >= 0.5:
                continue  # resolves after all (reported, if at all, as ii_V_I)
            if ctx.root_prob(irun, t + 9) >= 0.5 and (t + 9) % 12 == key.tonic:
                continue  # iv–♭VII7–I backdoor progression: reported by the borrowed rule
                          # (♭VII7 with backdoor resolution), not as a failed ii–V of ♭III
            dec_root = t + (8 if minor else 9)
            p_dec = ctx.root_prob(irun, dec_root)
            nxt_seg = irun.segs[0]
            if p_dec * cfg["unresolved_deceptive"] >= cfg["unresolved_other"]:
                status, factor = "deceptive", cfg["unresolved_deceptive"] * p_dec
            else:
                status, factor = "unresolved", cfg["unresolved_other"]
        tonic = t == key.tonic
        if tonic and status == "deceptive":
            continue  # reported by the deceptive_cadence rule
        sc.apply("resolution", {"deceptive": "deceptive: lands a step above the expected I",
                                "unresolved": "the expected I does not follow",
                                "end": "input ends after V"}[status], factor)
        _target_factor(ctx, sc, key, t)
        if sc.value < ctx.min_conf:
            continue
        tnum = target_numeral(key, t, "min" if minor else "maj")
        label = ("ii–V" if tonic else f"ii–V/{tnum}") + f" ({status})"
        segs = [ii_seg, v_seg] + ([nxt_seg] if nxt_seg is not None and status == "deceptive" else [])
        if tonic:
            functions, explains = [ctx.numeral(s) for s in segs], []
        else:
            functions = [applied_label(ctx.segs[ii_seg].chord, t, mode, tnum),
                         applied_label(ctx.segs[v_seg].chord, t, mode, tnum)] + \
                        ([ctx.numeral(nxt_seg)] if len(segs) == 3 else [])
            explains = [ii_seg, v_seg]
        ev = make_event(ctx, type_="ii_V", rule=RULE, label=label, seg_indices=segs, score=sc,
                        key_label=key.label, functions=functions, explains=explains,
                        attributes={"mode": mode, "target": tnum, "target_root": Key(t, mode).tonic_name,
                                    "tonicization": not tonic, "resolution": status})
        if best is None or ev.confidence > best.confidence:
            best = ev
    return best


def detect(ctx: AnalysisContext) -> list[Event]:
    out: list[Event] = []
    for j, vrun in enumerate(ctx.runs):
        if vrun.root is None:
            continue
        i, k = ctx.prev_run(j), ctx.next_run(j)
        if i is None:
            continue
        iirun = ctx.runs[i]
        irun = ctx.runs[k] if k is not None else None
        ev = _resolved(ctx, iirun, vrun, irun) if irun is not None else None
        if ev is None and ctx.cfg[RULE]["report_unresolved"]:
            ev = _unresolved(ctx, iirun, vrun, irun)
        if ev is not None:
            out.append(ev)
    return out
