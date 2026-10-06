"""ii–V–I (major) and ii–V–i / iiø7–V7–i (minor), including secondary ii–Vs.

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
"""

from __future__ import annotations

from ...schema import Event
from ...theory.key import Key
from ...theory.roman import applied_label, target_numeral
from ..context import AnalysisContext
from .base import Score, is_class, make_event, quality_factor

RULE = "ii_V_I"

II_MATCH = is_class("min", "hdim", "dim")
II_CONTRA = is_class("maj", "dom", "aug")
V_MATCH = is_class("dom", "maj", "aug")
V_SUS = is_class("sus")
V_CONTRA = is_class("min", "dim", "hdim")
I_MATCH = is_class("maj", "min")
I_CONTRA = is_class("dim", "hdim")


def detect(ctx: AnalysisContext) -> list[Event]:
    cfg = ctx.cfg[RULE]
    out: list[Event] = []
    for j, vrun in enumerate(ctx.runs):
        if vrun.root is None:
            continue
        i, k = ctx.prev_run(j), ctx.next_run(j)
        if i is None or k is None:
            continue
        iirun, irun = ctx.runs[i], ctx.runs[k]
        best: Event | None = None
        for t in ctx.roots(irun):
            p_ii = ctx.root_prob(iirun, t + 2)
            p_v = ctx.root_prob(vrun, t + 7)
            p_i = ctx.root_prob(irun, t)
            if min(p_ii, p_v, p_i) < ctx.min_root_prob:
                continue
            ii_seg = ctx.best_mass(iirun, t + 2, II_MATCH)[1]
            v_seg = ctx.best_mass(vrun, t + 7, V_MATCH)[1]
            i_seg = irun.segs[0]
            key, _ = ctx.key_of(i_seg)

            sc = Score()
            sc.apply("root_motion", "roots rise by a 4th twice (ii → V → I)", p_ii * p_v * p_i)
            m, c = ctx.cond(ii_seg, t + 2, II_MATCH), ctx.cond(ii_seg, t + 2, II_CONTRA)
            sc.apply("quality_ii", f"ii chord {ctx.chord_label(ii_seg)}: P(minor/ø)={m:.2f}, P(major)={c:.2f}",
                     quality_factor(ctx, m, c))
            m = ctx.cond(v_seg, t + 7, V_MATCH) + cfg["sus_v_match"] * ctx.cond(v_seg, t + 7, V_SUS)
            c = ctx.cond(v_seg, t + 7, V_CONTRA)
            sc.apply("quality_V", f"V chord {ctx.chord_label(v_seg)}: P(major 3rd)={m:.2f}, P(minor)={c:.2f}",
                     quality_factor(ctx, m, c))
            m, c = ctx.cond(i_seg, t, I_MATCH), ctx.cond(i_seg, t, I_CONTRA)
            sc.apply("quality_I", f"target {ctx.chord_label(i_seg)}: P(stable triad)={m:.2f}",
                     quality_factor(ctx, m, c))

            target_minor = ctx.cond(i_seg, t, is_class("min")) > ctx.cond(i_seg, t, is_class("maj", "dom"))
            mode = "minor" if target_minor else "major"
            tonic = t == key.tonic
            if tonic:
                sc.apply("target", f"resolves to the local tonic of {key.label}", cfg["target_tonic"])
            elif t in key.diatonic_pcs:
                sc.apply("target", f"tonicises a diatonic chord of {key.label}", cfg["target_diatonic"])
            else:
                sc.apply("target", f"tonicises a non-diatonic chord in {key.label}", cfg["target_other"])
            if sc.value < ctx.min_conf:
                continue

            segs = [ii_seg, v_seg, i_seg]
            label = "ii–V–I" if mode == "major" else "ii–V–i"
            tnum = target_numeral(key, t, "min" if target_minor else "maj")
            if tonic:
                functions = [ctx.numeral(s) for s in segs]
                explains: list[int] = []
            else:
                label += f" → {tnum}"
                tkey_mode = "minor" if target_minor else "major"
                functions = [applied_label(ctx.segs[ii_seg].chord, t, tkey_mode, tnum),
                             applied_label(ctx.segs[v_seg].chord, t, tkey_mode, tnum),
                             ctx.numeral(i_seg)]
                explains = [ii_seg, v_seg]
            ev = make_event(ctx, type_="ii_V_I", rule=RULE, label=label, seg_indices=segs, score=sc,
                            key_label=key.label, functions=functions, explains=explains,
                            attributes={"mode": mode, "target": tnum, "target_root": Key(t, mode).tonic_name,
                                        "tonicization": not tonic})
            if best is None or ev.confidence > best.confidence:
                best = ev
        if best is not None:
            out.append(best)
    return out
