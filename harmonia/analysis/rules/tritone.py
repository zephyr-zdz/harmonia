"""Tritone substitution: subV7 → target (e.g. ♭II7 → I, D♭7 → C).

Theory: a dominant seventh a tritone away from V7 shares its tritone (3rd and ♭7 swap
roles), so it resolves DOWN A SEMITONE to the same target. Evidence:
  * main — root motion down a semitone into the target.
  * quality — the ♭7 is what makes the substitution (shared tritone); a plain major triad
    on ♭II is better read as a borrowed Neapolitan/Phrygian ♭II, so its weight is low.
  * a preceding ii of the target (ii–subV–I) is recorded as an attribute.
"""

from __future__ import annotations

from ...schema import Event
from ...theory.chord import Chord
from ...theory.roman import diatonic_triad_class, target_numeral
from ..context import AnalysisContext
from .base import Score, is_class, make_event

RULE = "tritone_sub"


def _weight(chord: Chord, cfg: dict) -> float:
    return {"7": cfg["weight_dom7"], "aug7": cfg["weight_dom7"], "maj": cfg["weight_major_triad"],
            "7sus4": cfg["weight_7sus4"]}.get(chord.quality, 0.0)


def detect(ctx: AnalysisContext) -> list[Event]:
    cfg = ctx.cfg[RULE]
    out: list[Event] = []
    for j, run in enumerate(ctx.runs):
        if run.root is None:
            continue
        k = ctx.next_run(j)
        if k is None:
            continue
        nxt = ctx.runs[k]
        best: Event | None = None
        for r in ctx.roots(run):
            t = (r - 1) % 12
            ev_val, seg = ctx.best_mass(run, r, lambda ch: _weight(ch, cfg))
            p_t = ctx.root_prob(nxt, t)
            if ev_val * p_t <= 0:
                continue
            res_seg = nxt.segs[0]
            key, _ = ctx.key_of(res_seg)
            sc = Score()
            sc.apply("dominant_quality", f"{ctx.chord_label(seg)}: dominant-7th quality (shared tritone)", ev_val)
            sc.apply("root_motion", f"root falls a semitone to {key.spell(t)}", p_t)
            deg_qc = diatonic_triad_class(key, key.degree(t))
            if t == key.tonic:
                sc.apply("target", f"resolves to the local tonic of {key.label}", cfg["target_tonic"])
            elif deg_qc is not None and deg_qc != "dim":
                sc.apply("target", f"targets a diatonic chord of {key.label}", cfg["target_diatonic"])
            else:
                sc.apply("target", f"targets a non-diatonic chord in {key.label}", cfg["target_other"])
            prev = ctx.prev_run(j)
            preceded_by_ii = False
            if prev is not None:
                ii_val, _ = ctx.best_mass(ctx.runs[prev], t + 2, is_class("min", "hdim"))
                preceded_by_ii = ii_val >= 0.5
                if preceded_by_ii:
                    sc.note("context", "preceded by ii of the target (ii–subV–I)")
            if sc.value < ctx.min_conf:
                continue
            tnum = target_numeral(key, t, ctx.segs[res_seg].chord.qclass)
            suffix = "7" if ctx.segs[seg].chord.has_seventh else ""
            func = f"subV{suffix}" if t == key.tonic else f"subV{suffix}/{tnum}"
            ev = make_event(ctx, type_="tritone_sub", rule=RULE, label=f"{func} → {tnum}",
                            seg_indices=[seg, res_seg], score=sc, key_label=key.label,
                            functions=[func, ctx.numeral(res_seg)], explains=[seg],
                            attributes={"target": tnum, "target_root": key.spell(t),
                                        "preceded_by_ii": preceded_by_ii})
            if best is None or ev.confidence > best.confidence:
                best = ev
        if best is not None:
            out.append(best)
    return out
