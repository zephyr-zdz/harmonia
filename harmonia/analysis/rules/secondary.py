"""Secondary (applied) dominants V/x, V7/x, and secondary leading-tone chords vii°(7)/x.

Theory — secondary dominant: a major-third chord (optionally with a minor 7th) whose root
lies a 5th above a non-tonic target, borrowing the target's leading tone. What separates
V/x from diatonic fifth motion (I→IV, iii→vi, vi→ii) is precisely that the chord contains a
tone OUTSIDE the key — its major 3rd (E→Am in C: G#) or its minor 7th (C7→F in C: Bb).
So the chord must be "chromatically dominant" in the local key; how strongly it reads as a
dominant (7th chord > triad > 7sus4) is the soft quality evidence, and the root motion to
the target ("resolution") is the main contextual evidence:
    resolved   — next root a 5th below (E7 → Am)
    deceptive  — next root a step above the target's root, by analogy with V→vi / V→VI
                 (E7 → F in C, the J-pop "III7 → IV")
    unresolved — anything else (lower confidence; may lose to a borrowed-chord reading).
The primary dominant (root on scale degree 5) is never reported here.

Theory — secondary leading-tone chord: a diminished (or half-diminished) chord whose
chromatic root lies a semitone below a non-tonic target, e.g. C#°7 → Dm in C (vii°7/ii).
"""

from __future__ import annotations

from ...schema import Event
from ...theory.chord import Chord
from ...theory.key import Key
from ...theory.roman import applied_label, diatonic_triad_class, target_numeral
from ..context import AnalysisContext
from .base import Score, make_event

RULE_DOM = "secondary_dominant"
RULE_LT = "secondary_leading_tone"


def dominant_weight(chord: Chord, key: Key, cfg: dict) -> float:
    q = chord.quality
    weights = {"7": cfg["weight_dom7"], "aug7": cfg["weight_dom7"], "maj": cfg["weight_major_triad"],
               "maj6": cfg["weight_maj6"], "aug": cfg["weight_aug"], "7sus4": cfg["weight_7sus4"]}
    if q not in weights or chord.root is None:
        return 0.0
    tones = []
    if q != "7sus4":
        tones.append(chord.root + 4)       # major third = leading tone of the target
    if q in ("7", "aug7", "7sus4"):
        tones.append(chord.root + 10)      # minor seventh
    if q in ("aug", "aug7"):
        tones.append(chord.root + 8)
    if all(t % 12 in key.diatonic_pcs for t in tones):
        return 0.0                          # fully diatonic: ordinary fifth motion
    return weights[q]


def _target_mode(key: Key, target_root: int) -> str:
    qc = diatonic_triad_class(key, key.degree(target_root))
    return "minor" if qc == "min" else "major"


def detect_dominants(ctx: AnalysisContext) -> list[Event]:
    cfg = ctx.cfg[RULE_DOM]
    out: list[Event] = []
    for j, run in enumerate(ctx.runs):
        if run.root is None:
            continue
        k = ctx.next_run(j)
        best: Event | None = None
        for r in ctx.roots(run):
            key, kp = ctx.key_of(run.segs[-1])
            if key.degree(r) == 7:
                continue  # primary dominant
            target = (r + 5) % 12
            if diatonic_triad_class(key, key.degree(target)) == "dim":
                continue  # diminished triads are not tonicised
            ev_val, seg = ctx.best_mass(run, r, lambda ch, key=key: dominant_weight(ch, key, cfg))
            if ev_val <= 0:
                continue
            tmode = _target_mode(key, target)
            sc = Score()
            sc.apply("dominant_quality",
                     f"{ctx.chord_label(seg)} has a tone outside {key.label} acting as leading tone / b7 of "
                     f"{key.spell(target)}", ev_val)

            options = [("unresolved", cfg["unresolved"], None)]
            if k is not None:
                nxt = ctx.runs[k]
                dec_root = (r + (1 if tmode == "minor" else 2)) % 12
                options.append(("resolved", cfg["resolved"] * ctx.root_prob(nxt, target), nxt.segs[0]))
                options.append(("deceptive", cfg["deceptive"] * ctx.root_prob(nxt, dec_root), nxt.segs[0]))
            status, factor, res_seg = max(options, key=lambda o: o[1])
            if status == "unresolved" and target not in key.diatonic_pcs and not cfg["unresolved_nondiatonic"]:
                # V/♭VII, V/♭III ... that never reach their target: the implied chord is itself
                # chromatic and absent, so the reading is speculative (e.g. blues IV7 in rock
                # is not "V7/♭VII"). Left to plain numerals / the borrowed-chord rule.
                continue
            detail = {"resolved": f"resolves down a 5th to {key.spell(target)}",
                      "deceptive": "deceptive resolution (a step above the expected target)",
                      "unresolved": "does not resolve to its target"}[status]
            sc.apply("resolution", detail, factor)
            sc.apply("key", f"local key {key.label} (p={kp:.2f})", ctx.key_factor(kp))
            if sc.value < ctx.min_conf:
                continue

            target_q = None
            if status == "resolved" and res_seg is not None:
                target_q = ctx.segs[res_seg].chord.qclass
            tnum = target_numeral(key, target, target_q)
            func = applied_label(ctx.segs[seg].chord, target, tmode, tnum)
            label = func + ("" if status == "resolved" else f" ({status})")
            segs = [seg] + ([res_seg] if res_seg is not None and status != "unresolved" else [])
            functions = [func] + ([ctx.numeral(res_seg)] if len(segs) > 1 else [])
            ev = make_event(ctx, type_="secondary_dominant", rule=RULE_DOM, label=label, seg_indices=segs,
                            score=sc, key_label=key.label, functions=functions, explains=[seg],
                            attributes={"target": tnum, "target_root": key.spell(target), "resolution": status,
                                        "has_seventh": ctx.segs[seg].chord.has_seventh})
            if best is None or ev.confidence > best.confidence:
                best = ev
        if best is not None:
            out.append(best)
    return out


def _lt_weight(chord: Chord, cfg: dict) -> float:
    return {"dim7": cfg["weight_dim7"], "dim": cfg["weight_dim"], "hdim7": cfg["weight_hdim7"]}.get(chord.quality, 0.0)


def detect_leading_tone(ctx: AnalysisContext) -> list[Event]:
    cfg = ctx.cfg[RULE_LT]
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
            key, kp = ctx.key_of(run.segs[-1])
            target = (r + 1) % 12
            if r in key.diatonic_pcs:
                continue  # diatonic vii° / viiø7 (or ii° in minor): not a secondary function
            if diatonic_triad_class(key, key.degree(target)) == "dim":
                continue
            ev_val, seg = ctx.best_mass(run, r, lambda ch: _lt_weight(ch, cfg))
            p_t = ctx.root_prob(nxt, target)
            if ev_val * p_t <= 0:
                continue
            sc = Score()
            sc.apply("diminished_quality", f"{ctx.chord_label(seg)} is diminished on a chromatic root", ev_val)
            sc.apply("resolution", f"rises a semitone to {key.spell(target)}", p_t)
            sc.apply("key", f"local key {key.label} (p={kp:.2f})", ctx.key_factor(kp))
            if sc.value < ctx.min_conf:
                continue
            res_seg = nxt.segs[0]
            tq = ctx.segs[res_seg].chord.qclass
            tnum = target_numeral(key, target, tq)
            tmode = "minor" if tq == "min" else "major"
            func = applied_label(ctx.segs[seg].chord, target, tmode, tnum)
            ev = make_event(ctx, type_="secondary_leading_tone", rule=RULE_LT, label=func,
                            seg_indices=[seg, res_seg], score=sc, key_label=key.label,
                            functions=[func, ctx.numeral(res_seg)], explains=[seg],
                            attributes={"target": tnum, "target_root": key.spell(target)})
            if best is None or ev.confidence > best.confidence:
                best = ev
        if best is not None:
            out.append(best)
    return out
