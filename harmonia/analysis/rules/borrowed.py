"""Borrowed chords (modal interchange), driven by the [[borrowed.major]] / [[borrowed.minor]]
tables in the config. See the table comments for the theory of each entry.

Evidence:
  * quality_required entries (root diatonic, borrowed tone = 3rd/5th/7th, e.g. iv):
        P(root and matching quality)
  * root-chromatic entries (e.g. ♭VI): P(root) * (quality_base + (1-quality_base) * P(match | root)
        - contradict_penalty * P(opposite third or diminished | root))
  * only candidates containing at least one tone outside the key count
  * multiplied by the local-key confidence factor (borrowing is defined relative to a key)
A borrowed reading loses to a resolved functional reading of the same chord (see resolve.py).
"""

from __future__ import annotations

from ...schema import Event
from ...theory.chord import Chord
from ...theory.key import Key
from ..context import AnalysisContext
from .base import Score, make_event

RULE = "borrowed_chord"


def _chromatic(ch: Chord, key: Key) -> bool:
    return not ch.core_pcs() <= key.diatonic_pcs


def detect(ctx: AnalysisContext) -> list[Event]:
    cfg = ctx.cfg["borrowed"]
    qbase = cfg["quality_base"]
    out: list[Event] = []
    for j, run in enumerate(ctx.runs):
        if run.root is None:
            continue
        key, kp = ctx.key_of(run.segs[-1])
        best: Event | None = None
        for entry in cfg.get(key.mode, []):
            r = (key.tonic + entry["degree"]) % 12
            if ctx.root_prob(run, r) < ctx.min_root_prob:
                continue
            qcs = set(entry["qclasses"])

            def match(ch: Chord, qcs=qcs, key=key) -> float:
                return 1.0 if ch.qclass in qcs and _chromatic(ch, key) else 0.0

            sc = Score()
            if entry["quality_required"]:
                val, seg = ctx.best_mass(run, r, match)
                sc.apply("quality", f"{ctx.chord_label(seg)} has the borrowed quality of {entry['numeral']}", val)
            else:
                # opposite third / diminished quality contradicts the entry
                contra_q = {"min", "dim", "hdim"} if qcs <= {"maj", "dom"} else {"maj", "dom"}

                def contra(ch: Chord, contra_q=contra_q) -> float:
                    return 1.0 if ch.qclass in contra_q else 0.0

                val, seg = -1.0, run.segs[-1]
                for s in run.segs:
                    dist = ctx.segs[s].dist
                    chrom = dist.mass(r, lambda ch, key=key: 1.0 if _chromatic(ch, key) else 0.0)
                    qf = qbase + (1 - qbase) * dist.cond(r, match) - ctx.cfg["quality"]["contradict_penalty"] * dist.cond(r, contra)
                    v = chrom * max(qf, 0.0)
                    if v > val:
                        val, seg = v, s
                sc.apply("root", f"chromatic root {key.spell(r)} = {entry['numeral'].rstrip('7')} of {key.label}; "
                                 f"quality {'matches' if ctx.cond(seg, r, match) > 0.5 else 'does not match'}", val)
            sc.apply("key", f"local key {key.label} (p={kp:.2f})", ctx.key_factor(kp))
            sc.note("source", f"borrowed from {entry['source']}")
            attrs = {"source": entry["source"], "entry": entry["numeral"], "theory_note": entry.get("note", "")}
            if entry["degree"] == 10:
                k = ctx.next_run(j)
                if k is not None and ctx.root_prob(ctx.runs[k], key.tonic) >= 0.5:
                    sc.note("context", "resolves to I (backdoor resolution)")
                    attrs["backdoor"] = True
            if sc.value < ctx.min_conf:
                continue
            ev = make_event(ctx, type_="borrowed_chord", rule=RULE,
                            label=f"{ctx.numeral(seg)} (borrowed: {entry['source']})",
                            seg_indices=[seg], score=sc, key_label=key.label, explains=[seg],
                            attributes=attrs)
            if best is None or ev.confidence > best.confidence:
                best = ev
        if best is not None:
            out.append(best)
    return out
