"""Named progressions (王道進行, 丸サ進行, 小室進行, カノン進行 ...), matched key-independently.

Each idiom in config ``[[named_progressions.idiom]]`` is a sequence of (scale degree,
allowed quality classes) defined relative to a REFERENCE MAJOR KEY — the key in which the
progression is conventionally named (e.g. 丸サ進行 = IVmaj7–III7–vim7–vm7–I7). Matching runs
before key estimation over all 12 transpositions, on root runs, with soft evidence.

Optionally (``use_key_prior = true``) a match adds ``key_prior * confidence`` to the HMM
emission of its reference key over the matched segments, so that the conventional naming
wins where the idiom itself is tonally ambiguous. Enabled only for idioms where the user
chose the reference explicitly (see CLAUDE.md, decisions log).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..theory.key import Key
from .context import AnalysisContext

_DEFINITE = {"maj", "min", "dom", "dim", "hdim", "aug"}


@dataclass
class IdiomMatch:
    name: str
    alias: str
    ref_key: Key
    seg_indices: list[int]   # one representative segment per idiom chord
    all_segs: list[int]      # every segment covered
    confidence: float
    use_key_prior: bool
    degrees: tuple[int, ...] = ()   # idiom degrees relative to ref_key


def match_idioms(ctx: AnalysisContext) -> list[IdiomMatch]:
    cfg = ctx.cfg["named_progressions"]
    q = ctx.cfg["quality"]
    out: list[IdiomMatch] = []
    for idiom in cfg.get("idiom", []):
        degrees = idiom["degrees"]
        allowed = [set(a) for a in idiom["qclasses"]]
        for start, run0 in enumerate(ctx.runs):
            if run0.root is None:
                continue
            tonic = (run0.root - degrees[0]) % 12
            runs = [start]
            while len(runs) < len(degrees):
                nxt = ctx.next_run(runs[-1])
                if nxt is None:
                    break
                runs.append(nxt)
            if len(runs) < len(degrees):
                continue
            conf = 1.0
            reps: list[int] = []
            for ri, deg, ok in zip(runs, degrees, allowed):
                run = ctx.runs[ri]
                pc = (tonic + deg) % 12
                conf *= ctx.root_prob(run, pc)
                val, seg = ctx.best_mass(run, pc, lambda ch, ok=ok: 1.0 if ch.qclass in ok else 0.0)
                m = ctx.cond(seg, pc, lambda ch, ok=ok: 1.0 if ch.qclass in ok else 0.0)
                c = ctx.cond(seg, pc, lambda ch, ok=ok: 1.0 if ch.qclass in _DEFINITE - ok else 0.0)
                conf *= max(q["base"] + (1 - q["base"]) * m - q["contradict_penalty"] * c, 0.0)
                reps.append(seg)
                if conf < ctx.min_conf:
                    break
            if conf < ctx.min_conf:
                continue
            all_segs = [s for ri in runs for s in ctx.runs[ri].segs]
            out.append(IdiomMatch(idiom["name"], idiom.get("alias", ""), Key(tonic, "major"), reps,
                                  all_segs, round(conf, 4), bool(idiom.get("use_key_prior", False)),
                                  tuple(degrees)))
    return out


def loop_segments(ctx: AnalysisContext, matches: list[IdiomMatch], min_root_prob: float = 0.5) -> frozenset[int]:
    """Segments that belong to a named progression, including a loop restart right after it
    (its first chord again) or a pickup right before it (its last chord). Such chords are
    defined by the idiom (王道 opens on IV, 丸サ loops back to IVmaj7), so the key model must not
    read them as "the song starts / ends on its tonic"."""
    run_of = {s: r.index for r in ctx.runs for s in r.segs}
    out: set[int] = set()
    for m in matches:
        out.update(m.all_segs)
        if not m.degrees:
            continue
        first_pc = (m.ref_key.tonic + m.degrees[0]) % 12
        last_pc = (m.ref_key.tonic + m.degrees[-1]) % 12
        for ri, pc in ((ctx.next_run(run_of[max(m.all_segs)]), first_pc),
                       (ctx.prev_run(run_of[min(m.all_segs)]), last_pc)):
            if ri is not None and ctx.root_prob(ctx.runs[ri], pc) >= min_root_prob:
                out.update(ctx.runs[ri].segs)
    return frozenset(out)


def key_prior_bonus(matches: list[IdiomMatch], n_segs: int, cfg: dict) -> list[dict[Key, float]]:
    """Per-segment additive emission bonus {key: bonus} from idioms with use_key_prior."""
    bonus: list[dict[Key, float]] = [dict() for _ in range(n_segs)]
    w = cfg["named_progressions"]["key_prior"]
    for m in matches:
        if not m.use_key_prior:
            continue
        for s in m.all_segs:
            bonus[s][m.ref_key] = max(bonus[s].get(m.ref_key, 0.0), w * m.confidence)
    return bonus
