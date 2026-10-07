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

import math

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


def _opts(x) -> list[int]:
    return list(x) if isinstance(x, (list, tuple)) else [x]


def match_idioms(ctx: AnalysisContext) -> list[IdiomMatch]:
    """Match every configured idiom in all transpositions.

    Per position (see default_config.toml [[named_progressions.idiom]]):
      degrees[i]   root degree in the reference major key, or a list of alternatives;
      qclasses[i]  allowed quality classes (soft quality factor), or "any";
      bass[i]      optional: bass degree(s) that must sound (soft: P(bass) is a factor), or -1 = free.
    Confidence = geometric mean of the per-position factors (each ≥ [named_progressions]
    min_position_factor, so one clearly missing chord still breaks the match).
    ``min_length``: the first n positions are the idiom's distinctive core; a match may stop
    after them (the rest is reported when present). Overlapping matches of the same name keep
    the longest / most confident one.
    """
    cfg = ctx.cfg["named_progressions"]
    q = ctx.cfg["quality"]
    out: list[IdiomMatch] = []
    for idiom in cfg.get("idiom", []):
        degrees = [_opts(d) for d in idiom["degrees"]]
        n = len(degrees)
        quals = idiom.get("qclasses", "any")
        allowed = [None] * n if quals == "any" else [None if a == "any" else set(a) for a in quals]
        bass = [None if b == -1 else _opts(b) for b in idiom.get("bass", [-1] * n)]
        if not (len(allowed) == len(bass) == n):
            raise ValueError(f"idiom {idiom['name']!r}: degrees / qclasses / bass lengths differ")
        min_len = int(idiom.get("min_length", n))
        floor = cfg["min_position_factor"]
        for start, run0 in enumerate(ctx.runs):
            if run0.root is None:
                continue
            for first in degrees[0]:
                tonic = (run0.root - first) % 12
                factors, reps, runs, used = [], [], [], []
                ri: int | None = start
                for i in range(n):
                    if ri is None:
                        break
                    run = ctx.runs[ri]
                    opts = [first] if i == 0 else degrees[i]
                    deg = max(opts, key=lambda d: ctx.root_prob(run, tonic + d))
                    pc = (tonic + deg) % 12
                    p_root = min(1.0, sum(ctx.root_prob(run, tonic + d) for d in opts))
                    if bass[i] is None:
                        f = p_root
                    else:  # bass-defined position: the bass is the evidence, the root a bonus
                        f = ctx.bass_prob(run, {tonic + b for b in bass[i]}) * (q["base"] + (1 - q["base"]) * p_root)
                    ok = allowed[i]
                    if ok is None:
                        seg = ctx.best_mass(run, pc, lambda ch: 1.0)[1]
                    else:
                        seg = ctx.best_mass(run, pc, lambda ch, ok=ok: 1.0 if ch.qclass in ok else 0.0)[1]
                        m = ctx.cond(seg, pc, lambda ch, ok=ok: 1.0 if ch.qclass in ok else 0.0)
                        c = ctx.cond(seg, pc, lambda ch, ok=ok: 1.0 if ch.qclass in _DEFINITE - ok else 0.0)
                        f *= max(q["base"] + (1 - q["base"]) * m - q["contradict_penalty"] * c, 0.0)
                    if f < floor:
                        break
                    factors.append(f)
                    reps.append(seg)
                    runs.append(ri)
                    used.append(deg)
                    ri = ctx.next_run(ri)
                if len(reps) < min_len:
                    continue
                # geometric mean: "how well does each chord fit", independent of idiom length
                conf = math.exp(sum(math.log(f) for f in factors) / len(factors))
                if conf < ctx.min_conf:
                    continue
                all_segs = [s for r in runs for s in ctx.runs[r].segs]
                out.append(IdiomMatch(idiom["name"], idiom.get("alias", ""), Key(tonic, "major"), reps,
                                      all_segs, round(conf, 4), bool(idiom.get("use_key_prior", False)),
                                      tuple(used)))
    return _dedupe(out)


def _dedupe(matches: list[IdiomMatch]) -> list[IdiomMatch]:
    """Same-name matches over overlapping chords: keep the longest, then most confident."""
    kept: list[IdiomMatch] = []
    for m in sorted(matches, key=lambda m: (-len(m.seg_indices), -m.confidence, m.seg_indices[0])):
        if any(k.name == m.name and set(k.seg_indices) & set(m.seg_indices) for k in kept):
            continue
        kept.append(m)
    return sorted(kept, key=lambda m: (m.seg_indices[0], m.name))


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
