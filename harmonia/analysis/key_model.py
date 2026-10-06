"""Local-key estimation: a chord-level HMM over the 24 major/minor keys.

Emission  = how idiomatically each chord functions in a key (hand-written table of
            log-preferences below, MAJOR_FIT / MINOR_FIT) + a cadence bonus when V(7)
            resolves to the key's tonic. Soft candidates are mixed: log sum_c p_c exp(s_c).
Transition = stay (0) or switch (-switch_penalty) to any other key.
Decoding  = Viterbi path, then regions shorter than min_region_beats are merged into a
            neighbour (short excursions are tonicizations, not modulations).
Confidence = forward-backward posterior of the chosen key at each segment.

NOTE: the posteriors are NOT calibrated yet; calibrate on data/eval/dev in Phase 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..theory.chord import ADDED_TONE, Chord
from ..theory.key import ALL_KEYS, KEY_INDEX, Key
from .segments import SegData

UNLISTED = -4.0  # chord types not listed for a degree: very unidiomatic in that key

# Keys: scale degree (semitones above tonic) -> {quality class: log-preference}.
# 0 = ordinary diatonic chord. Positive = extra evidence for this key (tonic / dominant).
# Negative = idiomatic but chromatic (secondary dominants, borrowed chords).
MAJOR_FIT: dict[int, dict[str, float]] = {
    0: {"maj": 0.7, "dom": -1.2, "min": -2.0, "sus": -0.3, "aug": -1.5},   # I (tonic); I7 = V7/IV; i borrowed
    1: {"maj": -2.5, "dom": -2.0, "dim": -1.5},                             # bII Neapolitan; bII7 subV; #i°7 passing
    2: {"min": 0.0, "hdim": -1.5, "dim": -2.0, "maj": -1.2, "dom": -1.0, "sus": -0.5},  # ii; iiø7 borrowed; II(7) = V(7)/V
    3: {"maj": -1.5, "dom": -2.5, "dim": -1.5},                             # bIII borrowed; #ii°7 passing
    4: {"min": -0.2, "maj": -1.2, "dom": -1.0, "sus": -1.0},                # iii; III(7) = V(7)/vi
    5: {"maj": 0.0, "min": -1.2, "dom": -1.5, "sus": -0.5},                 # IV; iv borrowed; IV7 (blues)
    6: {"hdim": -1.8, "dim": -1.8, "dom": -2.5},                            # #ivø7 / #iv°7
    7: {"maj": 0.3, "dom": 0.5, "min": -1.5, "sus": 0.0, "aug": -1.2},      # V (dominant); v borrowed
    8: {"maj": -1.2, "dom": -2.0},                                          # bVI borrowed; bVI7
    9: {"min": 0.0, "maj": -1.5, "dom": -1.2, "sus": -0.8},                 # vi; VI(7) = V(7)/ii
    10: {"maj": -1.2, "dom": -1.5},                                         # bVII borrowed; bVII7 backdoor
    11: {"dim": -0.8, "hdim": -0.5, "maj": -2.0, "dom": -1.5},              # vii° / viiø7; VII(7) = V(7)/iii
}
MINOR_FIT: dict[int, dict[str, float]] = {
    0: {"min": 0.7, "maj": -1.5, "dom": -1.5, "sus": -0.3},                 # i (tonic); I Picardy; I7 = V7/iv
    1: {"maj": -1.5, "dom": -2.0},                                          # bII Neapolitan; subV
    2: {"hdim": -0.3, "dim": -0.5, "min": -1.2, "maj": -1.5, "dom": -1.0, "sus": -0.8},  # iiø7; ii Dorian; V/V
    3: {"maj": 0.0, "aug": -1.0, "dom": -1.2},                              # bIII; bIII+; V7/bVI
    5: {"min": 0.0, "maj": -1.0, "dom": -1.2, "sus": -0.5},                 # iv; IV(7) Dorian
    7: {"min": -0.2, "maj": 0.3, "dom": 0.5, "sus": 0.0},                   # v natural; V / V7 harmonic (dominant)
    8: {"maj": 0.0, "dom": -1.5},                                           # bVI
    9: {"hdim": -1.2, "dim": -1.5},                                         # viø7 (melodic minor)
    10: {"maj": 0.0, "dom": -0.3},                                          # bVII; bVII7 = V7/bIII
    11: {"dim": -0.5},                                                      # vii°7 (harmonic)
}


def chord_key_score(chord: Chord, key: Key, cfg: dict) -> float:
    if chord.root is None:
        return 0.0
    deg = key.degree(chord.root)
    row = (MAJOR_FIT if key.is_major else MINOR_FIT).get(deg, {})
    qc = chord.qclass
    if qc == "pow":
        s = max(row.get("maj", UNLISTED), row.get("min", UNLISTED))
    else:
        s = row.get(qc, UNLISTED)
    added = ADDED_TONE.get(chord.quality)
    if added is not None and (chord.root + added) % 12 not in key.diatonic_pcs:
        s -= cfg["key"]["seventh_off_scale_penalty"]
    return s


@dataclass
class KeyAnalysis:
    seg_keys: list[Key]
    seg_probs: list[float]
    posteriors: list[list[float]]
    emissions: list[list[float]]
    regions: list[tuple[int, int, Key]]       # (first_seg, last_seg_exclusive, key)
    global_key: Key
    global_dist: list[tuple[Key, float]]      # sorted, top first
    ambiguous: bool
    source: str                               # "estimated" | "given"


def _lse(xs: list[float]) -> float:
    m = max(xs)
    if m == -math.inf:
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _weight(seg: SegData, kc: dict) -> float:
    w = seg.beats / kc["beats_per_weight_unit"]
    return min(max(w, kc["weight_min"]), kc["weight_max"])


def emission_matrix(segs: list[SegData], cfg: dict) -> list[list[float]]:
    kc = cfg["key"]
    out: list[list[float]] = []
    prev: SegData | None = None
    for seg in segs:
        cands = [c for c in seg.dist.cands if c.chord.root is not None and c.prob > 0]
        row = [0.0] * len(ALL_KEYS)
        if cands:
            total = sum(c.prob for c in seg.dist.cands if c.prob > 0) or 1.0
            no_root = sum(c.prob for c in seg.dist.cands if c.chord.root is None) / total
            w = _weight(seg, kc)
            for ki, key in enumerate(ALL_KEYS):
                terms = [math.log(c.prob / total) + chord_key_score(c.chord, key, cfg) for c in cands]
                if no_root > 0:
                    terms.append(math.log(no_root))
                e = _lse(terms)
                if not key.is_major:
                    e += kc["minor_bias"]
                row[ki] = w * e
                # cadence: V / V7 of this key immediately followed by its tonic triad
                if prev is not None:
                    dom = prev.dist.mass((key.tonic + 7) % 12, lambda ch: 1.0 if ch.qclass in ("maj", "dom") else 0.0)
                    tq = "maj" if key.is_major else "min"
                    ton = seg.dist.mass(key.tonic, lambda ch, tq=tq: 1.0 if ch.qclass == tq else 0.0)
                    row[ki] += kc["cadence_bonus"] * dom * ton
            prev = seg
        out.append(row)
    return out


def _viterbi(E: list[list[float]], penalty: float) -> list[int]:
    n, K = len(E), len(ALL_KEYS)
    score = list(E[0])
    back: list[list[int]] = []
    for t in range(1, n):
        best = max(range(K), key=lambda k: score[k])
        new, bp = [], []
        for k in range(K):
            stay, switch = score[k], score[best] - penalty
            if stay >= switch:
                new.append(stay + E[t][k])
                bp.append(k)
            else:
                new.append(switch + E[t][k])
                bp.append(best)
        score = new
        back.append(bp)
    k = max(range(K), key=lambda i: score[i])
    path = [k]
    for bp in reversed(back):
        k = bp[k]
        path.append(k)
    return path[::-1]


def _forward_backward(E: list[list[float]], penalty: float) -> list[list[float]]:
    n, K = len(E), len(ALL_KEYS)

    def step(prev: list[float]) -> list[float]:
        out = []
        for k in range(K):
            others = _lse([prev[j] for j in range(K) if j != k]) - penalty
            out.append(_lse([prev[k], others]))
        return out

    alpha = [list(E[0])]
    for t in range(1, n):
        trans = step(alpha[-1])
        alpha.append([trans[k] + E[t][k] for k in range(K)])
    beta = [[0.0] * K for _ in range(n)]
    for t in range(n - 2, -1, -1):
        nxt = [beta[t + 1][k] + E[t + 1][k] for k in range(K)]
        beta[t] = step(nxt)  # transition matrix is symmetric
    post = []
    for t in range(n):
        g = [alpha[t][k] + beta[t][k] for k in range(K)]
        z = _lse(g)
        post.append([math.exp(x - z) for x in g])
    return post


def _regions(path: list[int]) -> list[list[int]]:
    regs: list[list[int]] = []
    for t, k in enumerate(path):
        if regs and regs[-1][2] == k:
            regs[-1][1] = t + 1
        else:
            regs.append([t, t + 1, k])
    return regs


def _merge_short_regions(path: list[int], E: list[list[float]], segs: list[SegData], min_beats: float) -> list[int]:
    path = list(path)
    while True:
        regs = _regions(path)
        if len(regs) <= 1:
            return path
        short = [(sum(s.beats for s in segs[a:b]), i) for i, (a, b, _) in enumerate(regs)
                 if sum(s.beats for s in segs[a:b]) < min_beats]
        if not short:
            return path
        _, i = min(short)
        a, b, _ = regs[i]
        neighbours = [regs[j][2] for j in (i - 1, i + 1) if 0 <= j < len(regs)]
        best = max(neighbours, key=lambda k: sum(E[t][k] for t in range(a, b)))
        path[a:b] = [best] * (b - a)


def estimate_keys(segs: list[SegData], cfg: dict, given: Key | None = None) -> KeyAnalysis:
    kc = cfg["key"]
    n = len(segs)
    if n == 0:
        raise ValueError("cannot estimate key of an empty progression")
    E = emission_matrix(segs, cfg)

    if given is not None:
        gi = KEY_INDEX[given]
        post = [[1.0 if k == gi else 0.0 for k in range(len(ALL_KEYS))] for _ in range(n)]
        return KeyAnalysis([given] * n, [1.0] * n, post, E, [(0, n, given)], given,
                           [(given, 1.0)], False, "given")

    path = _viterbi(E, kc["switch_penalty"])
    path = _merge_short_regions(path, E, segs, kc["min_region_beats"])
    post = _forward_backward(E, kc["switch_penalty"])
    seg_keys = [ALL_KEYS[k] for k in path]
    seg_probs = [post[t][path[t]] for t in range(n)]
    regions = [(a, b, ALL_KEYS[k]) for a, b, k in _regions(path)]
    # Global key = key holding the most posterior-weighted time (the "home" key of a
    # modulating song is the one it spends longest in).
    total = sum(s.beats for s in segs if s.chord.root is not None) or 1.0
    mass = [sum(post[t][k] * segs[t].beats for t in range(n) if segs[t].chord.root is not None) / total
            for k in range(len(ALL_KEYS))]
    gdist = sorted(((ALL_KEYS[k], mass[k]) for k in range(len(ALL_KEYS))), key=lambda kv: -kv[1])
    # Ambiguity = competing readings of the SAME passage (e.g. C major vs A minor over a
    # vi-centred loop), not time-sharing between keys of a modulating song.
    amb_beats = 0.0
    for t in range(n):
        if segs[t].chord.root is None:
            continue
        p1, p2 = sorted(post[t], reverse=True)[:2]
        if p2 >= kc["ambiguity_ratio"] * p1:
            amb_beats += segs[t].beats
    ambiguous = amb_beats / total > 0.5
    return KeyAnalysis(seg_keys, seg_probs, post, E, regions, gdist[0][0], gdist, ambiguous, "estimated")
