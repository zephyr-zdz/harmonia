"""Song structure (intro / verse / pre-chorus / chorus / bridge / outro) — option A, no new deps.

Works at the BAR level of the meter grid (frontend/meter.py), because pop sections are
built from bars and phrases (4 / 8 / 16 bars).

1. Bar features
   harmony  per-beat chroma, mixing audio chroma (librosa CQT) and the chroma implied by the
            decoded chord posteriors; the m beats of a bar are concatenated (harmonic rhythm).
   timbre   mean MFCC 2–13 + loudness (dB) per bar.
2. Similarity
   repetition  cosine of 2-bar embedded harmony (chord identity: root × major/minor), per
               transposition; sections are compared with ONE shift each (a last chorus a step
               up is still the chorus; transposed matches pay a small penalty).
   homogeneity Gaussian similarity of timbre.
3. Boundaries: Foote novelty on the combined matrix, then a dynamic programme choosing
   section lengths with a phrase prior (multiples of 4 bars preferred, 8 / 16 most).
4. Grouping: segments whose aligned bars repeat (mean repetition similarity ≥ threshold)
   share a cluster (letters A, B, C … in order of appearance).
5. Labels (J-pop form; Aメロ = verse, Bメロ = pre-chorus, サビ = chorus, Cメロ = bridge):
   chorus      repeated cluster with the highest loudness / repetition / late-song presence;
   pre-chorus  repeated cluster that most often directly precedes the chorus;
   verse       repeated cluster that most often precedes the pre-chorus (or the chorus);
   intro / outro  first / last segment when not a verse / chorus occurrence;
   bridge      non-repeated segment after the second chorus (not the last one);
   anything else: "inst" (repeated, not placed) or "other".
   These are heuristics — [UNCERTAIN] until scored against annotated sections.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..theory.chord import ChordParseError, parse_chord

LABELS = ("intro", "verse", "pre-chorus", "chorus", "bridge", "inst", "outro", "other")


@dataclass
class Section:
    start: float
    end: float
    start_bar: int          # index into the analysed bar list (0-based)
    end_bar: int            # exclusive
    label: str
    cluster: str            # repetition group: "A", "B", ...
    shift: int              # semitones UP relative to the cluster's first occurrence (modulation)
    offset: int             # bar k of the first occurrence ↔ bar k + offset of this one
    similarity: float       # mean repetition similarity to the rest of its cluster (0 if unique)


@dataclass
class Structure:
    sections: list[Section]
    bar_times: list[float]
    warnings: list[str] = field(default_factory=list)
    labels_from: str = "A"                     # "A" (rules) or "songformer:<file>" (external labels)
    beat_index: list[list[int]] = field(default_factory=list)   # beats of each analysed bar
    bar_similarity: np.ndarray | None = None   # repetition similarity between bars (not serialised)

    def to_dict(self) -> dict:
        return {"sections": [asdict(s) for s in self.sections], "warnings": self.warnings,
                "labels_from": self.labels_from}

    def label_at(self, t: float) -> str | None:
        for s in self.sections:
            if s.start <= t < s.end:
                return s.label
        return None


# ------------------------------------------------------------------------------ features

def _chord_templates(labels: list[str]) -> np.ndarray:
    T = np.zeros((len(labels), 12))
    for i, l in enumerate(labels):
        try:
            ch = parse_chord(l)
        except ChordParseError:
            continue
        if ch.root is None:
            continue
        for pc in ch.pitch_classes():
            T[i, pc] = 1.0
    return T


def _chord_classes(labels: list[str]) -> np.ndarray:
    """[C, 24] map from vocabulary chords to root × {major-third, minor-third} (N → none)."""
    M = np.zeros((len(labels), 24))
    for i, l in enumerate(labels):
        try:
            ch = parse_chord(l)
        except ChordParseError:
            continue
        if ch.root is None:
            continue
        minor = ch.qclass in ("min", "dim", "hdim")
        M[i, (12 if minor else 0) + ch.root] = 1.0
    return M


def _norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.where(n > 1e-9, n, 1.0)


def beat_features(audio_path: str | Path, edges: np.ndarray, posteriors: np.ndarray, labels: list[str],
                  sr: int = 22050, hop: int = 512) -> dict[str, np.ndarray]:
    import librosa
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    t = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    idx = np.searchsorted(t, edges)
    B = len(edges) - 1
    ch_a, mf, ld = np.zeros((B, 12)), np.zeros((B, 13)), np.zeros(B)
    for b in range(B):
        lo, hi = idx[b], max(idx[b + 1], idx[b] + 1)
        hi = min(hi, chroma.shape[1])
        lo = min(lo, hi - 1)
        ch_a[b] = chroma[:, lo:hi].mean(axis=1)
        mf[b] = mfcc[:, lo:hi].mean(axis=1)
        ld[b] = 20 * np.log10(rms[lo:hi].mean() + 1e-6)
    ch_c = posteriors @ _chord_templates(labels)
    chroma = 0.5 * _norm(ch_a) + 0.5 * _norm(ch_c)
    ident = posteriors @ _chord_classes(labels)          # P(root, major/minor) per beat
    return {"chroma": chroma, "ident": ident, "mfcc": mf, "loud": ld}


# ------------------------------------------------------------------------------ bars

def bar_matrix(feats: dict[str, np.ndarray], bars: list[int | None], m: int, ident_weight: float = 1.0):
    """Group beats into bars → (bar ids, beat index lists, harmony [n, m, 3, 12], timbre [n, d], loudness [n]).
    Harmony per beat = 3 pitch-class blocks (major-chord root, minor-chord root, chroma), so a
    transposition is one roll of the last axis. Chord identity is weighted by ``ident_weight``."""
    order: list[int] = []
    beats: dict[int, list[int]] = {}
    for b, bar in enumerate(bars):
        if bar is None:
            continue
        if bar not in beats:
            order.append(bar)
            beats[bar] = []
        beats[bar].append(b)
    per_beat = np.concatenate([ident_weight * feats["ident"].reshape(-1, 2, 12),
                               (1 - ident_weight) * feats["chroma"][:, None, :]], axis=1)   # [B, 3, 12]
    harm = np.zeros((len(order), m, 3, 12))
    for i, bar in enumerate(order):
        bb = beats[bar]
        for k in range(m):  # short (pickup / last) bars: repeat their last beat
            harm[i, k] = per_beat[bb[min(k, len(bb) - 1)]]
    mf = np.array([feats["mfcc"][beats[bar], 1:].mean(axis=0) for bar in order])
    ld = np.array([feats["loud"][beats[bar]].mean() for bar in order])
    mf = (mf - mf.mean(axis=0)) / (mf.std(axis=0) + 1e-6)
    timbre = np.concatenate([mf, ((ld - ld.mean()) / (ld.std() + 1e-6))[:, None]], axis=1)
    return order, [beats[b] for b in order], harm, timbre, ld


def repetition_matrix(harm: np.ndarray, embed: int) -> np.ndarray:
    """Cosine similarity of `embed`-bar windows under each transposition → S [12, n, n], where
    S[r, i, j] compares bar i with bar j transposed by r semitones. One shift is chosen per
    SECTION pair later (a modulated repeat moves as a whole; choosing it per bar would let any
    major-chord bar match any other)."""
    n = harm.shape[0]
    win = np.stack([np.concatenate([harm[min(i + e, n - 1)] for e in range(embed)], axis=0) for i in range(n)])
    flat = _norm(win.reshape(n, -1))
    out = np.zeros((12, n, n))
    for r in range(12):
        rot = _norm(np.roll(win, r, axis=-1).reshape(n, -1))
        out[r] = np.clip(flat @ rot.T, 0.0, 1.0)
    return out


def novelty(S: np.ndarray, half: int) -> np.ndarray:
    n = S.shape[0]
    k = np.arange(-half, half) + 0.5
    g = np.exp(-0.5 * (k / (0.5 * half)) ** 2)
    K = np.outer(g, g) * np.sign(np.outer(k, k))  # checkerboard: + within, − across
    pad = np.pad(S, half, mode="edge")
    nov = np.array([np.sum(pad[i:i + 2 * half, i:i + 2 * half] * K) for i in range(n)])
    nov = np.maximum(nov, 0.0)
    return nov / (nov.max() + 1e-9)


def segment_bars(nov: np.ndarray, sc: dict) -> list[int]:
    """DP over boundary positions: maximise Σ novelty(boundary) + phrase prior(length) − cost.
    Returns boundary bar indices including 0 and n."""
    n = len(nov)
    prior = {int(k): v for k, v in sc["length_prior"].items()}
    min_len = sc["min_bars"]

    def length_score(L: int) -> float:
        return prior.get(L, sc["prior_multiple_of_4"] if L % 4 == 0 else sc["prior_other"])

    best = np.full(n + 1, -np.inf)
    back = np.zeros(n + 1, dtype=int)
    best[0] = 0.0
    for j in range(1, n + 1):
        for i in range(0, j):
            L = j - i
            if best[i] == -np.inf or (L < min_len and not (i == 0 or j == n)):
                continue
            s = best[i] + length_score(L) + (sc["novelty_weight"] * nov[j] if j < n else 0.0) - sc["boundary_cost"]
            if s > best[j]:
                best[j], back[j] = s, i
    cuts = [n]
    while cuts[-1] > 0:
        cuts.append(int(back[cuts[-1]]))
    return cuts[::-1]


def _seg_similarity(S: np.ndarray, a: tuple[int, int], b: tuple[int, int], slack: int,
                    shift_penalty: float) -> tuple[float, int, int]:
    """Best mean similarity along a diagonal between two sections over bar offsets (±slack) and
    ONE transposition r, scaled by (overlap / longer length)^0.25 → (similarity, r, offset)."""
    best, best_r, best_off = 0.0, 0, 0
    la, lb = a[1] - a[0], b[1] - b[0]
    for off in range(-slack, slack + 1):
        ks = np.array([k for k in range(la) if 0 <= k + off < lb])
        if len(ks) < max(2, min(la, lb) // 2):
            continue
        scale = (len(ks) / max(la, lb)) ** 0.25
        diag = S[:, a[0] + ks, b[0] + ks + off].mean(axis=1) - shift_penalty * (np.arange(12) > 0)
        r = int(np.argmax(diag))
        v = float(diag[r] * scale)
        if v > best:
            best, best_r, best_off = v, r, off
    return best, best_r, best_off


# ------------------------------------------------------------------------------ main

def analyse(harm: np.ndarray, timbre: np.ndarray, loud: np.ndarray, bar_times: list[float], end_time: float,
            cfg: dict, external: list[tuple[float, float, str]] | None = None) -> Structure:
    """``external``: sections from another labeller (SongFormer). Their boundaries (snapped to
    bar lines) and labels are used as is; repetition groups and transposition shifts are still
    computed here (SongFormer gives labels but no grouping or modulation)."""
    sc = cfg["structure"]
    n = harm.shape[0]
    warnings: list[str] = []
    if n < 8:
        return Structure([], bar_times, ["fewer than 8 bars: no structure analysis"])
    S_all = repetition_matrix(harm, sc["embed_bars"])
    S_rep = S_all[0]                      # untransposed: boundaries and pooling
    D = np.linalg.norm(timbre[:, None, :] - timbre[None, :, :], axis=-1)
    sigma = np.median(D[D > 0]) if np.any(D > 0) else 1.0
    S_tim = np.exp(-(D ** 2) / (2 * sigma ** 2))
    S = sc["repetition_weight"] * S_rep + (1 - sc["repetition_weight"]) * S_tim
    nov = novelty(S, sc["kernel_bars"])
    ext_labels: list[str] | None = None
    if external:
        cuts, ext_labels = snap_external(external, bar_times, n)
    else:
        cuts = segment_bars(nov, sc)
    segs = list(zip(cuts[:-1], cuts[1:]))

    # clusters (average linkage over segment similarity)
    k = len(segs)
    sim = np.zeros((k, k))
    shf = np.zeros((k, k), dtype=int)
    offs = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            if i != j:
                sim[i, j], shf[i, j], offs[i, j] = _seg_similarity(S_all, segs[i], segs[j], sc["slack_bars"],
                                                                    sc["shift_penalty"])
    cluster = list(range(k))
    groups = {i: [i] for i in range(k)}
    while True:
        best, pair = sc["cluster_threshold"], None
        keys = sorted(groups)
        for x in range(len(keys)):
            for y in range(x + 1, len(keys)):
                a, b = groups[keys[x]], groups[keys[y]]
                s = float(np.mean([sim[i, j] for i in a for j in b]))
                if s >= best:
                    best, pair = s, (keys[x], keys[y])
        if pair is None:
            break
        a, b = pair
        groups[a] += groups.pop(b)
        for i in groups[a]:
            cluster[i] = a
    letters: dict[int, str] = {}
    for c in cluster:
        if c not in letters:
            letters[c] = chr(ord("A") + len(letters)) if len(letters) < 26 else f"Z{len(letters)}"
    members: dict[int, list[int]] = {}
    for i, c in enumerate(cluster):
        members.setdefault(c, []).append(i)

    labels = ext_labels if ext_labels is not None else _label(segs, cluster, members, loud, sc)
    sections: list[Section] = []
    for i, (a, b) in enumerate(segs):
        mates = [j for j in members[cluster[i]] if j != i]
        first = members[cluster[i]][0]
        sections.append(Section(
            start=bar_times[a], end=bar_times[b] if b < len(bar_times) else end_time, start_bar=a, end_bar=b,
            label=labels[i], cluster=letters[cluster[i]], shift=int(-shf[first, i]) % 12 if i != first else 0,
            offset=int(offs[first, i]) if i != first else 0,
            similarity=round(float(np.mean([sim[i, j] for j in mates])), 3) if mates else 0.0))
    if sections:
        sections[0].start = 0.0  # the first section also covers any pre-roll / pickup
    if not any(s.label == "chorus" for s in sections):
        warnings.append("no repeated section found: chorus not identified")
    return Structure(sections, bar_times, warnings, bar_similarity=S_rep)


_EXTERNAL_LABELS = {"prechorus": "pre-chorus", "pre-chorus": "pre-chorus", "intro": "intro", "verse": "verse",
                    "chorus": "chorus", "bridge": "bridge", "inst": "inst", "outro": "outro"}


def snap_external(external: list[tuple[float, float, str]], bar_times: list[float], n: int
                  ) -> tuple[list[int], list[str]]:
    """External sections → bar cut points + labels. Boundaries move to the nearest bar line;
    "silence" and sections that collapse to zero bars are absorbed by their neighbours; labels
    outside our set become "other"."""
    secs = [(s, e, _EXTERNAL_LABELS.get(l, "other")) for s, e, l in external if l != "silence" and e > s]
    if not secs:
        return [0, n], ["other"]
    bt = np.asarray(bar_times)
    cuts, labs = [0], []
    for k, (s, e, lab) in enumerate(secs):
        if k == 0:
            labs.append(lab)
            continue
        c = int(np.argmin(np.abs(bt - s)))
        if c <= cuts[-1] or c >= n:   # collapsed onto the previous cut: keep the earlier label
            continue
        cuts.append(c)
        labs.append(lab)
    cuts.append(n)
    return cuts, labs


def _label(segs, cluster, members, loud, sc) -> list[str]:
    k = len(segs)
    lab = ["other"] * k
    seg_loud = np.array([loud[a:b].mean() for a, b in segs])
    z = (seg_loud - seg_loud.mean()) / (seg_loud.std() + 1e-6)
    repeated = [c for c, m in members.items() if len(m) >= 2]
    if not repeated:
        return lab

    def chorus_score(c: int) -> float:
        m = members[c]
        late = any(segs[i][0] >= 0.6 * segs[-1][1] for i in m)
        bars = sum(segs[i][1] - segs[i][0] for i in m)
        return float(np.mean(z[m])) + sc["chorus_repeat_weight"] * len(m) + sc["chorus_late_bonus"] * late \
            + 0.01 * bars

    chorus = max(repeated, key=chorus_score)

    def precedes(c: int, target: int) -> int:
        return sum(1 for i in range(1, k) if cluster[i] == target and cluster[i - 1] == c)

    others = [c for c in repeated if c != chorus]
    pre = max(others, key=lambda c: precedes(c, chorus), default=None)
    if pre is not None and precedes(pre, chorus) < 2:
        pre = None
    verse = None
    target = pre if pre is not None else chorus
    cand = [c for c in others if c != pre]
    if cand:
        verse = max(cand, key=lambda c: (precedes(c, target), -min(members[c])))
        if precedes(verse, target) < 1:
            verse = None
    if pre is not None and verse is None:  # only one cluster before the chorus: call it the verse
        verse, pre = pre, None
    # a repeated group that directly follows EVERY occurrence of the chorus (except at the very
    # end) is the chorus's second half (16-bar chorus split into 8 + 8)
    tail = None
    ends = [i for i in range(k - 1) if cluster[i] == chorus and cluster[i + 1] != chorus]
    if len(ends) >= 2:
        nxt = {cluster[i + 1] for i in ends}
        if len(nxt) == 1 and next(iter(nxt)) in repeated and next(iter(nxt)) not in (pre, verse):
            tail = next(iter(nxt))
    for i in range(k):
        c = cluster[i]
        lab[i] = ("chorus" if c in (chorus, tail) else "pre-chorus" if c == pre else "verse" if c == verse
                  else "inst" if c in repeated else "other")
    chorus_idx = [i for i in range(k) if lab[i] == "chorus"]
    if lab[0] in ("other", "inst"):
        lab[0] = "intro"
    elif lab[0] == "chorus" and z[0] < np.mean(z[chorus_idx]) - 0.5:
        lab[0] = "intro"  # chorus theme played softly as an intro
    if k > 1 and lab[-1] in ("other", "inst"):
        lab[-1] = "outro"
    if len(chorus_idx) >= 2:
        for i in range(chorus_idx[1] + 1, k - 1):
            if lab[i] == "other":
                lab[i] = "bridge"
                break
    return lab


def external_sections(audio_path: str | Path, cfg: dict) -> tuple[list[tuple[float, float, str]] | None, str]:
    """Look for ``<audio stem>.songformer.json`` in [structure].external_dirs (relative to the repo)."""
    root = Path(__file__).resolve().parents[2]
    stem = Path(audio_path).stem
    for d in cfg["structure"].get("external_dirs", []):
        f = (root / d if not Path(d).is_absolute() else Path(d)) / f"{stem}.songformer.json"
        if f.is_file():
            import json
            secs = json.loads(f.read_text(encoding="utf-8"))["sections"]
            return [(float(x["start"]), float(x["end"]), str(x["label"])) for x in secs], str(f)
    return None, ""


def from_recognition_inputs(audio_path, edges, bars, beats_per_bar, posteriors, labels, cfg,
                            source_path: str | Path | None = None) -> Structure:
    m = beats_per_bar or 4
    feats = beat_features(audio_path, edges, posteriors, labels)
    order, beat_idx, harm, timbre, loud = bar_matrix(feats, bars, m, cfg["structure"]["identity_weight"])
    bar_times = [float(edges[bi[0]]) for bi in beat_idx]
    external, ext_file = external_sections(source_path or audio_path, cfg)
    st = analyse(harm, timbre, loud, bar_times, float(edges[-1]), cfg, external)
    if external:
        try:
            st.labels_from = "songformer:" + str(Path(ext_file).relative_to(Path(__file__).resolve().parents[2]))
        except ValueError:
            st.labels_from = "songformer:" + ext_file
    st.beat_index = beat_idx
    return st


def pool_posteriors(post: np.ndarray, st: Structure, cfg: dict) -> np.ndarray:
    """Share chord evidence between repetitions of the same section (same cluster, same
    transposition), bar by bar and beat by beat:
        log p'(beat) = (1 − w) log p(beat) + w · mean over repeats of log p(aligned beat),
    using only bar pairs whose repetition similarity ≥ pool_min_similarity. A chorus heard
    three times thus gets three looks at the same chords (recognition errors are partly
    independent between repetitions)."""
    sc = cfg["structure"]
    w = sc["pool_weight"]
    if not w or st.bar_similarity is None or not st.sections:
        return post
    lp = np.log(np.maximum(post, 1e-6))
    out = lp.copy()
    groups: dict[tuple[str, int], list[Section]] = {}
    for s in st.sections:
        groups.setdefault((s.cluster, s.shift), []).append(s)
    for secs in groups.values():
        if len(secs) < 2:
            continue
        for s in secs:
            for k in range(s.end_bar - s.start_bar):
                a = s.start_bar + k
                partners = []
                for o in secs:
                    kb = k - s.offset + o.offset   # same bar of the first occurrence
                    if o is s or not 0 <= kb < o.end_bar - o.start_bar:
                        continue
                    b = o.start_bar + kb
                    if st.bar_similarity[a, b] >= sc["pool_min_similarity"]:
                        partners.append(b)
                if not partners:
                    continue
                for j, beat in enumerate(st.beat_index[a]):
                    others = [lp[st.beat_index[b][j]] for b in partners if j < len(st.beat_index[b])]
                    if others:
                        out[beat] = (1 - w) * lp[beat] + w * np.mean(others, axis=0)
    p = np.exp(out - out.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)

