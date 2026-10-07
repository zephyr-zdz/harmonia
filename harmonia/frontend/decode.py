"""Beat-synchronous chord decoding with a harmonic transition prior.

1. Beat observation: mean frame log-likelihood within each beat interval, scaled by
   ``obs_weight`` (temperature; uncalibrated until tuned on data/eval/dev).
2. Transitions between beats: stay = 0; change a→b = -change_penalty + prior(a→b), where the
   prior is a log-probability over (root interval, quality-class pair) — by default a
   hand-set table (descending-5th root motion and its common cousins favoured), optionally
   replaced by statistics estimated from ChoCo annotations (see transition_prior.py).
3. Forward–backward → per-beat chord posterior → top-k candidates (soft evidence for the
   analysis layer); Viterbi → the decoded path (reported in warnings when it disagrees with
   the per-beat argmax).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..theory.chord import Chord, ChordParseError, parse_chord
from .chords import FrameEvidence

# Hand-set default root-motion log-prior (semitones up from old root to new root).
# Theory: in pop / tonal harmony, root motion by 4th up (= 5th down) dominates, then 2nds
# and 3rds; tritone and semitone motions are rarer (they occur, e.g. tritone subs, ♭VI→V).
DEFAULT_ROOT_MOTION = {0: -1.0, 1: -1.6, 2: -0.8, 3: -1.0, 4: -1.0, 5: -0.3, 6: -1.8,
                       7: -0.7, 8: -1.0, 9: -0.9, 10: -0.8, 11: -1.5}


@dataclass
class BeatDecoding:
    labels: list[str]
    posteriors: np.ndarray     # [B, C]
    path: list[int]            # Viterbi chord index per beat
    bass: np.ndarray | None    # [B, 13]


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def transition_matrix(labels: list[str], dc: dict, prior: dict | None = None) -> np.ndarray:
    chords: list[Chord | None] = []
    for l in labels:
        try:
            chords.append(parse_chord(l))
        except ChordParseError:
            chords.append(None)
    C = len(labels)
    motion = prior.get("root_motion", DEFAULT_ROOT_MOTION) if prior else DEFAULT_ROOT_MOTION
    qpair = prior.get("quality_pair", {}) if prior else {}
    A = np.full((C, C), -dc["change_penalty"], dtype=np.float64)
    for i, a in enumerate(chords):
        for j, b in enumerate(chords):
            if i == j:
                A[i, j] = 0.0
                continue
            if a is None or b is None or a.root is None or b.root is None:
                A[i, j] += dc["no_chord_change"]
                continue
            A[i, j] += motion[(b.root - a.root) % 12]
            A[i, j] += qpair.get(f"{a.qclass}>{b.qclass}", 0.0)
    return A


def beat_observations(ev: FrameEvidence, edges: np.ndarray, dc: dict) -> tuple[np.ndarray, np.ndarray | None]:
    B = len(edges) - 1
    idx = np.searchsorted(ev.times, edges)
    obs = np.zeros((B, ev.loglik.shape[1]))
    bass = np.zeros((B, 13)) if ev.bass is not None else None
    for b in range(B):
        lo, hi = idx[b], max(idx[b + 1], idx[b] + 1)
        hi = min(hi, len(ev.times))
        lo = min(lo, hi - 1)
        ll = ev.loglik[lo:hi]
        ll = np.where(np.isfinite(ll), ll, -50.0)
        obs[b] = dc["obs_weight"] * ll.mean(axis=0)
        if bass is not None:
            bass[b] = ev.bass[lo:hi].mean(axis=0)
    bias = dc.get("seventh_bias", 0.0)
    if bias:
        obs += bias * _seventh_mask(tuple(ev.labels))[None, :]
    return obs, bass


_MASKS: dict[tuple[str, ...], np.ndarray] = {}


def _seventh_mask(labels: tuple[str, ...]) -> np.ndarray:
    """1.0 for vocabulary chords with a 7th. ``seventh_bias`` is a class-prior correction:
    the chord model under-predicts 7ths relative to J-pop practice (maj7(9) -> maj)."""
    if labels not in _MASKS:
        m = np.zeros(len(labels))
        for i, l in enumerate(labels):
            try:
                m[i] = 1.0 if parse_chord(l).has_seventh else 0.0
            except ChordParseError:
                pass
        _MASKS[labels] = m
    return _MASKS[labels]


def metrical_extra(positions: list[int | None] | None, beats_per_bar: int | None, dc: dict, B: int) -> np.ndarray:
    """Extra log-cost of a chord change INTO beat t, by metrical position: bar line 0, half-bar
    ``change_extra_halfbar``, other beats ``change_extra_offbeat`` (pop harmony mostly changes
    on bar lines and half bars)."""
    extra = np.zeros(B)
    if not positions or not beats_per_bar:
        return extra
    # half-bar beat: 4/4 -> 3, 6/8 -> 4, 12/8 -> 7; odd meters (3/4, 5/4) have none
    half = beats_per_bar // 2 + 1 if beats_per_bar % 2 == 0 else None
    for t, p in enumerate(positions[:B]):
        if p is None or p == 1:
            continue
        extra[t] = dc["change_extra_halfbar"] if p == half else dc["change_extra_offbeat"]
    return extra


def decode(ev: FrameEvidence, edges: np.ndarray, dc: dict, prior: dict | None = None,
           positions: list[int | None] | None = None, beats_per_bar: int | None = None) -> BeatDecoding:
    obs, bass = beat_observations(ev, edges, dc)
    A = transition_matrix(ev.labels, dc, prior)
    B, C = obs.shape
    extra = metrical_extra(positions, beats_per_bar, dc, B)
    eye = np.eye(C, dtype=bool)

    def At(t: int) -> np.ndarray:
        if extra[t] == 0:
            return A
        M = A - extra[t]
        M[eye] = A[eye]
        return M
    # forward-backward (log space)
    alpha = np.zeros((B, C))
    alpha[0] = obs[0] - _logsumexp(obs[0], 0)
    for t in range(1, B):
        alpha[t] = obs[t] + _logsumexp(alpha[t - 1][:, None] + At(t), 0)
        alpha[t] -= _logsumexp(alpha[t], 0)
    beta = np.zeros((B, C))
    for t in range(B - 2, -1, -1):
        beta[t] = _logsumexp(At(t + 1) + (obs[t + 1] + beta[t + 1])[None, :], 1)
        beta[t] -= _logsumexp(beta[t], 0)
    g = alpha + beta
    post = np.exp(g - _logsumexp(g, 1)[:, None])
    # Viterbi
    score = obs[0].copy()
    back = np.zeros((B, C), dtype=int)
    for t in range(1, B):
        cand = score[:, None] + At(t)
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(C)] + obs[t]
    path = [int(np.argmax(score))]
    for t in range(B - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return BeatDecoding(ev.labels, post, path[::-1], bass)
